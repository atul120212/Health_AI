"""
Health Worker AI — GenAI assistant for ASHA workers and PHC nurses.

Capabilities:
  - Clinical protocol lookup (NHM guidelines, IMNCI, ANC protocols)
  - Voice-driven patient record updates (field worker → HMIS)
  - Referral guidance (condition-based facility selection)
  - Drug dosage and supply chain queries
  - Real-time decision support during home visits

Uses Claude claude-opus-4-7 with tool use + streaming. Sarvam AI provides
Tamil/Kannada voice I/O for field workers with limited literacy.
"""

import json
import logging

import anthropic

from config.settings import CLAUDE_MODEL, ANTHROPIC_API_KEY
from src.middleware import hmis_bridge
from src.shared.sarvam_client import sarvam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

HEALTH_WORKER_SYSTEM_PROMPT = """You are "Swastha Sahayak" (ஸ்வஸ்தா சகாயக் / ಸ್ವಸ್ಥ ಸಹಾಯಕ), an AI
assistant for ASHA workers and PHC nurses under the National Health Mission.

You support frontline health workers with:

1. **Protocol lookup** — NHM clinical guidelines, IMNCI (Integrated
   Management of Neonatal and Childhood Illness), ANC protocols, RMNCH+A,
   immunisation schedule norms, and disease management protocols.

2. **Patient record updates** — Help workers document home visit observations
   into structured fields (weight, BP, symptoms, referral decision) and
   submit to HMIS via voice.

3. **Referral guidance** — Recommend the appropriate facility level
   (sub-centre → PHC → CHC → District Hospital → Tertiary) based on the
   patient's condition and urgency, and retrieve nearest facility details.

4. **Drug and supply** — Dosage guidance for IFA, ORS, Zinc, Cotrimoxazole,
   Oxytocin, and other NHM essential medicines; flag contraindications.

5. **Danger sign alerts** — For IMNCI danger signs, obstetric emergencies
   (PPH, eclampsia), or grade-3 SAM, immediately flag for emergency
   referral and provide 108 ambulance instructions.

IMPORTANT GUIDELINES:
- You are assisting a trained health worker, NOT diagnosing patients directly.
- Use the worker's local language (Tamil or Kannada) for field visit queries.
- For clinical decisions, cite the specific NHM/MOHFW guideline chapter.
- Never override a doctor's written prescription.
- Worker safety: if a field visit poses a security risk, advise accordingly.
- Keep responses brief — workers are often in the field with poor connectivity.

Use the tools to look up patient records, submit updates, and find referral
facilities.
"""

# NHM protocol knowledge base (structured summaries — extend with real docs)
NHM_PROTOCOLS = {
    "anc_protocol": """ANC Protocol (NHM):
- ANC-1: ≤12 weeks — BP, weight, Hb, blood group, urine albumin+sugar, HIV/VDRL, TT-1, IFA start
- ANC-2: 14–26 weeks — BP, weight, fundal height, foetal heart, TT-2, IFA continue
- ANC-3: 28–34 weeks — above + ultrasound if not done, PMSMA camp
- ANC-4: 36 weeks onwards — presentation, engagement, birth preparedness plan
Danger signs: BP >140/90, severe oedema, absent foetal movement, bleeding → immediate referral""",

    "imnci_danger_signs": """IMNCI General Danger Signs (refer immediately):
- Not able to drink/breastfeed
- Vomits everything
- Convulsions
- Lethargic/unconscious
→ CLASSIFY as VERY SEVERE DISEASE → refer urgently to hospital, give first dose IM antibiotic""",

    "sam_management": """Severe Acute Malnutrition (SAM):
- Grade 3: MUAC <11.5cm or WFH <-3SD or bilateral pitting oedema
- NRC referral criteria: SAM with any medical complication, SAM in <6 months, SAM with poor appetite
- RUTF dose: 100 kcal/kg/day — 92g RUTF per kg per week
- Follow-up: weekly for 8 weeks at NRC, then monthly at AWC""",

    "ors_zinc": """ORS + Zinc Protocol (Diarrhoea):
- ORS: 50–100 mL after each loose stool (child <2y: 50–100 mL; >2y: 100–200 mL)
- Zinc: 20 mg/day for 14 days (>6 months); 10 mg/day (2–6 months)
- Continue feeding; no antibiotics unless bloody diarrhoea/cholera suspicion
- Refer if: >3 days, blood in stool, sunken eyes, not drinking, lethargic""",
}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

HEALTH_WORKER_TOOLS: list[anthropic.types.ToolParam] = [
    {
        "name": "get_patient_record",
        "description": (
            "Retrieve a patient's HMIS record for review during a home visit "
            "or health facility encounter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "Patient's mobile number",
                }
            },
            "required": ["phone"],
        },
    },
    {
        "name": "update_patient_record",
        "description": (
            "Submit a structured patient record update to HMIS from a field "
            "visit or PHC encounter. Include only the fields being updated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "update_payload": {
                    "type": "object",
                    "description": (
                        "Key-value pairs of fields to update, e.g. "
                        '{"weight_kg": 52.5, "bp_systolic": 118, "visit_notes": "..."}'
                    ),
                },
            },
            "required": ["patient_id", "update_payload"],
        },
    },
    {
        "name": "get_referral_facilities",
        "description": (
            "Find the nearest appropriate referral facilities for a given "
            "condition and urgency level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "district": {"type": "string"},
                "condition": {
                    "type": "string",
                    "description": "Medical condition or speciality needed (e.g. obstetrics, SAM, trauma)",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["emergency", "urgent", "routine"],
                    "description": "Urgency level for referral",
                },
            },
            "required": ["district", "condition"],
        },
    },
    {
        "name": "lookup_protocol",
        "description": "Look up a clinical protocol or guideline from the NHM knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {
                "protocol_key": {
                    "type": "string",
                    "enum": list(NHM_PROTOCOLS.keys()),
                    "description": "Protocol identifier",
                }
            },
            "required": ["protocol_key"],
        },
    },
    {
        "name": "get_anc_schedule",
        "description": "Get the ANC visit schedule and completion status for a pregnant patient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"}
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_immunization_schedule",
        "description": "Get immunisation schedule with due/overdue/given status for a child.",
        "input_schema": {
            "type": "object",
            "properties": {
                "child_id": {"type": "string"}
            },
            "required": ["child_id"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def _execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "get_patient_record":
            record = await hmis_bridge.get_patient_by_phone(tool_input["phone"])
            return json.dumps(record or {"error": "Patient not found"}, default=str)

        if tool_name == "update_patient_record":
            result = await hmis_bridge.update_patient_record(
                tool_input["patient_id"], tool_input["update_payload"]
            )
            return json.dumps(result, default=str)

        if tool_name == "get_referral_facilities":
            facilities = await hmis_bridge.get_referral_facilities(
                tool_input["district"],
                tool_input["condition"],
                tool_input.get("urgency", "routine"),
            )
            return json.dumps(facilities, default=str)

        if tool_name == "lookup_protocol":
            key = tool_input["protocol_key"]
            return NHM_PROTOCOLS.get(key, f"Protocol '{key}' not found in local knowledge base.")

        if tool_name == "get_anc_schedule":
            schedule = await hmis_bridge.get_anc_schedule(tool_input["patient_id"])
            return json.dumps(schedule, default=str)

        if tool_name == "get_immunization_schedule":
            schedule = await hmis_bridge.get_immunization_schedule(tool_input["child_id"])
            return json.dumps(schedule, default=str)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Core assistant
# ---------------------------------------------------------------------------

class HealthWorkerAssistant:
    """
    GenAI assistant for ASHA workers and PHC nurses.

    Accepts full conversation history per turn (stateless service layer).
    Supports both voice (via Sarvam) and text I/O.
    """

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def chat(
        self,
        messages: list[dict],
        language: str = "ta",
        worker_role: str = "asha",
    ) -> str:
        """
        Process a health worker query and return the AI response.

        Args:
            messages: Full conversation history.
            language: 'ta', 'kn', or 'en'.
            worker_role: 'asha' or 'nurse' — affects response depth.
        """
        lang_hint = {
            "ta": "Respond in Tamil (தமிழ்). Use simple language.",
            "kn": "Respond in Kannada (ಕನ್ನಡ). Use simple language.",
            "en": "Respond in English.",
        }.get(language, "")

        role_hint = (
            "You are assisting an ASHA worker doing community home visits."
            if worker_role == "asha"
            else "You are assisting a PHC/CHC nurse at a health facility."
        )

        system = f"{HEALTH_WORKER_SYSTEM_PROMPT}\n\n{role_hint}\n{lang_hint}".strip()

        loop_messages = list(messages)

        while True:
            async with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1200,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=HEALTH_WORKER_TOOLS,
                messages=loop_messages,
            ) as stream:
                collected_text = ""
                tool_use_blocks: list[dict] = []
                current_tool: dict | None = None
                current_input_json = ""

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                            }
                            current_input_json = ""
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            collected_text += delta.text
                        elif delta.type == "input_json_delta" and current_tool:
                            current_input_json += delta.partial_json
                    elif event.type == "content_block_stop":
                        if current_tool:
                            current_tool["input"] = json.loads(current_input_json or "{}")
                            tool_use_blocks.append(current_tool)
                            current_tool = None
                            current_input_json = ""

                final_message = await stream.get_final_message()

            if not tool_use_blocks:
                return collected_text

            # Execute tools and loop
            tool_results = []
            for tool in tool_use_blocks:
                result_str = await _execute_tool(tool["name"], tool["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": result_str,
                    }
                )

            loop_messages.append({"role": "assistant", "content": final_message.content})
            loop_messages.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------ #
    #  Voice pipeline for field workers                                   #
    # ------------------------------------------------------------------ #

    async def process_voice_query(
        self,
        audio_bytes: bytes,
        conversation_history: list[dict],
        language_code: str = "ta-IN",
        worker_role: str = "asha",
    ) -> tuple[str, bytes]:
        """
        Voice round-trip: STT → Claude → TTS.

        Returns:
            (transcript, audio_response_wav)
        """
        transcript = await sarvam.speech_to_text(audio_bytes, language_code)
        logger.info("Worker STT (%s, %s): %s", worker_role, language_code, transcript)

        lang = language_code.split("-")[0]
        conversation_history.append({"role": "user", "content": transcript})
        reply = await self.chat(conversation_history, language=lang, worker_role=worker_role)

        speaker_map = {"ta-IN": "meera", "kn-IN": "amol"}
        speaker = speaker_map.get(language_code, "meera")
        audio_out = await sarvam.text_to_speech(reply, language_code, speaker)

        return transcript, audio_out

    async def transcribe_and_structure_record_update(
        self,
        audio_bytes: bytes,
        patient_id: str,
        language_code: str = "ta-IN",
    ) -> dict:
        """
        Specialised flow: worker speaks observations → extract structured
        fields → submit to HMIS.

        Returns the HMIS update confirmation dict.
        """
        transcript = await sarvam.speech_to_text(audio_bytes, language_code)
        lang = language_code.split("-")[0]

        extraction_prompt = f"""Extract structured patient visit data from this {lang} field note.
Return a JSON object with any of these keys (only include what's mentioned):
weight_kg, height_cm, bp_systolic, bp_diastolic, temperature_c, pulse_bpm,
fundal_height_cm, foetal_heart_rate, oedema_grade, haemoglobin_g_dl,
urine_albumin, urine_sugar, visit_notes, danger_signs, referral_needed,
next_visit_date.

Field note: {transcript}

Return ONLY valid JSON, no explanation."""

        response = await self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": extraction_prompt}],
        )
        raw = response.content[0].text.strip()

        try:
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            update_payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse structured update from: %s", raw)
            update_payload = {"visit_notes": transcript}

        result = await hmis_bridge.update_patient_record(patient_id, update_payload)
        return {"transcript": transcript, "extracted_fields": update_payload, "hmis_result": result}


# Singleton
health_worker_assistant = HealthWorkerAssistant()
