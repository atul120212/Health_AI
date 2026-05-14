"""
Citizen Health AI — Tamil/Kannada voice assistant.

Covers:
  - Hospital navigation (PHC/CHC/GH locations, OPD timings)
  - Ayushman Bharat PM-JAY & CMCHIS eligibility checking
  - Appointment booking
  - Maternal health reminders (ANC schedule, immunisation)
  - NHM programme queries (JSSK, PMSMA, Pulse Polio, etc.)

Uses Sarvam AI sarvam-30b with tool use + streaming, and Sarvam AI for
Tamil/Kannada TTS/STT so citizens can interact entirely by voice.
"""

import json
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from config.settings import SARVAM_API_KEY, SARVAM_BASE_URL, SARVAM_LLM_MODEL
from src.middleware import hmis_bridge
from src.rag import nhim_retriever
from src.shared.sarvam_client import sarvam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (English — model translates intent; Sarvam handles voice)
# ---------------------------------------------------------------------------

CITIZEN_SYSTEM_PROMPT = """You are "Aarogya Sakhi" (ஆரோக்ய சகி / ಆರೋಗ್ಯ ಸಖಿ), a compassionate AI health
assistant for citizens in Tamil Nadu and Karnataka. You communicate in the
language the user speaks — Tamil, Kannada, or English — switching
automatically based on their messages.

Your role is to help citizens with:
1. **Hospital navigation** — Find nearby Government hospitals, PHCs, CHCs,
   OPD timings, and speciality availability.
2. **Insurance eligibility** — Check Ayushman Bharat PM-JAY and CMCHIS
   eligibility, explain coverage, and list empanelled hospitals.
3. **Appointment booking** — Book OPD slots at government hospitals.
4. **Maternal health** — Track ANC visit schedules, remind about PMSMA
   (Pradhan Mantri Surakshit Matritva Abhiyan) camps, JSSK entitlements.
5. **Immunisation** — Check child immunisation schedule and due dates.
6. **NHM programmes** — Answer questions about Janani Suraksha Yojana,
   Mission Indradhanush, RBSK, Poshan Abhiyaan, and related schemes.

IMPORTANT GUIDELINES:
- Always be warm, respectful, and use simple language.
- Never give medical diagnoses — direct to a doctor for clinical advice.
- For emergencies, immediately provide 108 (ambulance) helpline.
- Confirm bookings aloud before finalising.
- When income or ration card data is missing, ask the user before checking eligibility.
- Respond concisely — citizens may be on a phone IVR with limited patience.

Use the provided tools to look up real data from the health system.
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

CITIZEN_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": (
                "Retrieve a citizen's patient record from HMIS using their mobile number. "
                "Returns patient demographics, pregnancy status, ABHA ID, and ration card details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "10-digit mobile number of the citizen",
                    }
                },
                "required": ["phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_insurance_eligibility",
            "description": (
                "Check whether the citizen is eligible for Ayushman Bharat PM-JAY "
                "and/or CMCHIS health insurance schemes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "HMIS patient ID returned by get_patient_info",
                    },
                    "annual_income": {
                        "type": "number",
                        "description": "Annual household income in INR",
                    },
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointment_slots",
            "description": "Fetch available OPD appointment slots at a government hospital.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hospital_id": {
                        "type": "string",
                        "description": "Hospital identifier (e.g. 'GH-CHN-001')",
                    },
                    "department": {
                        "type": "string",
                        "description": "Medical department (e.g. General, Obstetrics, Paediatrics)",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date for slot search in YYYY-MM-DD format",
                    },
                },
                "required": ["hospital_id", "department", "date_from"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a confirmed OPD appointment for the citizen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "hospital_id": {"type": "string"},
                    "department": {"type": "string"},
                    "slot_datetime": {
                        "type": "string",
                        "description": "ISO 8601 datetime of the chosen slot",
                    },
                    "doctor_name": {"type": "string"},
                },
                "required": [
                    "patient_id",
                    "hospital_id",
                    "department",
                    "slot_datetime",
                    "doctor_name",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anc_schedule",
            "description": "Get the Antenatal Care (ANC) visit schedule for a pregnant patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_immunization_schedule",
            "description": "Get the child immunisation schedule and due/overdue vaccines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "child_id": {"type": "string"}
                },
                "required": ["child_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the appropriate HMIS bridge method."""
    try:
        if tool_name == "get_patient_info":
            result = await hmis_bridge.get_patient_by_phone(tool_input["phone"])
            return json.dumps(result or {"error": "Patient not found"}, default=str)

        if tool_name == "check_insurance_eligibility":
            pid = tool_input["patient_id"]
            pmjay = await hmis_bridge.check_ayushman_eligibility(pid)
            cmchis = await hmis_bridge.check_cmchis_eligibility(pid)
            return json.dumps({"pmjay": pmjay, "cmchis": cmchis}, default=str)

        if tool_name == "get_appointment_slots":
            from datetime import date
            date_from = date.fromisoformat(tool_input["date_from"])
            slots = await hmis_bridge.get_available_slots(
                tool_input["hospital_id"],
                tool_input["department"],
                date_from,
            )
            return json.dumps(slots, default=str)

        if tool_name == "book_appointment":
            from datetime import datetime
            slots_dt = datetime.fromisoformat(tool_input["slot_datetime"])
            result = await hmis_bridge.book_appointment(
                tool_input["patient_id"],
                tool_input["hospital_id"],
                tool_input["department"],
                slots_dt,
                tool_input["doctor_name"],
            )
            return json.dumps(result, default=str)

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

class CitizenAssistant:
    """
    Stateless Sarvam-powered assistant for citizen health queries.

    Each call to `chat()` accepts a full conversation history so callers
    can maintain session state on their side (IVR session dict / portal
    WebSocket context).
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=SARVAM_API_KEY,
            base_url=f"{SARVAM_BASE_URL}/v1",
        )

    async def chat(
        self,
        messages: list[dict],
        language: str = "ta",
        stream: bool = True,
    ) -> str:
        """
        Run a single conversational turn.

        Args:
            messages: Full conversation history in OpenAI message format.
            language: 'ta' (Tamil), 'kn' (Kannada), or 'en' (English).
            stream: Whether to use streaming (recommended for IVR latency).

        Returns:
            Assistant reply as a plain string.
        """
        lang_hint = {
            "ta": "Please respond in Tamil (தமிழ்).",
            "kn": "Please respond in Kannada (ಕನ್ನಡ).",
            "en": "Please respond in English.",
        }.get(language, "")

        # RAG: retrieve relevant NHIM scheme context from the last user message
        last_user_content = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        rag_context = nhim_retriever.format_context(last_user_content, top_k=3)

        system = CITIZEN_SYSTEM_PROMPT
        if rag_context:
            system = f"{system}\n\n{rag_context}"
        system = f"{system}\n\n{lang_hint}".strip()

        # Agentic loop — keeps going until no more tool calls
        loop_messages = list(messages)
        final_text = ""

        while True:
            if stream:
                final_text = await self._stream_turn(system, loop_messages)
                break

            response = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=SARVAM_LLM_MODEL,
                max_tokens=1024,
                messages=[{"role": "system", "content": system}] + loop_messages,  # type: ignore[arg-type]
                tools=CITIZEN_TOOLS,  # type: ignore[arg-type]
            )
            choice = response.choices[0]
            final_text = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                break
            loop_messages = await self._handle_tool_calls(
                loop_messages, choice.message, tool_calls
            )

        return final_text

    async def _stream_turn(self, system: str, messages: list[dict]) -> str:
        """Stream a single model turn, handle tool calls, return final text."""
        collected_text = ""
        tool_calls_map: dict[int, dict] = {}

        stream = await self.client.chat.completions.create(  # type: ignore[call-overload]
            model=SARVAM_LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "system", "content": system}] + messages,  # type: ignore[arg-type]
            tools=CITIZEN_TOOLS,  # type: ignore[arg-type]
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                collected_text += delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_map[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_map[idx]["function"]["arguments"] += tc_delta.function.arguments

        tool_calls = list(tool_calls_map.values())
        if not tool_calls:
            return collected_text

        messages.append({
            "role": "assistant",
            "content": collected_text or None,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            result_str = await _execute_tool(
                tc["function"]["name"],
                json.loads(tc["function"]["arguments"] or "{}"),
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str,
            })

        return await self._stream_turn(system, messages)

    def _parse_response(self, response) -> tuple[str, list]:
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = choice.message.tool_calls or []
        return text, tool_calls

    async def _handle_tool_calls(
        self,
        messages: list[dict],
        assistant_message,
        tool_calls: list,
    ) -> list[dict]:
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result_str = await _execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
        return messages

    # ------------------------------------------------------------------ #
    #  Voice pipeline helpers (IVR path)                                  #
    # ------------------------------------------------------------------ #

    async def process_voice_input(
        self,
        audio_bytes: bytes,
        conversation_history: list[dict],
        language_code: str = "ta-IN",
    ) -> tuple[str, bytes]:
        """
        Full voice round-trip: STT → Sarvam LLM → TTS.

        Returns:
            (transcript, response_audio_wav_bytes)
        """
        # 1. Transcribe citizen's speech
        transcript = await sarvam.speech_to_text(audio_bytes, language_code)
        logger.info("STT transcript (%s): %s", language_code, transcript)

        # 2. Add to history and call the LLM
        lang = language_code.split("-")[0]  # "ta-IN" → "ta"
        conversation_history.append({"role": "user", "content": transcript})
        reply_text = await self.chat(conversation_history, language=lang)

        # 3. Convert reply to speech
        speaker_map = {"ta": "anushka", "kn": "anushka", "en": "anushka"}
        speaker = speaker_map.get(lang, "anushka")
        audio_out = await sarvam.text_to_speech(reply_text, language_code, speaker)

        return transcript, audio_out


# Singleton
citizen_assistant = CitizenAssistant()
