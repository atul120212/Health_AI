"""
Citizen Health AI — FastAPI application entry point.

Endpoints:
  POST /citizen/chat          — text chat (portal widget)
  POST /citizen/voice         — voice round-trip IVR (WAV in → WAV out)
  POST /worker/chat           — ASHA/PHC nurse text query
  POST /worker/voice          — worker voice query
  POST /worker/voice-update   — voice-driven patient record update
  POST /surveillance/analyse  — outbreak analysis for a district+disease
  POST /surveillance/scan     — scan all priority diseases for a district
  GET  /health                — health check

  Real-time (Pipecat + LiveKit):
  POST /rt/token              — create LiveKit room + return user JWT
  GET  /rt/rooms              — list active RT rooms (debug)
  POST /rt/end/{room_name}    — terminate an RT room/pipeline
"""

import base64
import logging
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path when running as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import HOST, LIVEKIT_URL, PORT
from src.citizen_ai import citizen_assistant
from src.health_worker_ai import health_worker_assistant
from src.ivr import session_manager
from src.shared.sarvam_client import sarvam
from src.surveillance_ai import outbreak_detector
from src.voice_pipeline import room_manager, start_pipeline
from src.voice_pipeline.room_manager import (
    create_room,
    new_room_name,
    user_token as lk_user_token,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Citizen Health AI",
    description=(
        "Tamil/Kannada AI health assistant for citizens, ASHA workers, "
        "and disease surveillance officers."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static IVR interface at /static/
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    language: str = Field("ta", pattern="^(ta|kn|en)$")


class ChatResponse(BaseModel):
    reply: str


class WorkerChatRequest(BaseModel):
    messages: list[Message]
    language: str = Field("ta", pattern="^(ta|kn|en)$")
    worker_role: str = Field("asha", pattern="^(asha|nurse)$")


class SurveillanceRequest(BaseModel):
    district: str
    disease: str
    weeks: int = 52


class SurveillanceScanRequest(BaseModel):
    district: str
    diseases: list[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ---------------------------------------------------------------------------
# Root & health check
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/ivr.html")


@app.get("/docs-ui", include_in_schema=False)
async def docs_ui():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "citizen-health-ai"}


# ---------------------------------------------------------------------------
# Citizen AI — text (portal widget)
# ---------------------------------------------------------------------------

@app.post("/citizen/chat", response_model=ChatResponse)
async def citizen_chat(req: ChatRequest):
    """
    Portal widget: text-based citizen health query in Tamil/Kannada/English.
    """
    try:
        history = _to_messages(req.messages)
        reply = await citizen_assistant.chat(history, language=req.language)
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("citizen/chat error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Citizen AI — voice (IVR)
# ---------------------------------------------------------------------------

@app.post(
    "/citizen/voice",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def citizen_voice(
    audio: UploadFile = File(..., description="WAV audio from IVR"),
    language_code: str = Form("ta-IN"),
    conversation_json: str = Form("[]"),
):
    """
    IVR endpoint: accept citizen WAV speech, return WAV audio reply.

    `conversation_json` is a JSON-encoded list of prior {role, content} turns.
    """
    import json

    try:
        audio_bytes = await audio.read()
        history = json.loads(conversation_json)
        _, audio_out = await citizen_assistant.process_voice_input(
            audio_bytes, history, language_code
        )
        return Response(content=audio_out, media_type="audio/wav")
    except Exception as exc:
        logger.exception("citizen/voice error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health Worker AI — text
# ---------------------------------------------------------------------------

@app.post("/worker/chat", response_model=ChatResponse)
async def worker_chat(req: WorkerChatRequest):
    """
    ASHA worker or PHC nurse text query (protocol lookup, referral, etc.).
    """
    try:
        history = _to_messages(req.messages)
        reply = await health_worker_assistant.chat(
            history, language=req.language, worker_role=req.worker_role
        )
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("worker/chat error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health Worker AI — voice
# ---------------------------------------------------------------------------

@app.post(
    "/worker/voice",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def worker_voice(
    audio: UploadFile = File(...),
    language_code: str = Form("ta-IN"),
    worker_role: str = Form("asha"),
    conversation_json: str = Form("[]"),
):
    """
    Voice query for field workers. Returns WAV audio response.
    """
    import json

    try:
        audio_bytes = await audio.read()
        history = json.loads(conversation_json)
        _, audio_out = await health_worker_assistant.process_voice_query(
            audio_bytes, history, language_code, worker_role
        )
        return Response(content=audio_out, media_type="audio/wav")
    except Exception as exc:
        logger.exception("worker/voice error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/worker/voice-update")
async def worker_voice_update(
    audio: UploadFile = File(...),
    patient_id: str = Form(...),
    language_code: str = Form("ta-IN"),
):
    """
    ASHA worker speaks field observations → structured HMIS record update.

    Returns the extracted fields and HMIS confirmation.
    """
    try:
        audio_bytes = await audio.read()
        result = await health_worker_assistant.transcribe_and_structure_record_update(
            audio_bytes, patient_id, language_code
        )
        return result
    except Exception as exc:
        logger.exception("worker/voice-update error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Surveillance AI
# ---------------------------------------------------------------------------

@app.post("/surveillance/analyse")
async def surveillance_analyse(req: SurveillanceRequest):
    """
    Run outbreak analysis for a single district + disease.

    Returns status (outbreak / pre_alert / normal), z-score,
    DHO alert narrative, and weekly trend data.
    """
    try:
        result = await outbreak_detector.analyse_district(
            req.district, req.disease, req.weeks
        )
        return result
    except Exception as exc:
        logger.exception("surveillance/analyse error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/surveillance/scan")
async def surveillance_scan(req: SurveillanceScanRequest):
    """
    Scan all priority diseases for a district and return only alerts
    (outbreak or pre_alert), sorted by z-score descending.
    """
    try:
        alerts = await outbreak_detector.scan_all_priority_diseases(
            req.district, req.diseases
        )
        return {"district": req.district, "alerts": alerts, "total": len(alerts)}
    except Exception as exc:
        logger.exception("surveillance/scan error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# IVR — Interactive Voice Response endpoints
# ---------------------------------------------------------------------------
# All three assistants share this IVR layer.
# STT:  Sarvam Saarika (saarika:v2) — all modules
# TTS:  Sarvam Bulbul  (bulbul:v2)  — Citizen AI module
# RAG:  NHIM knowledge base injected into Citizen AI system prompt

_IVR_GREETINGS = {
    "ta-IN": (
        "வணக்கம்! நான் ஆரோக்ய சகி. மருத்துவமனை, காப்பீடு, தடுப்பூசி, "
        "தாய்மை சுகாதாரம் பற்றி உங்களுக்கு உதவலாம். கேளுங்கள்."
    ),
    "kn-IN": (
        "ನಮಸ್ಕಾರ! ನಾನು ಆರೋಗ್ಯ ಸಖಿ. ಆಸ್ಪತ್ರೆ, ವಿಮೆ, ಲಸಿಕೆ, "
        "ತಾಯಿ ಆರೋಗ್ಯ ಸೇವೆಗಳ ಬಗ್ಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ಕೇಳಿ."
    ),
    "en-IN": (
        "Hello! I am Aarogya Sakhi, your health assistant. "
        "I can help with hospitals, insurance, vaccination, and maternal health. "
        "How can I help you today?"
    ),
}

_WORKER_GREETINGS = {
    "ta-IN": "வணக்கம்! நான் ஸ்வஸ்தா சகாயக். NHM நெறிமுறைகள், நோயாளி பதிவுகள், பரிந்துரை ஆகியவற்றில் உதவலாம்.",
    "kn-IN": "ನಮಸ್ಕಾರ! ನಾನು ಸ್ವಸ್ಥ ಸಹಾಯಕ. NHM ಮಾರ್ಗದರ್ಶಿ, ರೋಗಿ ದಾಖಲೆ, ರೆಫರಲ್ ಸಹಾಯ ಮಾಡಲು ಇದ್ದೇನೆ.",
    "en-IN": "Hello! I am Swastha Sahayak. I can assist with NHM protocols, patient records, and referral guidance.",
}

_TTS_SPEAKERS = {"ta-IN": "anushka", "kn-IN": "anushka", "en-IN": "anushka"}


@app.post("/ivr/start", tags=["IVR"])
async def ivr_start(
    language_code: str = Form("ta-IN"),
    module: str = Form("citizen"),        # citizen | worker
    worker_role: str = Form("asha"),      # asha | nurse
):
    """
    Start a new IVR session.

    Returns session_id, greeting text, and Bulbul TTS audio (base64 WAV).
    """
    session = session_manager.create(
        language_code=language_code,
        module=module,
        worker_role=worker_role,
    )

    greetings = _IVR_GREETINGS if module == "citizen" else _WORKER_GREETINGS
    greeting_text = greetings.get(language_code, greetings["en-IN"])

    speaker = _TTS_SPEAKERS.get(language_code, "anushka")
    audio_bytes = await sarvam.text_to_speech(greeting_text, language_code, speaker)

    return {
        "session_id": session.session_id,
        "language_code": language_code,
        "module": module,
        "greeting_text": greeting_text,
        "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else "",
    }


@app.post("/ivr/speak", tags=["IVR"])
async def ivr_speak(
    session_id: str = Form(...),
    audio: UploadFile = File(..., description="User voice recording (WAV/WebM)"),
):
    """
    Process one voice turn in an IVR session.

    Pipeline:
      Saarika STT → RAG-augmented LLM (citizen) or plain LLM (worker) → Bulbul TTS

    Returns transcript, AI reply text, and TTS audio (base64 WAV).
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired. Call /ivr/start.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio received.")

    # ── Step 1: Sarvam Saarika STT ────────────────────────────────────────
    try:
        transcript, detected_lang = await sarvam.speech_to_text(audio_bytes, session.language_code)
    except Exception as exc:
        logger.exception("STT error")
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {exc}")

    # Use detected language for LLM + TTS so the response matches what was spoken
    if detected_lang and detected_lang != session.language_code:
        logger.info("Language switch: %s → %s (auto-detected)", session.language_code, detected_lang)
        session.language_code = detected_lang

    if not transcript or not transcript.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio — speak clearly and try again.")

    # ── Step 2: LLM (RAG context auto-injected for citizen module) ────────
    session.history.append({"role": "user", "content": transcript})
    try:
        if session.module == "worker":
            reply = await health_worker_assistant.chat(
                session.history, language=session.lang, worker_role=session.worker_role
            )
        else:
            reply = await citizen_assistant.chat(session.history, language=session.lang)
    except Exception as exc:
        session.history.pop()          # remove failed user turn
        logger.exception("LLM error")
        raise HTTPException(status_code=502, detail=f"LLM inference failed: {exc}")

    session.history.append({"role": "assistant", "content": reply})

    # ── Step 3: Sarvam Bulbul TTS (graceful degradation) ─────────────────
    # TTS text limit is 1500 chars for bulbul:v2; _clean_for_tts runs inside
    # sarvam.text_to_speech, so truncate the raw reply here as a safety net.
    tts_text = reply if len(reply) <= 1500 else reply[:1497] + "."
    speaker = _TTS_SPEAKERS.get(session.language_code, "anushka")
    try:
        audio_out = await sarvam.text_to_speech(tts_text, session.language_code, speaker)
    except Exception as exc:
        logger.warning("TTS failed (%s) — returning text-only response", exc)
        audio_out = b""               # degrade gracefully; client shows text

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "audio_b64": base64.b64encode(audio_out).decode() if audio_out else "",
        "turn_count": session.turn_count,
        "language_code": session.language_code,
    }


@app.get("/ivr/session/{session_id}", tags=["IVR"])
async def ivr_session_info(session_id: str):
    """Get current session state and full conversation history."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {
        "session_id": session_id,
        "language_code": session.language_code,
        "module": session.module,
        "worker_role": session.worker_role,
        "turn_count": session.turn_count,
        "history": session.history,
    }


@app.post("/ivr/end/{session_id}", tags=["IVR"])
async def ivr_end(session_id: str):
    """Terminate an IVR session and free its resources."""
    session_manager.end(session_id)
    return {"status": "ended", "session_id": session_id}


@app.get("/ivr/sessions", tags=["IVR"])
async def ivr_list_sessions():
    """List all active IVR sessions (debug/monitoring)."""
    return {"sessions": session_manager.list_active()}


# ---------------------------------------------------------------------------
# Real-time voice  —  Pipecat pipeline + LiveKit WebRTC transport
# ---------------------------------------------------------------------------
# Architecture:
#   Browser ←── WebRTC audio ──→ LiveKit room
#                                      ↕
#                       Pipecat pipeline (server-side bot participant):
#                       Saarika STT → sarvam-30b LLM → Bulbul TTS
#
# Benefits over the classic /ivr/* endpoints:
#   • No hold-to-record; Silero VAD detects end-of-turn automatically
#   • Barge-in / interruption support
#   • Sub-second latency (streaming audio over WebRTC)
#   • Browser auto-plays the bot's voice; no base64 polling needed


class RTSessionRequest(BaseModel):
    module: str = Field("citizen", pattern="^(citizen|worker)$")
    language_code: str = Field("en-IN", pattern="^(ta-IN|kn-IN|en-IN)$")
    worker_role: str = Field("asha", pattern="^(asha|nurse)$")


@app.post("/rt/token", tags=["RealTime"])
async def rt_token(req: RTSessionRequest, background_tasks: BackgroundTasks):
    """
    Create a LiveKit room and return the user's JWT.

    The caller should:
      1. Connect to LiveKit using the returned `livekit_url` and `token`.
      2. Enable the microphone — Silero VAD handles turn detection.
      3. Subscribe to remote audio tracks — the bot's speech auto-plays.

    A Pipecat pipeline starts in the background, joins the same room as a
    bot participant, and drives the full STT → LLM → TTS flow.
    """
    if not LIVEKIT_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "Real-time mode is disabled: LIVEKIT_URL is not configured. "
                "Use the classic /ivr/* endpoints or set LIVEKIT_URL in .env."
            ),
        )

    room_name = new_room_name(req.module)

    # Pre-create the room (optional — LiveKit auto-creates on first join)
    await create_room(room_name)

    # Register for monitoring
    room_manager.register(room_name, req.module, req.language_code)

    # Generate user JWT
    participant_id = f"user-{uuid.uuid4().hex[:6]}"
    token = lk_user_token(room_name, participant_id)

    if not token:
        raise HTTPException(status_code=500, detail="Failed to generate LiveKit token")

    # Launch Pipecat pipeline in background (joins same room as bot)
    background_tasks.add_task(
        start_pipeline,
        room_name=room_name,
        module=req.module,
        language_code=req.language_code,
        worker_role=req.worker_role,
    )

    logger.info(
        "RT session started: room=%s module=%s lang=%s participant=%s",
        room_name, req.module, req.language_code, participant_id,
    )

    return {
        "room_name": room_name,
        "token": token,
        "livekit_url": LIVEKIT_URL,
        "participant_id": participant_id,
        "module": req.module,
        "language_code": req.language_code,
    }


@app.post("/rt/end/{room_name}", tags=["RealTime"])
async def rt_end(room_name: str):
    """Unregister an RT room (the Pipecat pipeline self-terminates when the room empties)."""
    room_manager.unregister(room_name)
    return {"status": "unregistered", "room_name": room_name}


@app.get("/rt/rooms", tags=["RealTime"])
async def rt_list_rooms():
    """List active real-time rooms (debug/monitoring)."""
    return {
        "livekit_configured": bool(LIVEKIT_URL),
        "rooms": room_manager.list_active(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
