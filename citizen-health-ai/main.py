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
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config.settings import HOST, PORT
from src.citizen_ai import citizen_assistant
from src.health_worker_ai import health_worker_assistant
from src.surveillance_ai import outbreak_detector

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

def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

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
        history = _to_anthropic_messages(req.messages)
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
        history = _to_anthropic_messages(req.messages)
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
