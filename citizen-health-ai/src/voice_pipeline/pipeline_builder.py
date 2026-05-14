"""
Pipecat pipeline builder for each health AI module.
Compatible with pipecat-ai 0.0.108.

Pipeline per session:
  LiveKitTransport.input()              — WebRTC audio in from browser
    → SileroVADAnalyzer (in params)     — detects start/end of speech
    → SarvamSTTService                  — Saarika v2.5  speech → text
    → LLMUserResponseAggregator         — accumulates user turn
    → OpenAILLMService (sarvam-30b)     — LLM inference (streaming)
    → SarvamTTSService                  — Bulbul v2  text → PCM chunks
    → LLMAssistantResponseAggregator    — records assistant turn for history
  LiveKitTransport.output()             — WebRTC audio out to browser

Features:
  • VAD-based turn detection — no hold-to-record button needed
  • Barge-in / interruption (allow_interruptions=True)
  • Streaming TTS — first audio chunk plays while rest is synthesising
  • Full conversation history maintained across turns
"""

import logging

from config.settings import (
    LIVEKIT_URL,
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    SARVAM_LLM_MODEL,
)
from src.citizen_ai.citizen_assistant import CITIZEN_SYSTEM_PROMPT
from src.health_worker_ai.health_worker_assistant import HEALTH_WORKER_SYSTEM_PROMPT
from src.rag.retriever import nhim_retriever

from .room_manager import bot_token
from .sarvam_stt import SarvamSTTService
from .sarvam_tts import SarvamTTSService

logger = logging.getLogger(__name__)

_CITIZEN_SPEAKERS = {"ta-IN": "anushka", "kn-IN": "anushka", "en-IN": "anushka"}
_WORKER_SPEAKERS  = {"ta-IN": "abhilash", "kn-IN": "abhilash", "en-IN": "abhilash"}


def _build_system_prompt(module: str, language_code: str, worker_role: str) -> str:
    lang = language_code.split("-")[0]
    lang_hint = {
        "ta": "Respond in Tamil (தமிழ்). Keep answers brief.",
        "kn": "Respond in Kannada (ಕನ್ನಡ). Keep answers brief.",
        "en": "Respond in English. Keep answers concise.",
    }.get(lang, "")

    if module == "worker":
        role_hint = (
            "You are assisting an ASHA worker doing community home visits."
            if worker_role == "asha"
            else "You are assisting a PHC/CHC nurse at a health facility."
        )
        return f"{HEALTH_WORKER_SYSTEM_PROMPT}\n\n{role_hint}\n{lang_hint}".strip()

    rag = nhim_retriever.format_context("health insurance hospital scheme", top_k=2)
    extra = f"\n\n{rag}" if rag else ""
    return f"{CITIZEN_SYSTEM_PROMPT}{extra}\n\n{lang_hint}".strip()


async def start_pipeline(
    room_name: str,
    module: str = "citizen",
    language_code: str = "en-IN",
    worker_role: str = "asha",
) -> None:
    """
    Build and run a Pipecat pipeline connected to a LiveKit room.

    Called as a FastAPI BackgroundTask — runs until the room empties or
    the pipeline errors out.
    """
    if not LIVEKIT_URL:
        logger.error("LIVEKIT_URL not configured — cannot start RT pipeline")
        return

    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_response import (
            LLMAssistantResponseAggregator,
            LLMUserResponseAggregator,
        )
        from pipecat.services.openai import OpenAILLMService
        from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
    except ImportError as exc:
        logger.error(
            "Pipecat import failed — run: pip install 'pipecat-ai[livekit,silero]'. Error: %s", exc
        )
        return

    logger.info(
        "Starting RT pipeline: room=%s module=%s lang=%s role=%s",
        room_name, module, language_code, worker_role,
    )

    # ── Transport ─────────────────────────────────────────────────────────────
    token = bot_token(room_name)
    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.8,
                    min_volume=0.6,
                )
            ),
        ),
    )

    # ── Services ──────────────────────────────────────────────────────────────
    stt = SarvamSTTService(language_code=language_code)

    speaker_map = _WORKER_SPEAKERS if module == "worker" else _CITIZEN_SPEAKERS
    tts = SarvamTTSService(
        language_code=language_code,
        speaker=speaker_map.get(language_code, "anushka"),
    )

    llm = OpenAILLMService(
        api_key=SARVAM_API_KEY,
        base_url=f"{SARVAM_BASE_URL}/v1",
        model=SARVAM_LLM_MODEL,
    )

    # ── Conversation history aggregators ──────────────────────────────────────
    system_prompt = _build_system_prompt(module, language_code, worker_role)
    messages = [{"role": "system", "content": system_prompt}]

    user_agg      = LLMUserResponseAggregator(messages=messages)
    assistant_agg = LLMAssistantResponseAggregator(messages=messages)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    logger.info("RT pipeline running for room: %s", room_name)
    try:
        await runner.run(task)
    except Exception as exc:
        logger.error("RT pipeline error (room=%s): %s", room_name, exc)
    finally:
        logger.info("RT pipeline ended for room: %s", room_name)
