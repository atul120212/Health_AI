"""
Pipecat STTService backed by Sarvam Saarika (saarika:v2.5).
Compatible with pipecat-ai 0.0.108.
"""

import logging
import time
from collections.abc import AsyncGenerator

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language

from src.shared.sarvam_client import sarvam

logger = logging.getLogger(__name__)

# Map Sarvam BCP-47 codes to pipecat Language enum values
_LANG_MAP: dict[str, Language] = {
    "ta-IN": Language.TA_IN,
    "kn-IN": Language.KN_IN,
    "en-IN": Language.EN_IN,
    "hi-IN": Language.HI_IN,
}


class SarvamSTTService(STTService):
    """
    Sarvam Saarika STT as a Pipecat STTService.

    The base class accumulates AudioRawFrames between VAD events and
    passes the complete utterance to run_stt().

    language_code: "unknown" lets Sarvam auto-detect the spoken language.
    Pass "ta-IN", "kn-IN", or "en-IN" to force a specific language.
    """

    def __init__(self, language_code: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self._language_code = language_code

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:  # type: ignore[override]
        if not audio or len(audio) < 1600:   # < ~50 ms at 16 kHz 16-bit — skip noise
            return

        try:
            transcript, detected_lang = await sarvam.speech_to_text(
                audio, self._language_code
            )
        except Exception as exc:
            logger.error("Sarvam STT failed: %s", exc)
            yield ErrorFrame(error=str(exc))
            return

        text = transcript.strip()
        if not text:
            return

        pipecat_lang = _LANG_MAP.get(detected_lang)
        logger.info("STT (%s): %r", detected_lang, text[:120])
        yield TranscriptionFrame(
            text=text,
            user_id="user",
            timestamp=str(time.time()),
            language=pipecat_lang,
        )
