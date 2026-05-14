"""
Pipecat TTSService backed by Sarvam Bulbul (bulbul:v2).
Compatible with pipecat-ai 0.0.108.
"""

import io
import logging
import wave
from collections.abc import AsyncGenerator

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

from src.shared.sarvam_client import _clean_for_tts, sarvam

logger = logging.getLogger(__name__)

_TTS_MAX_CHARS = 1500       # Bulbul v2 character limit
_CHUNK_SAMPLES = 22050      # 1-second chunk at 22 050 Hz
_CHUNK_BYTES   = _CHUNK_SAMPLES * 2  # 16-bit mono = 2 bytes/sample


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Extract raw PCM and format info from a WAV blob."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()


class SarvamTTSService(TTSService):
    """
    Sarvam Bulbul TTS as a Pipecat TTSService.

    The base TTSService receives TextFrames, calls run_tts(), and wraps
    the output with TTSStartedFrame / TTSStoppedFrame bookending.

    language_code: BCP-47, e.g. "ta-IN", "kn-IN", "en-IN"
    speaker      : Bulbul voice — anushka (F), abhilash (M), manisha, vidya, arya, karun, hitesh
    pace         : speech rate multiplier (0.5 – 2.0)
    """

    def __init__(
        self,
        language_code: str = "en-IN",
        speaker: str = "anushka",
        pace: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._language_code = language_code
        self._speaker = speaker
        self._pace = pace

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:  # type: ignore[override]
        clean = _clean_for_tts(text)
        if not clean:
            return

        if len(clean) > _TTS_MAX_CHARS:
            clean = clean[:_TTS_MAX_CHARS - 1] + "."

        try:
            wav_bytes = await sarvam.text_to_speech(
                clean, self._language_code, self._speaker, self._pace
            )
        except Exception as exc:
            logger.error("Sarvam TTS failed: %s", exc)
            yield ErrorFrame(error=str(exc))
            return

        if not wav_bytes:
            logger.warning("TTS: empty audio returned")
            return

        try:
            pcm, sample_rate, num_channels = _wav_to_pcm(wav_bytes)
        except Exception as exc:
            logger.error("TTS WAV decode failed: %s", exc)
            yield ErrorFrame(error=str(exc))
            return

        logger.info(
            "TTS: %d chars → %d PCM bytes @ %d Hz %dch",
            len(clean), len(pcm), sample_rate, num_channels,
        )

        for offset in range(0, len(pcm), _CHUNK_BYTES):
            yield TTSAudioRawFrame(
                audio=pcm[offset : offset + _CHUNK_BYTES],
                sample_rate=sample_rate,
                num_channels=num_channels,
                context_id=context_id,
            )
