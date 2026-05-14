"""
Sarvam AI client wrapper for Tamil/Kannada TTS and STT.

Sarvam AI provides:
  - Speech-to-Text (Indic languages including Tamil & Kannada)
  - Text-to-Speech (natural Indic voices)
  - Translation (Indic ↔ English)
  - Transliteration

This client wraps those APIs for use in IVR and portal widget flows.
"""

import httpx
import logging
from typing import Any

from config.settings import SARVAM_API_KEY, SARVAM_BASE_URL

logger = logging.getLogger(__name__)


class SarvamClient:
    """Thin async client for Sarvam AI APIs."""

    def __init__(self):
        self.base_url = SARVAM_BASE_URL
        self.headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

    async def speech_to_text(
        self, audio_bytes: bytes, language_code: str = "ta-IN"
    ) -> str:
        """
        Transcribe audio (WAV/MP3) to text using Sarvam STT.

        language_code: "ta-IN" (Tamil), "kn-IN" (Kannada), "en-IN" (English)
        """
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning placeholder transcript")
            return "[Placeholder STT: audio transcription would appear here]"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/speech-to-text-translate",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"model": "saarika:v2", "language_code": language_code},
            )
            resp.raise_for_status()
            return resp.json().get("transcript", "")

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "ta-IN",
        speaker: str = "meera",
        speed: float = 1.0,
    ) -> bytes:
        """
        Convert text to speech audio (WAV) using Sarvam TTS.

        Tamil speakers: meera, pavithra, maitreyi, kalpana
        Kannada speakers: amol, arjun
        """
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning empty audio")
            return b""

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/text-to-speech",
                headers=self.headers,
                json={
                    "inputs": [text],
                    "target_language_code": language_code,
                    "speaker": speaker,
                    "model": "bulbul:v1",
                    "speed": speed,
                    "enable_preprocessing": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # Sarvam returns base64-encoded WAV
            import base64
            audio_b64 = data["audios"][0]
            return base64.b64decode(audio_b64)

    async def translate(
        self, text: str, source_lang: str = "ta-IN", target_lang: str = "en-IN"
    ) -> str:
        """Translate text between Indic languages and English."""
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning original text")
            return text

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/translate",
                headers=self.headers,
                json={
                    "input": text,
                    "source_language_code": source_lang,
                    "target_language_code": target_lang,
                    "model": "mayura:v1",
                },
            )
            resp.raise_for_status()
            return resp.json().get("translated_text", text)


# Singleton
sarvam = SarvamClient()
