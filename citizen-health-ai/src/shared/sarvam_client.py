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

from config.settings import SARVAM_API_KEY, SARVAM_BASE_URL

logger = logging.getLogger(__name__)


def _detect_audio_mime(audio_bytes: bytes) -> tuple[str, str]:
    """
    Return (filename, mime_type) by inspecting magic bytes.
    Browsers (MediaRecorder) produce WebM; direct uploads may be WAV/MP3.
    """
    if audio_bytes[:4] == b"RIFF":
        return "audio.wav", "audio/wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "audio.mp3", "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio.ogg", "audio/ogg"
    if audio_bytes[:4] == b"fLaC":
        return "audio.flac", "audio/flac"
    # WebM magic: \x1a\x45\xdf\xa3  (also covers Matroska)
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "audio.webm", "audio/webm"
    # Default: treat unknown as WebM (most common from browser MediaRecorder)
    return "audio.webm", "audio/webm"


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
        Transcribe audio using Sarvam Saarika STT (saarika:v2.5).

        Endpoint : POST /speech-to-text
        Formats  : WAV, MP3, AAC, OGG, FLAC, WebM, MP4 — all accepted
        language_code: "ta-IN" | "kn-IN" | "en-IN" | "unknown" (auto-detect)
        """
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning placeholder transcript")
            return "[Placeholder STT: audio transcription would appear here]"

        filename, mime = _detect_audio_mime(audio_bytes)
        logger.info("STT upload: %s (%d bytes)", mime, len(audio_bytes))

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": (filename, audio_bytes, mime)},
                data={"model": "saarika:v2.5", "language_code": language_code},
            )
            if not resp.is_success:
                logger.error("Sarvam STT %s — %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            return resp.json().get("transcript", "")

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "ta-IN",
        speaker: str = "anushka",
        pace: float = 1.0,
    ) -> bytes:
        """
        Convert text to speech audio (WAV) using Sarvam Bulbul TTS (bulbul:v2).

        Valid speakers — female: anushka, manisha, vidya, arya
                         male  : abhilash, karun, hitesh
        All speakers support all 11 Indic languages + English (en-IN).
        """
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning empty audio")
            return b""

        import base64
        payload = {
            "text": text,                        # string, not array
            "target_language_code": language_code,
            "model": "bulbul:v2",
            "speaker": speaker,
            "pace": pace,
            "enable_preprocessing": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/text-to-speech",
                headers=self.headers,
                json=payload,
            )
            if not resp.is_success:
                logger.error(
                    "Sarvam TTS %s — body: %s — text_len=%d text_preview=%r",
                    resp.status_code,
                    resp.text[:500],
                    len(text),
                    text[:200],
                )
                resp.raise_for_status()
            data = resp.json()
            return base64.b64decode(data["audios"][0])

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
