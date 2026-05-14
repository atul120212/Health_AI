"""
Sarvam AI client wrapper for Tamil/Kannada TTS and STT.

Sarvam AI provides:
  - Speech-to-Text (Indic languages including Tamil & Kannada)
  - Text-to-Speech (natural Indic voices)
  - Translation (Indic ↔ English)
  - Transliteration

This client wraps those APIs for use in IVR and portal widget flows.
"""

import re

import httpx
import logging

from config.settings import SARVAM_API_KEY, SARVAM_BASE_URL

logger = logging.getLogger(__name__)


def _clean_for_tts(text: str) -> str:
    """
    Strip markdown and non-speech characters from LLM output before TTS.

    LLMs often return **bold**, *italic*, bullet lists, code spans, etc.
    These are visually useful but sound wrong when read aloud.
    """
    # Remove markdown headers (# Heading)
    text = re.sub(r'#{1,6}\s*', '', text)
    # Remove bold/italic markers (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    # Remove inline code (`code`)
    text = re.sub(r'`+[^`]*`+', '', text)
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Convert bullet / list markers to a natural pause (comma or period)
    text = re.sub(r'^\s*[-*•·]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)
    # Remove non-speech symbols: ~, ^, |, \, {}, <>, backtick, @, #
    text = re.sub(r'[~^|\\{}@#<>]', '', text)
    # Collapse repeated punctuation left after stripping (e.g. "…" artefacts)
    text = re.sub(r'([.?!,;:\-])\1+', r'\1', text)
    # Collapse multiple blank lines → single newline (becomes a pause)
    text = re.sub(r'\n{2,}', '\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _clean_transcript(text: str) -> str:
    """
    Normalise raw STT output into natural speech text.

    Removes artefacts that ASR models sometimes emit:
      • Repeated punctuation  ("really???" → "really?", "ok..." → "ok.")
      • Non-speech symbols    (*, #, @, ~, `, |, ^, {, }, <, >)
      • Leading/trailing noise (dashes, underscores, stray quotes)
      • Redundant whitespace
    Leaves sentence-ending punctuation intact so downstream TTS pauses correctly.
    """
    # Collapse runs of the same punctuation mark
    text = re.sub(r'([.?!,;:\-])\1+', r'\1', text)
    # Remove characters that are noise in transcripts but not speech
    text = re.sub(r'[*#@~`|^{}<>\\]', '', text)
    # Collapse multiple spaces / tabs
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


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
        self, audio_bytes: bytes, language_code: str = "unknown"
    ) -> tuple[str, str]:
        """
        Transcribe audio using Sarvam Saarika STT (saarika:v2.5).

        Endpoint : POST /speech-to-text
        Formats  : WAV, MP3, AAC, OGG, FLAC, WebM, MP4 — all accepted

        language_code: pass "unknown" (default) to let Sarvam auto-detect the
            spoken language; pass a BCP-47 code ("ta-IN", "kn-IN", "en-IN") to
            force a specific language.

        Returns:
            (transcript, detected_language_code)
            - transcript            cleaned, naturalised text
            - detected_language_code  BCP-47 code as reported by Sarvam
              (falls back to the input language_code if absent in the response)
        """
        if not SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set — returning placeholder transcript")
            return "[Placeholder STT: audio transcription would appear here]", language_code

        filename, mime = _detect_audio_mime(audio_bytes)
        logger.info("STT upload: %s (%d bytes), lang hint=%s", mime, len(audio_bytes), language_code)

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

            data = resp.json()
            raw_transcript = data.get("transcript", "")
            detected_lang = data.get("language_code") or language_code

            transcript = _clean_transcript(raw_transcript)
            logger.info(
                "STT detected_lang=%s raw_len=%d clean_len=%d transcript=%r",
                detected_lang, len(raw_transcript), len(transcript), transcript[:120],
            )
            return transcript, detected_lang

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
        clean_text = _clean_for_tts(text)
        if not clean_text:
            logger.warning("TTS: text was empty after cleaning, skipping")
            return b""
        if len(clean_text) != len(text):
            logger.debug("TTS cleaned %d→%d chars", len(text), len(clean_text))

        payload = {
            "text": clean_text,                  # cleaned, markdown-free string
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
