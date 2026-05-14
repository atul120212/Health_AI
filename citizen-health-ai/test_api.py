#!/usr/bin/env python3
"""
Integration test script for all Citizen Health AI endpoints.

Run with:  python test_api.py

Uses in-process ASGI transport — no running server required.
Patches the Sarvam LLM + STT/TTS to avoid real API calls.
"""

import asyncio
import json
import struct
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import httpx

# ---------------------------------------------------------------------------
# Minimal silent WAV for voice endpoint tests
# ---------------------------------------------------------------------------

def make_silent_wav(duration_s: float = 0.1, sample_rate: int = 8000) -> bytes:
    num_samples = int(duration_s * sample_rate)
    audio_data = b"\x00\x00" * num_samples          # 16-bit PCM silence
    data_size = len(audio_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1,            # PCM
        1,            # mono
        sample_rate,
        sample_rate * 2,   # byte rate
        2,                 # block align
        16,                # bits per sample
        b"data", data_size,
    )
    return header + audio_data


SILENT_WAV = make_silent_wav()

# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

MOCK_CITIZEN_REPLY   = "நீங்கள் ஆயுஷ்மான் பாரத் PM-JAY திட்டத்திற்கு தகுதியானவர். மருத்துவமனை: அரசு பொது மருத்துவமனை, சென்னை."
MOCK_WORKER_REPLY    = "IMNCI நெறிமுறை: ORS கொடுங்கள், முக்கிய அறிகுறிகளை கவனியுங்கள்."
MOCK_EXTRACT_JSON    = json.dumps({
    "weight_kg": 52.5,
    "bp_systolic": 118,
    "bp_diastolic": 76,
    "visit_notes": "Normal checkup, no danger signs observed.",
})
MOCK_ALERT_JSON      = json.dumps({
    "narrative": (
        "Dengue cases in Chennai rose to 3.1 standard deviations above baseline "
        "this week (47 suspected cases vs. mean of 12). The 4-week trend shows a "
        "consistent upward slope. Immediate response is recommended."
    ),
    "actions": [
        "Activate district rapid response team within 24 hours.",
        "Deploy fogging operations in affected wards.",
        "Distribute dengue awareness materials at PHCs.",
        "Enhance larval source reduction activities.",
        "Issue public advisory via local media.",
    ],
})

MOCK_STT_TRANSCRIPT  = "மருத்துவமனை எங்கே இருக்கிறது? என்னால் Ayushman Bharat திட்டத்திற்கு தகுதி பெற முடியுமா?"


def _non_stream_response(text: str):
    """Build a minimal OpenAI-compatible non-streaming chat completion."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


async def _stream_response(text: str):
    """Async generator that yields minimal OpenAI-compatible streaming chunks."""
    # Text chunk
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    yield chunk

    # Terminal chunk
    end_delta = MagicMock()
    end_delta.content = None
    end_delta.tool_calls = None
    end_choice = MagicMock()
    end_choice.delta = end_delta
    end_choice.finish_reason = "stop"
    end_chunk = MagicMock()
    end_chunk.choices = [end_choice]
    yield end_chunk


async def _mock_create(*args, **kwargs):
    """
    Unified mock for client.chat.completions.create.

    Routes to the right mock response based on the messages content.
    """
    messages = kwargs.get("messages", [])
    last_content = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("content"):
            last_content = m["content"]
            break

    stream = kwargs.get("stream", False)

    # Pick mock based on context
    if "field note" in last_content.lower() or "extract structured" in last_content.lower():
        reply = MOCK_EXTRACT_JSON
    elif any(kw in last_content.lower() for kw in ("outbreak", "z-score", "district health")):
        reply = MOCK_ALERT_JSON
    elif any(kw in last_content.lower() for kw in ("worker", "asha", "imnci", "protocol")):
        reply = MOCK_WORKER_REPLY
    else:
        reply = MOCK_CITIZEN_REPLY

    if stream:
        return _stream_response(reply)
    return _non_stream_response(reply)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_tests():
    # Import app + singletons (LLM clients created here)
    from main import app
    from src.citizen_ai.citizen_assistant       import citizen_assistant
    from src.health_worker_ai.health_worker_assistant import health_worker_assistant
    from src.surveillance_ai.outbreak_detector  import outbreak_detector
    from src.shared.sarvam_client               import sarvam

    # ── Patch LLM clients on every singleton ───────────────────────────────
    for assistant in (citizen_assistant, health_worker_assistant, outbreak_detector):
        assistant.client = MagicMock()
        assistant.client.chat = MagicMock()
        assistant.client.chat.completions = MagicMock()
        assistant.client.chat.completions.create = AsyncMock(side_effect=_mock_create)

    # ── Patch Sarvam STT / TTS ─────────────────────────────────────────────
    sarvam.speech_to_text = AsyncMock(return_value=MOCK_STT_TRANSCRIPT)
    sarvam.text_to_speech = AsyncMock(return_value=SILENT_WAV)

    # ── HTTP client against the in-process ASGI app ─────────────────────────
    passed = failed = 0

    def ok(name, detail=""):
        nonlocal passed
        passed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  \033[32mPASS\033[0m  {name}{suffix}")

    def fail(name, err):
        nonlocal failed
        failed += 1
        print(f"  \033[31mFAIL\033[0m  {name}: {err}")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:

        # ── 1. Root redirect ───────────────────────────────────────────────
        try:
            r = await c.get("/", follow_redirects=False)
            assert r.status_code in (301, 302, 307, 308), f"expected redirect, got {r.status_code}"
            ok("GET /", f"→ {r.headers.get('location')}")
        except Exception as e:
            fail("GET /", e)

        # ── 2. Health check ────────────────────────────────────────────────
        try:
            r = await c.get("/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            ok("GET /health", body)
        except Exception as e:
            fail("GET /health", e)

        # ── 3. Citizen chat (Tamil) ────────────────────────────────────────
        try:
            payload = {
                "messages": [
                    {"role": "user", "content": "என்னுடைய Ayushman Bharat தகுதி என்ன?"}
                ],
                "language": "ta",
            }
            r = await c.post("/citizen/chat", json=payload)
            assert r.status_code == 200, r.text
            body = r.json()
            assert "reply" in body and body["reply"]
            ok("POST /citizen/chat", f"reply: {body['reply'][:60]}…")
        except Exception as e:
            fail("POST /citizen/chat", e)

        # ── 4. Citizen chat (Kannada) ──────────────────────────────────────
        try:
            payload = {
                "messages": [
                    {"role": "user", "content": "ಹತ್ತಿರದ ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?"}
                ],
                "language": "kn",
            }
            r = await c.post("/citizen/chat", json=payload)
            assert r.status_code == 200, r.text
            ok("POST /citizen/chat (Kannada)", r.json()["reply"][:60] + "…")
        except Exception as e:
            fail("POST /citizen/chat (Kannada)", e)

        # ── 5. Citizen voice (IVR) ─────────────────────────────────────────
        try:
            r = await c.post(
                "/citizen/voice",
                files={"audio": ("test.wav", SILENT_WAV, "audio/wav")},
                data={"language_code": "ta-IN", "conversation_json": "[]"},
            )
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("audio/wav")
            ok("POST /citizen/voice", f"{len(r.content)} bytes WAV returned")
        except Exception as e:
            fail("POST /citizen/voice", e)

        # ── 6. Worker chat (ASHA) ──────────────────────────────────────────
        try:
            payload = {
                "messages": [
                    {"role": "user", "content": "5-year-old child with fever 3 days, not drinking. What protocol?"}
                ],
                "language": "en",
                "worker_role": "asha",
            }
            r = await c.post("/worker/chat", json=payload)
            assert r.status_code == 200, r.text
            ok("POST /worker/chat (ASHA)", r.json()["reply"][:60] + "…")
        except Exception as e:
            fail("POST /worker/chat (ASHA)", e)

        # ── 7. Worker chat (nurse) ─────────────────────────────────────────
        try:
            payload = {
                "messages": [
                    {"role": "user", "content": "ORS dosage for child under 2 years?"}
                ],
                "language": "en",
                "worker_role": "nurse",
            }
            r = await c.post("/worker/chat", json=payload)
            assert r.status_code == 200, r.text
            ok("POST /worker/chat (nurse)", r.json()["reply"][:60] + "…")
        except Exception as e:
            fail("POST /worker/chat (nurse)", e)

        # ── 8. Worker voice query ──────────────────────────────────────────
        try:
            r = await c.post(
                "/worker/voice",
                files={"audio": ("field.wav", SILENT_WAV, "audio/wav")},
                data={"language_code": "ta-IN", "worker_role": "asha", "conversation_json": "[]"},
            )
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("audio/wav")
            ok("POST /worker/voice", f"{len(r.content)} bytes WAV returned")
        except Exception as e:
            fail("POST /worker/voice", e)

        # ── 9. Worker voice-update (field note → HMIS) ─────────────────────
        try:
            r = await c.post(
                "/worker/voice-update",
                files={"audio": ("visit.wav", SILENT_WAV, "audio/wav")},
                data={"patient_id": "TN-2024-00123", "language_code": "ta-IN"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert "transcript" in body and "extracted_fields" in body and "hmis_result" in body
            ok(
                "POST /worker/voice-update",
                f"fields extracted: {list(body['extracted_fields'].keys())}",
            )
        except Exception as e:
            fail("POST /worker/voice-update", e)

        # ── 10. Surveillance analyse (outbreak) ────────────────────────────
        try:
            r = await c.post(
                "/surveillance/analyse",
                json={"district": "Chennai", "disease": "dengue", "weeks": 52},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert "status" in body and "z_score" in body
            ok(
                "POST /surveillance/analyse",
                f"status={body['status']}, z_score={body['z_score']}, "
                f"narrative={'yes' if body.get('alert_narrative') else 'none'}",
            )
        except Exception as e:
            fail("POST /surveillance/analyse", e)

        # ── 11. Surveillance scan (all priority diseases) ──────────────────
        try:
            r = await c.post(
                "/surveillance/scan",
                json={"district": "Chennai"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert "alerts" in body and "total" in body
            ok(
                "POST /surveillance/scan",
                f"total alerts: {body['total']}, diseases flagged: "
                f"{[a['disease'] for a in body['alerts'][:3]]}",
            )
        except Exception as e:
            fail("POST /surveillance/scan", e)

    # ── Summary ────────────────────────────────────────────────────────────
    total = passed + failed
    print()
    print(f"  {'─' * 50}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED ← fix these before demo")
    else:
        print("  ·  all green ✓")
    print()
    return failed


if __name__ == "__main__":
    print()
    print("  Citizen Health AI — endpoint tests")
    print("  ════════════════════════════════════")
    print()
    failures = asyncio.run(run_tests())
    sys.exit(failures)
