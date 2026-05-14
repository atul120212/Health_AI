"""
LiveKit room and access-token management.

Provides:
  - create_room()      — create a LiveKit room via the server API
  - user_token()       — JWT for a browser participant
  - bot_token()        — JWT for the server-side Pipecat bot participant
  - list_rooms()       — active rooms (debug)

Token generation is lightweight and works with livekit-api (pure Python).
Room creation optionally uses the LiveKit server API; if the server is not
reachable we skip room creation (LiveKit auto-creates rooms on first join).
"""

import logging
import time
import uuid
from datetime import timedelta

from config.settings import LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL

logger = logging.getLogger(__name__)

_LIVEKIT_AVAILABLE = bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET)


def _make_token(room_name: str, identity: str, ttl_seconds: int = 3600) -> str:
    """
    Generate a signed LiveKit JWT using livekit-api.

    Falls back to an empty string if livekit-api is not installed or
    credentials are not configured.
    """
    if not _LIVEKIT_AVAILABLE:
        return ""
    try:
        from livekit.api import AccessToken, VideoGrants

        grants = VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(grants)
            .with_ttl(timedelta(seconds=ttl_seconds))
        )
        return token.to_jwt()
    except Exception as exc:
        logger.error("LiveKit token generation failed: %s", exc)
        return ""


def new_room_name(module: str) -> str:
    return f"health-{module}-{uuid.uuid4().hex[:10]}"


def user_token(room_name: str, participant_id: str = "user") -> str:
    return _make_token(room_name, identity=participant_id)


def bot_token(room_name: str) -> str:
    return _make_token(room_name, identity="aarogya-bot")


async def create_room(room_name: str) -> bool:
    """
    Pre-create a LiveKit room via the server API.

    Returns True on success. LiveKit auto-creates rooms on first join, so
    a False return does not prevent the session from working.
    """
    if not _LIVEKIT_AVAILABLE:
        return False
    try:
        from livekit import api as lk_api

        async with lk_api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        ) as api:
            await api.room.create_room(lk_api.CreateRoomRequest(name=room_name))
        logger.info("LiveKit room created: %s", room_name)
        return True
    except Exception as exc:
        logger.warning("Could not pre-create LiveKit room (will auto-create): %s", exc)
        return False


class RoomManager:
    """Lightweight session registry — maps room_name → metadata."""

    def __init__(self):
        self._rooms: dict[str, dict] = {}

    def register(self, room_name: str, module: str, language_code: str) -> None:
        self._rooms[room_name] = {
            "module": module,
            "language_code": language_code,
            "created_at": time.time(),
        }

    def unregister(self, room_name: str) -> None:
        self._rooms.pop(room_name, None)

    def list_active(self) -> list[dict]:
        return [{"room": k, **v} for k, v in self._rooms.items()]


# Singleton
room_manager = RoomManager()
