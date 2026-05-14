"""
IVR session state management — in-memory store with TTL-based expiry.

Each IVR session tracks:
  - language_code  (ta-IN / kn-IN / en-IN)
  - conversation history (OpenAI message format)
  - module  (citizen / worker / surveillance)
  - timestamps for TTL expiry
"""

import time
import uuid
from dataclasses import dataclass, field

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity


@dataclass
class IVRSession:
    session_id: str
    language_code: str = "ta-IN"
    module: str = "citizen"          # citizen | worker | surveillance
    worker_role: str = "asha"        # asha | nurse  (only relevant for worker module)
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS

    @property
    def lang(self) -> str:
        """Short language code: 'ta-IN' → 'ta'."""
        return self.language_code.split("-")[0]

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2


class SessionManager:
    """Thread-safe (asyncio single-threaded) in-memory IVR session store."""

    def __init__(self) -> None:
        self._store: dict[str, IVRSession] = {}

    def create(
        self,
        language_code: str = "ta-IN",
        module: str = "citizen",
        worker_role: str = "asha",
    ) -> IVRSession:
        session = IVRSession(
            session_id=str(uuid.uuid4()),
            language_code=language_code,
            module=module,
            worker_role=worker_role,
        )
        self._store[session.session_id] = session
        self._evict_expired()
        return session

    def get(self, session_id: str) -> IVRSession | None:
        session = self._store.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            del self._store[session_id]
            return None
        session.touch()
        return session

    def end(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_active(self) -> list[dict]:
        self._evict_expired()
        return [
            {
                "session_id": s.session_id,
                "language_code": s.language_code,
                "module": s.module,
                "turn_count": s.turn_count,
                "age_seconds": int(time.time() - s.created_at),
            }
            for s in self._store.values()
        ]

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._store.items() if s.is_expired]
        for sid in expired:
            del self._store[sid]


# Module-level singleton
session_manager = SessionManager()
