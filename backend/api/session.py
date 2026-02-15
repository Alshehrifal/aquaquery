"""In-memory session storage for chat history."""

import uuid
from datetime import datetime, timezone
from typing import Any

from backend.data.schema import Message


class SessionStore:
    """In-memory chat session storage. Replace with Redis/Postgres for production."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = {}

    def get_or_create_session(self, session_id: str | None = None) -> str:
        """Get existing session or create a new one. Returns session_id."""
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        visualization: dict[str, Any] | None = None,
        sources: list[str] | None = None,
    ) -> Message:
        """Add a message to a session. Returns the created Message."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        msg = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            visualization=visualization,
            sources=sources or [],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._sessions[session_id].append(msg)
        return msg

    def get_history(self, session_id: str) -> list[Message]:
        """Get all messages for a session."""
        return list(self._sessions.get(session_id, []))

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions
