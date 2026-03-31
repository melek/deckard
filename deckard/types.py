"""Shared types for deckard server and client."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Status(enum.Enum):
    """Lifecycle status of a queued request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class QueuedRequest:
    """A request waiting for a human response."""
    id: str
    created_at: str
    model: str
    messages: list[dict]
    stream: bool
    status: Status
    response: str | None = None
    responded_at: str | None = None


@dataclass
class DeckardConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8421
    db_path: str = "~/.deckard/deckard.db"
    simulate_latency: bool = False
    chunk_delay_ms: int = 2
