"""Tests for deckard types."""
import pytest
from deckard.types import Status, QueuedRequest, DeckardConfig


class TestStatus:
    def test_status_values(self):
        assert Status.PENDING.value == "pending"
        assert Status.IN_PROGRESS.value == "in_progress"
        assert Status.COMPLETED.value == "completed"
        assert Status.ABANDONED.value == "abandoned"

    def test_status_from_string(self):
        assert Status("pending") == Status.PENDING
        assert Status("completed") == Status.COMPLETED


class TestQueuedRequest:
    def test_construction(self):
        req = QueuedRequest(
            id="test-123",
            created_at="2026-03-30T14:00:00Z",
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            status=Status.PENDING,
        )
        assert req.id == "test-123"
        assert req.response is None
        assert req.responded_at is None

    def test_with_response(self):
        req = QueuedRequest(
            id="test-456",
            created_at="2026-03-30T14:00:00Z",
            model="deckard",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            status=Status.COMPLETED,
            response="world",
            responded_at="2026-03-30T14:01:00Z",
        )
        assert req.response == "world"
        assert req.stream is True


class TestDeckardConfig:
    def test_defaults(self):
        cfg = DeckardConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8421
        assert cfg.simulate_latency is False
        assert cfg.chunk_delay_ms == 2

    def test_override(self):
        cfg = DeckardConfig(port=9000, simulate_latency=True, chunk_delay_ms=30)
        assert cfg.port == 9000
        assert cfg.simulate_latency is True
        assert cfg.chunk_delay_ms == 30
