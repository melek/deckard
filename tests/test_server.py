"""Tests for deckard HTTP server."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error

import pytest

from deckard.server import (
    DeckardServer,
    conversation_id,
    estimate_prompt_tokens,
    estimate_completion_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    srv = DeckardServer(host="127.0.0.1", port=0, db_path=str(tmp_path / "test.db"))
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{srv.port}"
    srv.shutdown()


def _post_json(url: str, data: dict, timeout: float = 5) -> tuple[int, dict]:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get_json(url: str, timeout: float = 5) -> tuple[int, dict | list]:
    resp = urllib.request.urlopen(url, timeout=timeout)
    return resp.status, json.loads(resp.read())


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestConversationId:
    def test_single_message(self):
        cid = conversation_id([{"role": "user", "content": "hello"}])
        # Single message = last user message excluded, context is empty string
        assert len(cid) == 16

    def test_multiple_messages_same_prefix(self):
        msgs1 = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "question A"},
        ]
        msgs2 = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "question B"},
        ]
        assert conversation_id(msgs1) == conversation_id(msgs2)

    def test_different_prefix(self):
        msgs1 = [{"role": "user", "content": "aaa"}, {"role": "user", "content": "x"}]
        msgs2 = [{"role": "user", "content": "bbb"}, {"role": "user", "content": "x"}]
        assert conversation_id(msgs1) != conversation_id(msgs2)


class TestTokenEstimation:
    def test_prompt_tokens(self):
        msgs = [{"role": "user", "content": "hello world"}]
        assert estimate_prompt_tokens(msgs) == len("hello world") // 4

    def test_completion_tokens(self):
        assert estimate_completion_tokens("one two three") == 3

    def test_completion_tokens_empty(self):
        # "".split() == [] → 0 words, but len([""])=1 won't happen
        assert estimate_completion_tokens("") == 0


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, server):
        code, data = _get_json(f"{server}/health")
        assert code == 200
        assert data["status"] == "ok"
        assert data["pending"] == 0


class TestModels:
    def test_returns_deckard(self, server):
        code, data = _get_json(f"{server}/v1/models")
        assert code == 200
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "deckard"
        assert data["data"][0]["owned_by"] == "human"


class TestQueueEndpoint:
    def test_initially_empty(self, server):
        code, data = _get_json(f"{server}/_deckard/queue")
        assert code == 200
        assert data == []


class TestRespondErrors:
    def test_nonexistent_id(self, server):
        code, data = _post_json(
            f"{server}/_deckard/queue/nonexistent-id/respond",
            {"response": "hello"},
        )
        assert code == 404

    def test_malformed_json(self, server):
        req = urllib.request.Request(
            f"{server}/_deckard/queue/some-id/respond",
            data=b"not json{{{",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            code = resp.status
        except urllib.error.HTTPError as e:
            code = e.code
        assert code == 400

    def test_missing_response_field(self, server):
        code, data = _post_json(
            f"{server}/_deckard/queue/some-id/respond",
            {"wrong_field": "hello"},
        )
        assert code == 400


class TestFullRoundtrip:
    def test_non_streaming(self, server):
        """POST completion → appears in queue → TUI responds → client gets answer."""
        result = {}
        error = {}

        def client():
            try:
                code, data = _post_json(
                    f"{server}/v1/chat/completions",
                    {
                        "model": "deckard",
                        "messages": [{"role": "user", "content": "What is 2+2?"}],
                        "stream": False,
                    },
                    timeout=10,
                )
                result["code"] = code
                result["data"] = data
            except Exception as e:
                error["exc"] = e

        t = threading.Thread(target=client)
        t.start()

        # Wait for request to appear in queue
        for _ in range(50):
            _, queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Request never appeared in queue")

        assert len(queue) == 1
        req_id = queue[0]["id"]
        assert queue[0]["status"] == "pending"
        assert queue[0]["messages"][0]["content"] == "What is 2+2?"

        # Health should show 1 pending
        _, health = _get_json(f"{server}/health")
        assert health["pending"] == 1

        # TUI responds
        code, resp_data = _post_json(
            f"{server}/_deckard/queue/{req_id}/respond",
            {"response": "The answer is four.", "reading_ms": 500, "composing_ms": 1500},
        )
        assert code == 200

        # Wait for client thread
        t.join(timeout=10)
        assert "exc" not in error, f"Client error: {error.get('exc')}"

        assert result["code"] == 200
        data = result["data"]
        assert data["choices"][0]["message"]["content"] == "The answer is four."
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify token estimation in usage
        assert data["usage"]["prompt_tokens"] == len("What is 2+2?") // 4
        assert data["usage"]["completion_tokens"] == 4  # "The answer is four."
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 4

    def test_streaming(self, server):
        """POST with stream=true → SSE chunks."""
        import http.client
        from urllib.parse import urlparse

        result = {}
        error = {}

        def client():
            try:
                parsed = urlparse(server)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({
                    "model": "deckard",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                })
                conn.request("POST", "/v1/chat/completions", body=body,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                raw = resp.read().decode()
                result["raw"] = raw
                result["status"] = resp.status
                conn.close()
            except Exception as e:
                error["exc"] = e

        t = threading.Thread(target=client)
        t.start()

        # Wait for request to appear in queue
        for _ in range(50):
            _, queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Request never appeared in queue")

        req_id = queue[0]["id"]

        # TUI responds
        _post_json(
            f"{server}/_deckard/queue/{req_id}/respond",
            {"response": "Hello there friend"},
        )

        t.join(timeout=10)
        assert "exc" not in error, f"Client error: {error.get('exc')}"

        raw = result["raw"]
        lines = [l for l in raw.strip().split("\n") if l.startswith("data: ")]

        # Parse SSE events
        events = []
        for line in lines:
            payload = line[len("data: "):]
            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(payload))

        # First: role chunk
        assert events[0]["choices"][0]["delta"] == {"role": "assistant"}

        # Middle: content chunks (one per word: "Hello", "there", "friend")
        content_chunks = [
            e for e in events
            if isinstance(e, dict) and "content" in e.get("choices", [{}])[0].get("delta", {})
        ]
        assert len(content_chunks) == 3
        reassembled = "".join(
            c["choices"][0]["delta"]["content"] for c in content_chunks
        )
        assert reassembled == "Hello there friend"

        # Second-to-last: stop chunk
        stop_chunk = events[-2]
        assert stop_chunk["choices"][0]["finish_reason"] == "stop"
        assert stop_chunk["choices"][0]["delta"] == {}

        # Last: [DONE] sentinel
        assert events[-1] == "[DONE]"

    def test_double_respond_returns_409(self, server):
        """Responding twice to the same request returns 409."""
        def client():
            _post_json(
                f"{server}/v1/chat/completions",
                {"model": "deckard", "messages": [{"role": "user", "content": "test"}]},
                timeout=10,
            )

        t = threading.Thread(target=client)
        t.start()

        for _ in range(50):
            _, queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)

        req_id = queue[0]["id"]

        code1, _ = _post_json(
            f"{server}/_deckard/queue/{req_id}/respond",
            {"response": "first"},
        )
        assert code1 == 200

        t.join(timeout=10)

        # Second respond should be 409 (or 404 if already cleaned up)
        code2, _ = _post_json(
            f"{server}/_deckard/queue/{req_id}/respond",
            {"response": "second"},
        )
        assert code2 in (409, 404)


class TestTokenUsage:
    def test_usage_field_accuracy(self, server):
        """Token estimation matches specification: chars//4 for prompt, word count for completion."""
        result = {}

        def client():
            code, data = _post_json(
                f"{server}/v1/chat/completions",
                {
                    "model": "deckard",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Count to five."},
                    ],
                },
                timeout=10,
            )
            result["data"] = data

        t = threading.Thread(target=client)
        t.start()

        for _ in range(50):
            _, queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)

        req_id = queue[0]["id"]
        response_text = "one two three four five"
        _post_json(
            f"{server}/_deckard/queue/{req_id}/respond",
            {"response": response_text},
        )

        t.join(timeout=10)

        usage = result["data"]["usage"]
        expected_prompt = (len("You are helpful.") + len("Count to five.")) // 4
        expected_completion = 5  # five words

        assert usage["prompt_tokens"] == expected_prompt
        assert usage["completion_tokens"] == expected_completion
        assert usage["total_tokens"] == expected_prompt + expected_completion
