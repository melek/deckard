# Deckard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a human-as-LLM endpoint — an OpenAI-compatible HTTP server with a Textual TUI where a human reads prompts and types responses.

**Architecture:** Four modules (types → server → client → CLI entry), built bottom-up. Server uses stdlib `http.server` with `ThreadingHTTPServer`. Client uses Textual. Communication via HTTP (same pattern as UIDI REPL ↔ daemon). SQLite for durable logging.

**Tech Stack:** Python 3.10+, Textual (TUI), SQLite (logging), stdlib `http.server` + `threading` + `json` + `uuid` + `hashlib`.

---

## File Structure

| File | Responsibility | Task |
|------|---------------|------|
| `deckard/__init__.py` | Package marker | 1 |
| `deckard/types.py` | Status enum, QueuedRequest, DeckardConfig | 1 |
| `deckard/server.py` | HTTP server, request queue, SQLite, SSE streaming | 2, 3 |
| `deckard/client.py` | Textual TUI app | 4, 5 |
| `deckard/app.tcss` | Textual stylesheet | 4 |
| `deckard/__main__.py` | CLI entry, arg parsing, server discovery | 6 |
| `tests/test_types.py` | Type construction and serialization | 1 |
| `tests/test_server.py` | HTTP endpoints, queue mechanics, SSE | 2, 3 |
| `pyproject.toml` | Project metadata and dependencies | 1 |
| `CLAUDE.md` | Project instructions for Claude Code | 7 |

---

### Task 1: Project Scaffold + Types

**Files:**
- Create: `deckard/__init__.py`
- Create: `deckard/types.py`
- Create: `tests/__init__.py`
- Create: `tests/test_types.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "deckard"
version = "0.1.0"
description = "Human-as-LLM endpoint — OpenAI-compatible API server with TUI"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = ["textual>=1.0"]

[project.scripts]
deckard = "deckard.__main__:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

Create `deckard/__init__.py`:
```python
"""Deckard — Human-as-LLM endpoint."""
```

Create `tests/__init__.py` (empty file).

- [ ] **Step 3: Write failing tests for types**

Create `tests/test_types.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_types.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'deckard.types'`

- [ ] **Step 5: Implement types**

Create `deckard/types.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_types.py -v
```

Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
cd /home/melek/workshop/deckard
git add pyproject.toml deckard/__init__.py deckard/types.py tests/__init__.py tests/test_types.py
git commit -m "feat: project scaffold + Status, QueuedRequest, DeckardConfig types"
```

---

### Task 2: Server — HTTP Endpoints + Queue + SQLite

**Files:**
- Create: `deckard/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing tests for server endpoints**

Create `tests/test_server.py`:

```python
"""Tests for deckard HTTP server."""
import json
import threading
import time
import urllib.request
import urllib.error
import pytest
from deckard.server import DeckardServer


@pytest.fixture
def server():
    """Start a deckard server on a random port, yield base URL, shut down after."""
    srv = DeckardServer(host="127.0.0.1", port=0)  # port=0 → OS picks
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    # Wait for server to be ready
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{srv.port}"
    srv.shutdown()


def _post_json(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _get_json(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


class TestHealthEndpoint:
    def test_health_returns_ok(self, server):
        data = _get_json(f"{server}/health")
        assert data["status"] == "ok"
        assert "pending" in data

    def test_health_pending_count(self, server):
        data = _get_json(f"{server}/health")
        assert data["pending"] == 0


class TestModelsEndpoint:
    def test_models_returns_deckard(self, server):
        data = _get_json(f"{server}/v1/models")
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "deckard"


class TestQueueEndpoint:
    def test_queue_initially_empty(self, server):
        data = _get_json(f"{server}/_deckard/queue")
        assert data == []

    def test_completion_request_queues(self, server):
        """POST /v1/chat/completions queues a request, respond via internal API."""
        body = {
            "model": "deckard",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        # Send completion in a thread (it blocks until response)
        result = {}
        def do_request():
            result["data"] = _post_json(f"{server}/v1/chat/completions", body)
        t = threading.Thread(target=do_request)
        t.start()

        # Wait for it to appear in queue
        for _ in range(50):
            queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        assert len(queue) == 1
        req_id = queue[0]["id"]
        assert queue[0]["status"] == "pending"

        # Respond via internal API
        resp = _post_json(f"{server}/_deckard/queue/{req_id}/respond", {
            "response": "world",
            "reading_ms": 500,
            "composing_ms": 1000,
        })
        assert resp.get("status") == "ok"

        # Original request should complete
        t.join(timeout=5)
        assert "data" in result
        assert result["data"]["choices"][0]["message"]["content"] == "world"

    def test_respond_404_unknown_id(self, server):
        try:
            _post_json(f"{server}/_deckard/queue/nonexistent/respond", {"response": "x"})
            assert False, "should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_respond_400_malformed_json(self, server):
        req = urllib.request.Request(
            f"{server}/_deckard/queue/fake/respond",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400


class TestTokenEstimation:
    def test_completion_includes_usage(self, server):
        body = {
            "model": "deckard",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        }
        result = {}
        def do_request():
            result["data"] = _post_json(f"{server}/v1/chat/completions", body)
        t = threading.Thread(target=do_request)
        t.start()

        for _ in range(50):
            queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        req_id = queue[0]["id"]

        _post_json(f"{server}/_deckard/queue/{req_id}/respond", {
            "response": "Paris is the capital of France",
            "reading_ms": 100,
            "composing_ms": 200,
        })
        t.join(timeout=5)

        usage = result["data"].get("usage", {})
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert usage["completion_tokens"] == 6  # word count: "Paris is the capital of France"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_server.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'deckard.server'`

- [ ] **Step 3: Implement server**

Create `deckard/server.py`:

```python
"""Deckard HTTP server — OpenAI-compatible endpoint with human-in-the-loop queue."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from deckard.types import DeckardConfig, QueuedRequest, Status

logger = logging.getLogger("deckard")


class DeckardServer:
    """Headless HTTP server that queues LLM requests for human response."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8421,
        db_path: str | None = None,
        simulate_latency: bool = False,
        chunk_delay_ms: int = 2,
    ):
        self.config = DeckardConfig(
            host=host,
            port=port,
            db_path=db_path or str(Path("~/.deckard/deckard.db").expanduser()),
            simulate_latency=simulate_latency,
            chunk_delay_ms=chunk_delay_ms,
        )
        self._queue: dict[str, dict] = {}  # id → {request, event, response}
        self._queue_lock = threading.Lock()
        self._db = self._init_db()
        self._server: ThreadingHTTPServer | None = None
        self.port: int | None = None

        # Recover abandoned requests from previous run
        self._recover_abandoned()

    def _init_db(self) -> sqlite3.Connection:
        path = Path(self.config.db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                messages TEXT NOT NULL,
                stream INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'in_progress', 'completed', 'abandoned')),
                response TEXT,
                responded_at TEXT,
                duration_ms INTEGER,
                reading_ms INTEGER,
                composing_ms INTEGER,
                estimated_prompt_tokens INTEGER,
                estimated_completion_tokens INTEGER,
                conversation_id TEXT
            )
        """)
        conn.commit()
        return conn

    def _recover_abandoned(self):
        cursor = self._db.execute(
            "SELECT id, created_at, model, messages, stream, status FROM requests WHERE status IN ('pending', 'in_progress')"
        )
        for row in cursor.fetchall():
            req_id, created_at, model, messages_json, stream, status = row
            self._queue[req_id] = {
                "request": QueuedRequest(
                    id=req_id,
                    created_at=created_at,
                    model=model,
                    messages=json.loads(messages_json),
                    stream=bool(stream),
                    status=Status.ABANDONED,
                ),
                "event": threading.Event(),
                "response": None,
                "abandoned": True,
            }
            self._db.execute(
                "UPDATE requests SET status = 'abandoned' WHERE id = ?", (req_id,)
            )
        self._db.commit()

    def serve(self):
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("deckard serving on %s:%d", self.config.host, self.port)
        self._server.serve_forever()

    def shutdown(self):
        # Mark pending/in_progress as abandoned
        with self._queue_lock:
            for req_id, entry in self._queue.items():
                if entry["request"].status in (Status.PENDING, Status.IN_PROGRESS):
                    self._db.execute(
                        "UPDATE requests SET status = 'abandoned' WHERE id = ?",
                        (req_id,),
                    )
            self._db.commit()
        if self._server:
            self._server.shutdown()

    def _handle_shutdown(self, signum, frame):
        logger.info("shutting down (signal %d)", signum)
        self.shutdown()

    def enqueue(self, messages: list[dict], model: str, stream: bool) -> str:
        req_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Conversation ID: SHA-256 of all message contents except last user message
        conv_parts = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.get("role") == "user":
                continue
            conv_parts.append(msg.get("content", ""))
        conv_id = hashlib.sha256("".join(conv_parts).encode()).hexdigest()[:16]

        prompt_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4

        request = QueuedRequest(
            id=req_id,
            created_at=now,
            model=model,
            messages=messages,
            stream=stream,
            status=Status.PENDING,
        )

        event = threading.Event()
        with self._queue_lock:
            self._queue[req_id] = {
                "request": request,
                "event": event,
                "response": None,
                "abandoned": False,
            }

        self._db.execute(
            """INSERT INTO requests
               (id, created_at, model, messages, stream, status, estimated_prompt_tokens, conversation_id)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (req_id, now, model, json.dumps(messages), int(stream), prompt_tokens, conv_id),
        )
        self._db.commit()

        logger.info("queued %s from model=%s (%d messages)", req_id, model, len(messages))
        return req_id

    def wait_for_response(self, req_id: str, timeout: float = 600) -> str | None:
        entry = self._queue.get(req_id)
        if not entry:
            return None
        entry["event"].wait(timeout=timeout)
        return entry.get("response")

    def respond(self, req_id: str, response: str, reading_ms: int = 0, composing_ms: int = 0) -> bool:
        with self._queue_lock:
            entry = self._queue.get(req_id)
            if not entry:
                return False
            if entry["request"].status == Status.COMPLETED:
                return False

        now = datetime.now(timezone.utc).isoformat()
        created = entry["request"].created_at
        duration_ms = int(
            (datetime.fromisoformat(now) - datetime.fromisoformat(created)).total_seconds() * 1000
        )
        completion_tokens = len(response.split())

        entry["response"] = response
        entry["request"].status = Status.IN_PROGRESS
        entry["request"].response = response
        entry["request"].responded_at = now
        entry["event"].set()

        # Note: status set to in_progress here. The HTTP handler sets completed
        # only after successful delivery to the client socket (Leveson item 2).
        self._db.execute(
            """UPDATE requests SET
               status = 'in_progress', response = ?, responded_at = ?,
               duration_ms = ?, reading_ms = ?, composing_ms = ?,
               estimated_completion_tokens = ?
               WHERE id = ?""",
            (response, now, duration_ms, reading_ms, composing_ms, completion_tokens, req_id),
        )
        self._db.commit()
        return True

    def mark_delivered(self, req_id: str):
        """Mark request as completed after response delivered to client socket."""
        self._db.execute(
            "UPDATE requests SET status = 'completed' WHERE id = ?", (req_id,)
        )
        self._db.commit()
        with self._queue_lock:
            entry = self._queue.get(req_id)
            if entry:
                entry["request"].status = Status.COMPLETED

    def mark_delivery_failed(self, req_id: str):
        """Response received from human but delivery to client failed."""
        self._db.execute(
            "UPDATE requests SET status = 'abandoned' WHERE id = ?", (req_id,)
        )
        self._db.commit()

    def get_queue(self) -> list[dict]:
        with self._queue_lock:
            result = []
            for entry in self._queue.values():
                req = entry["request"]
                if req.status in (Status.PENDING, Status.ABANDONED):
                    result.append({
                        "id": req.id,
                        "created_at": req.created_at,
                        "model": req.model,
                        "messages": req.messages,
                        "stream": req.stream,
                        "status": req.status.value,
                        "abandoned": entry.get("abandoned", False),
                    })
            result.sort(key=lambda r: (r["status"] != "abandoned", r["created_at"]))
            return result


def _make_handler(server: DeckardServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug(format, *args)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                queue = server.get_queue()
                self._send_json({"status": "ok", "pending": len(queue)})
            elif path == "/v1/models":
                self._send_json({
                    "object": "list",
                    "data": [{"id": "deckard", "object": "model", "owned_by": "human"}],
                })
            elif path == "/_deckard/queue":
                self._send_json(server.get_queue())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.rstrip("/")
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode()) if length > 0 else {}
            except (json.JSONDecodeError, ValueError):
                self._send_json({"error": "invalid JSON"}, 400)
                return

            if path == "/v1/chat/completions":
                self._handle_completion(body)
            elif path.startswith("/_deckard/queue/") and path.endswith("/respond"):
                req_id = path[len("/_deckard/queue/"):-len("/respond")]
                self._handle_respond(req_id, body)
            else:
                self._send_json({"error": "not found"}, 404)

        def _handle_completion(self, body):
            messages = body.get("messages", [])
            model = body.get("model", "deckard")
            stream = body.get("stream", False)

            req_id = server.enqueue(messages, model, stream)
            response_text = server.wait_for_response(req_id)

            if response_text is None:
                self._send_json({"error": "timeout"}, 504)
                return

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            completion_tokens = len(response_text.split())

            if stream:
                self._stream_response(
                    completion_id, created, model, response_text, req_id
                )
            else:
                self._send_json({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
                server.mark_delivered(req_id)

        def _stream_response(self, completion_id, created, model, text, req_id):
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # Initial chunk: role
                self._write_sse_chunk(completion_id, created, model, None, None)

                # Content chunks: one per word
                words = text.split()
                delay = server.config.chunk_delay_ms / 1000.0
                if server.config.simulate_latency:
                    delay = 0.030

                for i, word in enumerate(words):
                    content = word + (" " if i < len(words) - 1 else "")
                    self._write_sse_chunk(completion_id, created, model, content, None)
                    time.sleep(delay)

                # Final chunk: finish_reason
                self._write_sse_chunk(completion_id, created, model, None, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

                server.mark_delivered(req_id)
            except BrokenPipeError:
                server.mark_delivery_failed(req_id)

        def _write_sse_chunk(self, completion_id, created, model, content, finish_reason):
            delta = {}
            if content is None and finish_reason is None:
                delta = {"role": "assistant"}
            elif content is not None:
                delta = {"content": content}

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        def _handle_respond(self, req_id, body):
            response = body.get("response")
            if response is None:
                self._send_json({"error": "missing 'response' field"}, 400)
                return

            reading_ms = body.get("reading_ms", 0)
            composing_ms = body.get("composing_ms", 0)

            success = server.respond(req_id, response, reading_ms, composing_ms)
            if success:
                self._send_json({"status": "ok"})
            else:
                # Could be 404 (not found) or 409 (already completed)
                entry = server._queue.get(req_id)
                if entry is None:
                    self._send_json({"error": "request not found"}, 404)
                else:
                    self._send_json({"error": "already completed"}, 409)

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    return Handler
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_server.py -v
```

Expected: All pass

- [ ] **Step 5: Run full test suite**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/ -v
```

Expected: All tests pass (types + server)

- [ ] **Step 6: Commit**

```bash
cd /home/melek/workshop/deckard
git add deckard/server.py tests/test_server.py
git commit -m "feat: HTTP server with OpenAI-compatible endpoints, queue, SQLite logging, SSE streaming"
```

---

### Task 3: SSE Streaming Tests

**Files:**
- Modify: `tests/test_server.py`

- [ ] **Step 1: Write SSE streaming test**

Add to `tests/test_server.py`:

```python
class TestSSEStreaming:
    def test_streaming_response(self, server):
        """stream=true returns SSE events with word-by-word chunks."""
        body = {
            "model": "deckard",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        result = {"chunks": []}
        def do_request():
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{server}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        result["chunks"].append(chunk)
                    elif line == "data: [DONE]":
                        result["done"] = True

        t = threading.Thread(target=do_request)
        t.start()

        for _ in range(50):
            queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        req_id = queue[0]["id"]

        _post_json(f"{server}/_deckard/queue/{req_id}/respond", {
            "response": "Hello world",
            "reading_ms": 0,
            "composing_ms": 0,
        })
        t.join(timeout=10)

        # First chunk: role
        assert result["chunks"][0]["choices"][0]["delta"].get("role") == "assistant"
        # Middle chunks: content words
        contents = [
            c["choices"][0]["delta"].get("content", "")
            for c in result["chunks"][1:-1]
        ]
        assert "".join(contents).strip() == "Hello world"
        # Last chunk: finish_reason
        assert result["chunks"][-1]["choices"][0]["finish_reason"] == "stop"
        # Done sentinel
        assert result.get("done") is True

    def test_streaming_content_type(self, server):
        """Streaming response has text/event-stream content type."""
        body = {
            "model": "deckard",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
        result = {}
        def do_request():
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{server}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result["content_type"] = resp.headers.get("Content-Type")
                for line in resp:
                    pass  # consume

        t = threading.Thread(target=do_request)
        t.start()

        for _ in range(50):
            queue = _get_json(f"{server}/_deckard/queue")
            if queue:
                break
            time.sleep(0.1)
        req_id = queue[0]["id"]

        _post_json(f"{server}/_deckard/queue/{req_id}/respond", {
            "response": "ok",
            "reading_ms": 0,
            "composing_ms": 0,
        })
        t.join(timeout=10)

        assert "text/event-stream" in result.get("content_type", "")
```

- [ ] **Step 2: Run streaming tests**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_server.py::TestSSEStreaming -v
```

Expected: All pass (server already implements streaming)

- [ ] **Step 3: Commit**

```bash
cd /home/melek/workshop/deckard
git add tests/test_server.py
git commit -m "test: SSE streaming verification — word chunks, content type, DONE sentinel"
```

---

### Task 4: Client TUI — Layout + Request Display

**Files:**
- Create: `deckard/client.py`
- Create: `deckard/app.tcss`

- [ ] **Step 1: Create Textual stylesheet**

Create `deckard/app.tcss`:

```css
Screen {
    layout: vertical;
}

#status-bar {
    dock: top;
    height: 1;
    background: $surface-lighten-1;
    color: $text;
    padding: 0 1;
}

#request-list {
    height: 1fr;
    min-height: 4;
    border-bottom: solid $surface-lighten-2;
    padding: 0 1;
}

#request-detail {
    height: 2fr;
    border-bottom: solid $surface-lighten-2;
    padding: 0 1;
    overflow-y: auto;
}

#response-input {
    height: auto;
    min-height: 3;
    max-height: 10;
    padding: 0 1;
}

TextArea {
    border: none;
    background: $surface;
}

TextArea:focus {
    border: none;
}
```

- [ ] **Step 2: Implement client TUI**

Create `deckard/client.py`:

```python
"""Deckard TUI client — human interface for responding to LLM requests."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, ListView, ListItem, RichLog, TextArea, Label

logger = logging.getLogger("deckard.client")

_IDLE_TIMEOUT_MS = 120_000  # 2 minutes


class StatusBar(Static):
    """Top bar: connection status, pending count, total count."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected = False
        self.pending = 0
        self.total = 0
        self._render()

    def _render(self):
        dot = "[green]●[/green] connected" if self.connected else "[red]●[/red] disconnected"
        self.update(f" deckard  ─  {dot}  ─  {self.pending} pending  ─  {self.total} total")

    def set_connected(self, connected: bool):
        self.connected = connected
        self._render()

    def set_counts(self, pending: int, total: int):
        self.pending = pending
        self.total = total
        self._render()


class RequestDetail(RichLog):
    """Shows full message array with colored role indicators."""

    def show_request(self, messages: list[dict], abandoned: bool = False):
        self.clear()
        if abandoned:
            self.write("[yellow]↻ recovered — no client waiting[/yellow]")
            self.write("")

        role_colors = {
            "system": "dim",
            "user": "cyan",
            "assistant": "green",
        }
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            color = role_colors.get(role, "white")

            if i > 0:
                self.write(f"[{color}]{'─' * 50}[/{color}]")
            self.write(f"[{color}]┃ {role}:[/{color}]")
            for line in content.split("\n"):
                self.write(f"[{color}]┃[/{color}] {line}")
        self.write("")


class DeckardClient(App):
    """Deckard TUI — human-as-LLM interface."""

    CSS_PATH = "app.tcss"
    TITLE = "deckard"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, host: str = "127.0.0.1", port: int = 8421):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self._requests: list[dict] = []
        self._selected_id: str | None = None
        self._reading_start: float | None = None
        self._composing_start: float | None = None
        self._reading_ms: int = 0
        self._composing_ms: int = 0
        self._last_keystroke: float = 0
        self._idle_accumulated_ms: int = 0

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        yield ListView(id="request-list")
        yield RequestDetail(highlight=True, markup=True, wrap=True, id="request-detail")
        yield Vertical(
            TextArea(id="response-input"),
            id="input-area",
        )

    def on_mount(self):
        # Hide detail and input until a request is selected
        self.query_one("#request-detail").display = False
        self.query_one("#input-area").display = False
        self._poll_queue()
        self.set_interval(2, self._poll_queue)

    @work(exclusive=True, thread=True)
    def _poll_queue(self):
        status_bar = self.query_one("#status-bar", StatusBar)
        try:
            with urllib.request.urlopen(f"{self.base_url}/_deckard/queue", timeout=3) as resp:
                data = json.loads(resp.read().decode())
            self._requests = data
            self.call_from_thread(status_bar.set_connected, True)
            self.call_from_thread(status_bar.set_counts, len(data), len(data))
            self.call_from_thread(self._refresh_list)
        except Exception:
            self.call_from_thread(status_bar.set_connected, False)

    def _refresh_list(self):
        listview = self.query_one("#request-list", ListView)
        listview.clear()
        for i, req in enumerate(self._requests):
            ts = req["created_at"]
            if "T" in ts:
                ts = ts.split("T")[1][:5]
            model = req.get("model", "?")[:8]
            last_msg = ""
            messages = req.get("messages", [])
            if messages:
                last_msg = messages[-1].get("content", "")[:40]
            prefix = "↻ " if req.get("abandoned") else ""
            label = f"{prefix}[{i+1}] {ts}  {model:<8}  \"{last_msg}…\""
            listview.append(ListItem(Label(label), name=req["id"]))

    def on_list_view_selected(self, event: ListView.Selected):
        req_id = event.item.name
        req = next((r for r in self._requests if r["id"] == req_id), None)
        if not req:
            return

        self._selected_id = req_id

        # Show detail
        detail = self.query_one("#request-detail", RequestDetail)
        detail.display = True
        detail.show_request(req["messages"], abandoned=req.get("abandoned", False))

        # Show input
        input_area = self.query_one("#input-area")
        input_area.display = True
        text_area = self.query_one("#response-input", TextArea)
        text_area.text = ""
        text_area.focus()

        # Start reading timer
        self._reading_start = time.monotonic()
        self._composing_start = None
        self._reading_ms = 0
        self._composing_ms = 0
        self._last_keystroke = time.monotonic()
        self._idle_accumulated_ms = 0

    def on_text_area_changed(self, event):
        now = time.monotonic()

        # Idle guard: if gap > 2 minutes, don't count it
        if self._last_keystroke and (now - self._last_keystroke) > (_IDLE_TIMEOUT_MS / 1000):
            self._idle_accumulated_ms += int((now - self._last_keystroke) * 1000) - _IDLE_TIMEOUT_MS

        # Transition from reading to composing on first keystroke
        if self._composing_start is None and self._reading_start is not None:
            self._reading_ms = int((now - self._reading_start) * 1000) - self._idle_accumulated_ms
            self._reading_ms = max(0, self._reading_ms)
            self._composing_start = now
            self._idle_accumulated_ms = 0

        self._last_keystroke = now

    async def action_back(self):
        self.query_one("#request-detail").display = False
        self.query_one("#input-area").display = False
        self._selected_id = None
        self.query_one("#request-list", ListView).focus()

    def key_ctrl_enter(self):
        """Submit response on Ctrl+Enter."""
        if not self._selected_id:
            return
        text_area = self.query_one("#response-input", TextArea)
        response = text_area.text.strip()
        if not response:
            return

        # Finalize composing timer
        now = time.monotonic()
        if self._composing_start:
            idle_gap = 0
            if self._last_keystroke and (now - self._last_keystroke) > (_IDLE_TIMEOUT_MS / 1000):
                idle_gap = int((now - self._last_keystroke) * 1000) - _IDLE_TIMEOUT_MS
            self._composing_ms = int((now - self._composing_start) * 1000) - self._idle_accumulated_ms - idle_gap
            self._composing_ms = max(0, self._composing_ms)

        self._submit_response(self._selected_id, response, self._reading_ms, self._composing_ms)

    @work(exclusive=True, thread=True)
    def _submit_response(self, req_id: str, response: str, reading_ms: int, composing_ms: int):
        try:
            body = json.dumps({
                "response": response,
                "reading_ms": reading_ms,
                "composing_ms": composing_ms,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/_deckard/queue/{req_id}/respond",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read().decode())
        except Exception as e:
            logger.error("submit failed: %s", e)
            return

        # Reset view
        def _reset():
            self.query_one("#request-detail").display = False
            self.query_one("#input-area").display = False
            self.query_one("#response-input", TextArea).text = ""
            self._selected_id = None
            self.query_one("#request-list", ListView).focus()

        self.call_from_thread(_reset)
        # Trigger immediate re-poll
        self._poll_queue()
```

- [ ] **Step 3: Verify TUI loads without errors**

```bash
cd /home/melek/workshop/deckard && python3 -c "from deckard.client import DeckardClient; print('import ok')"
```

Expected: `import ok` (doesn't start the app, just validates imports)

- [ ] **Step 4: Commit**

```bash
cd /home/melek/workshop/deckard
git add deckard/client.py deckard/app.tcss
git commit -m "feat: Textual TUI client with request list, colored detail view, response input, duration tracking"
```

---

### Task 5: Client TUI — Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""Integration test: server + programmatic client interaction."""
import json
import threading
import time
import urllib.request
import pytest
from deckard.server import DeckardServer


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
    yield srv
    srv.shutdown()


def test_full_roundtrip(server):
    """Client sends request → appears in queue → respond → client gets answer."""
    base = f"http://127.0.0.1:{server.port}"

    # Simulate OpenAI client sending a request
    result = {}
    def client_request():
        body = json.dumps({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result["response"] = json.loads(resp.read().decode())

    t = threading.Thread(target=client_request)
    t.start()

    # Simulate TUI polling queue
    for _ in range(50):
        with urllib.request.urlopen(f"{base}/_deckard/queue", timeout=3) as resp:
            queue = json.loads(resp.read().decode())
        if queue:
            break
        time.sleep(0.1)

    assert len(queue) == 1
    assert queue[0]["messages"][1]["content"] == "What is 2+2?"

    # Simulate TUI submitting response
    req_id = queue[0]["id"]
    body = json.dumps({
        "response": "4",
        "reading_ms": 100,
        "composing_ms": 200,
    }).encode()
    req = urllib.request.Request(
        f"{base}/_deckard/queue/{req_id}/respond",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        json.loads(resp.read().decode())

    # Client should have gotten the response
    t.join(timeout=10)
    assert result["response"]["choices"][0]["message"]["content"] == "4"
    assert result["response"]["usage"]["completion_tokens"] == 1  # "4" is one word


def test_abandoned_recovery(server, tmp_path):
    """Requests survive server restart as abandoned."""
    import sqlite3
    base = f"http://127.0.0.1:{server.port}"

    # Send a request but don't respond
    def client_request():
        try:
            body = json.dumps({
                "model": "test",
                "messages": [{"role": "user", "content": "abandoned"}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # timeout expected

    t = threading.Thread(target=client_request)
    t.start()

    for _ in range(50):
        with urllib.request.urlopen(f"{base}/_deckard/queue", timeout=3) as resp:
            queue = json.loads(resp.read().decode())
        if queue:
            break
        time.sleep(0.1)

    assert len(queue) == 1

    # Shutdown marks as abandoned
    server.shutdown()
    t.join(timeout=5)

    # Check SQLite directly
    db_path = server.config.db_path
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT status FROM requests").fetchall()
    assert any(r[0] == "abandoned" for r in rows)
    conn.close()
```

- [ ] **Step 2: Run integration tests**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/test_integration.py -v
```

Expected: All pass

- [ ] **Step 3: Run full test suite**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
cd /home/melek/workshop/deckard
git add tests/test_integration.py
git commit -m "test: full roundtrip integration + abandoned request recovery"
```

---

### Task 6: CLI Entry + Server Discovery

**Files:**
- Create: `deckard/__main__.py`

- [ ] **Step 1: Implement CLI entry**

Create `deckard/__main__.py`:

```python
"""Deckard CLI entry point."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

from deckard.types import DeckardConfig


def _check_server(host: str, port: int) -> bool:
    """Return True if a deckard server is responding at host:port."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


def _start_server_background(args: argparse.Namespace) -> bool:
    """Fork a deckard server as a background process. Returns True if server comes up."""
    cmd = [sys.executable, "-m", "deckard", "serve", "--port", str(args.port)]
    if args.simulate_latency:
        cmd.append("--simulate-latency")

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Poll until ready
    for _ in range(50):
        if _check_server(args.host, args.port):
            return True
        time.sleep(0.1)
    return False


def _run_serve(args: argparse.Namespace):
    """Run headless server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    from deckard.server import DeckardServer

    srv = DeckardServer(
        host=args.host,
        port=args.port,
        simulate_latency=args.simulate_latency,
        chunk_delay_ms=30 if args.simulate_latency else 2,
    )
    srv.serve()


def _run_client(args: argparse.Namespace):
    """Run TUI client, connecting to existing server."""
    from deckard.client import DeckardClient

    app = DeckardClient(host=args.host, port=args.port)
    app.run()


def _run_combined(args: argparse.Namespace):
    """Find or start server, then open client."""
    if not _check_server(args.host, args.port):
        print(f"Starting deckard server on {args.host}:{args.port}...")
        if not _start_server_background(args):
            print("Failed to start server.", file=sys.stderr)
            sys.exit(1)

    _run_client(args)


def main():
    parser = argparse.ArgumentParser(
        prog="deckard",
        description="Human-as-LLM endpoint — OpenAI-compatible API server with TUI",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8421, help="Server port (default: 8421)")
    parser.add_argument("--simulate-latency", action="store_true", help="30ms/chunk streaming delay")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run headless server")
    subparsers.add_parser("client", help="Connect TUI to running server")

    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "client":
        _run_client(args)
    else:
        _run_combined(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI help**

```bash
cd /home/melek/workshop/deckard && python3 -m deckard --help
```

Expected: Shows help with `serve`, `client` subcommands and `--port`, `--simulate-latency` flags.

- [ ] **Step 3: Test serve subcommand starts**

```bash
cd /home/melek/workshop/deckard && timeout 3 python3 -m deckard serve --port 8422 || true
```

Expected: Starts serving, killed by timeout after 3 seconds. Output includes `deckard serving on 127.0.0.1:8422`.

- [ ] **Step 4: Commit**

```bash
cd /home/melek/workshop/deckard
git add deckard/__main__.py
git commit -m "feat: CLI entry — deckard, deckard serve, deckard client with server discovery"
```

---

### Task 7: CLAUDE.md + Final Polish

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create CLAUDE.md**

Create `CLAUDE.md`:

```markdown
# CLAUDE.md

## What This Is

Deckard is a human-as-LLM endpoint. It's an OpenAI-compatible HTTP server where a human reads prompts and types responses via a TUI. Any client that speaks the OpenAI chat completions API can point at deckard instead of an inference server.

## Commands

```bash
# Install (editable)
pip install -e .

# Run (starts server + opens TUI)
deckard

# Run server only (headless, logs to stdout)
deckard serve

# Connect TUI to running server
deckard client

# Options
deckard --port 9000              # override port (default: 8421)
deckard --simulate-latency       # 30ms/chunk streaming (default: 2ms)
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Architecture

- `deckard/server.py` — HTTP server, request queue, SQLite logging, SSE streaming
- `deckard/client.py` — Textual TUI
- `deckard/types.py` — Shared types (Status, QueuedRequest, DeckardConfig)
- `deckard/__main__.py` — CLI entry with server discovery

Data: `~/.deckard/deckard.db` (SQLite)

## Dependencies

- `textual` (TUI framework). Everything else stdlib.
- Python 3.10+
```

- [ ] **Step 2: Run full test suite one final time**

```bash
cd /home/melek/workshop/deckard && python3 -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
cd /home/melek/workshop/deckard
git add CLAUDE.md
git commit -m "docs: CLAUDE.md with commands, testing, and architecture overview"
```

---

## Parallelism Notes

| Task | Depends On | Can Parallel With |
|------|-----------|-------------------|
| 1 (Types + scaffold) | — | — |
| 2 (Server) | 1 | — |
| 3 (SSE tests) | 2 | 4 |
| 4 (Client TUI) | 1 | 3 |
| 5 (Integration test) | 2, 4 | — |
| 6 (CLI entry) | 2, 4 | 5 |
| 7 (CLAUDE.md) | 1-6 | — |
