"""Deckard HTTP server — OpenAI-compatible endpoints with human-in-the-loop queue."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any

from deckard.types import Status, QueuedRequest, DeckardConfig

# Maximum request body size (10 MB)
MAX_BODY_SIZE = 10 * 1024 * 1024

# Maximum queue depth — reject with 429 if exceeded
MAX_QUEUE_DEPTH = 100

# Default timeout for waiting on human response (seconds)
DEFAULT_WAIT_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Conversation ID
# ---------------------------------------------------------------------------

def conversation_id(messages: list[dict]) -> str:
    """SHA-256 of all message contents except the last user message, truncated to 16 chars."""
    context = messages[:-1] if messages else []
    blob = "".join(m.get("content", "") for m in context)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_prompt_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


def estimate_completion_tokens(response: str) -> int:
    return len(response.split())


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_SCHEMA = """\
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
);
"""


def _init_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _insert_request(conn: sqlite3.Connection, db_lock: threading.Lock, req: QueuedRequest, conv_id: str) -> None:
    with db_lock:
        conn.execute(
            "INSERT INTO requests (id, created_at, model, messages, stream, status, "
            "estimated_prompt_tokens, conversation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                req.id,
                req.created_at,
                req.model,
                json.dumps(req.messages),
                1 if req.stream else 0,
                req.status.value,
                estimate_prompt_tokens(req.messages),
                conv_id,
            ),
        )
        conn.commit()


def _complete_request(
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    req_id: str,
    response: str,
    responded_at: str,
    duration_ms: int,
    reading_ms: int | None,
    composing_ms: int | None,
) -> None:
    with db_lock:
        conn.execute(
            "UPDATE requests SET status=?, response=?, responded_at=?, duration_ms=?, "
            "reading_ms=?, composing_ms=?, estimated_completion_tokens=? WHERE id=?",
            (
                Status.COMPLETED.value,
                response,
                responded_at,
                duration_ms,
                reading_ms,
                composing_ms,
                estimate_completion_tokens(response),
                req_id,
            ),
        )
        conn.commit()


def _fail_request(
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    req_id: str,
    response: str | None,
    responded_at: str | None,
) -> None:
    """Mark a request as abandoned after delivery failure. Preserves the response text."""
    with db_lock:
        conn.execute(
            "UPDATE requests SET status=?, response=?, responded_at=? WHERE id=?",
            (Status.ABANDONED.value, response, responded_at, req_id),
        )
        conn.commit()


def _abandon_pending(conn: sqlite3.Connection) -> None:
    """Mark all pending/in_progress rows as abandoned."""
    conn.execute(
        "UPDATE requests SET status=? WHERE status IN (?, ?)",
        (Status.ABANDONED.value, Status.PENDING.value, Status.IN_PROGRESS.value),
    )
    conn.commit()


def _load_abandoned(conn: sqlite3.Connection) -> list[QueuedRequest]:
    """Load abandoned rows to re-queue on startup."""
    cursor = conn.execute(
        "SELECT id, created_at, model, messages, stream, status FROM requests WHERE status=?",
        (Status.ABANDONED.value,),
    )
    results = []
    for row in cursor.fetchall():
        results.append(QueuedRequest(
            id=row[0],
            created_at=row[1],
            model=row[2],
            messages=json.loads(row[3]),
            stream=bool(row[4]),
            status=Status.ABANDONED,
        ))
    return results


# ---------------------------------------------------------------------------
# In-memory queue
# ---------------------------------------------------------------------------

class _QueueEntry:
    __slots__ = (
        "request", "event", "abandoned",
        "duration_ms", "reading_ms", "composing_ms",
    )

    def __init__(self, request: QueuedRequest, *, abandoned: bool = False):
        self.request = request
        self.event = threading.Event()
        self.abandoned = abandoned
        self.duration_ms: int = 0
        self.reading_ms: int | None = None
        self.composing_ms: int | None = None
        if abandoned:
            # Don't block on abandoned — they're shown in queue but no HTTP client waits
            self.event.set()


class RequestQueue:
    """Thread-safe in-memory queue of pending human-response requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _QueueEntry] = {}

    def add(self, req: QueuedRequest, *, abandoned: bool = False) -> _QueueEntry:
        entry = _QueueEntry(req, abandoned=abandoned)
        with self._lock:
            self._entries[req.id] = entry
        return entry

    def get(self, req_id: str) -> _QueueEntry | None:
        with self._lock:
            return self._entries.get(req_id)

    def remove(self, req_id: str) -> None:
        with self._lock:
            self._entries.pop(req_id, None)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if not e.event.is_set())

    def snapshot(self) -> list[dict]:
        with self._lock:
            items = []
            for e in self._entries.values():
                r = e.request
                items.append({
                    "id": r.id,
                    "created_at": r.created_at,
                    "model": r.model,
                    "messages": r.messages,
                    "stream": r.stream,
                    "status": r.status.value,
                    "abandoned": e.abandoned,
                })
            return items

    def abandon_all(self) -> list[str]:
        """Set events on all pending entries so blocked handlers unblock. Returns IDs."""
        with self._lock:
            ids = []
            for req_id, entry in self._entries.items():
                if not entry.event.is_set():
                    entry.abandoned = True
                    entry.request.status = Status.ABANDONED
                    entry.event.set()
                    ids.append(req_id)
            return ids


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Stateless handler — accesses server state via self.server."""

    # Silence per-request log lines
    def log_message(self, format: str, *args: Any) -> None:
        pass

    # ---- routing ----

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/_deckard/queue":
            self._handle_queue_list()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_completions()
        elif self.path.startswith("/_deckard/queue/") and self.path.endswith("/respond"):
            req_id = self.path[len("/_deckard/queue/"):-len("/respond")]
            self._handle_respond(req_id)
        else:
            self._send_json(404, {"error": "not found"})

    # ---- public endpoints ----

    def _handle_health(self) -> None:
        srv: DeckardServer = self.server  # type: ignore[assignment]
        self._send_json(200, {"status": "ok", "pending": srv.queue.pending_count()})

    def _handle_models(self) -> None:
        self._send_json(200, {
            "object": "list",
            "data": [{"id": "deckard", "object": "model", "owned_by": "human"}],
        })

    def _handle_completions(self) -> None:
        srv: DeckardServer = self.server  # type: ignore[assignment]

        body = self._read_body()
        if body is None:
            return

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", "deckard")

        req_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        req = QueuedRequest(
            id=req_id,
            created_at=now,
            model=model,
            messages=messages,
            stream=stream,
            status=Status.PENDING,
        )

        # I1: reject if queue is full
        if srv.queue.pending_count() >= MAX_QUEUE_DEPTH:
            self._send_json(429, {"error": "queue full", "max": MAX_QUEUE_DEPTH})
            return

        conv_id = conversation_id(messages)
        _insert_request(srv.db, srv.db_lock, req, conv_id)

        entry = srv.queue.add(req)

        # I2: block with timeout so threads don't hang forever
        entry.event.wait(timeout=DEFAULT_WAIT_TIMEOUT)

        if not entry.event.is_set():
            # Timeout — no response from human
            srv.queue.remove(req_id)
            self._send_json(504, {"error": "timeout waiting for human response"})
            return

        if entry.abandoned:
            self._send_json(503, {"error": "server shutting down"})
            srv.queue.remove(req_id)
            return

        # Build response
        response_text = req.response or ""
        prompt_tok = estimate_prompt_tokens(messages)
        completion_tok = estimate_completion_tokens(response_text)

        if stream:
            self._stream_response(req_id, model, response_text, prompt_tok, completion_tok, srv)
        else:
            self._send_non_streaming(req_id, model, response_text, prompt_tok, completion_tok, srv)

        srv.queue.remove(req_id)

    def _send_non_streaming(
        self, req_id: str, model: str, text: str,
        prompt_tok: int, completion_tok: int, srv: "DeckardServer",
    ) -> None:
        payload = {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "total_tokens": prompt_tok + completion_tok,
            },
        }
        try:
            self._send_json(200, payload)
            # Leveson: mark completed only after successful delivery
            self._mark_delivered(srv, req_id)
        except BrokenPipeError:
            self._mark_delivery_failed(srv, req_id)

    def _stream_response(
        self, req_id: str, model: str, text: str,
        prompt_tok: int, completion_tok: int, srv: "DeckardServer",
    ) -> None:
        chunk_delay = srv.config.chunk_delay_ms / 1000.0

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def _sse_chunk(data: dict) -> None:
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

            created = int(time.time())

            # First chunk: role
            _sse_chunk({
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            })
            time.sleep(chunk_delay)

            # Content chunks: one per word
            words = text.split()
            for i, word in enumerate(words):
                content = word + " " if i < len(words) - 1 else word
                _sse_chunk({
                    "id": f"chatcmpl-{req_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }],
                })
                time.sleep(chunk_delay)

            # Final chunk: stop
            _sse_chunk({
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "total_tokens": prompt_tok + completion_tok,
                },
            })

            # Sentinel
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

            # Leveson: mark completed only after full delivery
            self._mark_delivered(srv, req_id)

            # Close connection so client sees end-of-stream
            self.close_connection = True

        except BrokenPipeError:
            self._mark_delivery_failed(srv, req_id)

    def _mark_delivered(self, srv: "DeckardServer", req_id: str) -> None:
        entry = srv.queue.get(req_id)
        if entry and entry.request.responded_at:
            _complete_request(
                srv.db,
                srv.db_lock,
                req_id,
                entry.request.response or "",
                entry.request.responded_at,
                entry.duration_ms,
                entry.reading_ms,
                entry.composing_ms,
            )

    def _mark_delivery_failed(self, srv: "DeckardServer", req_id: str) -> None:
        entry = srv.queue.get(req_id)
        if entry:
            _fail_request(
                srv.db, srv.db_lock, req_id,
                entry.request.response, entry.request.responded_at,
            )

    # ---- internal TUI endpoints ----

    def _handle_queue_list(self) -> None:
        srv: DeckardServer = self.server  # type: ignore[assignment]
        self._send_json(200, srv.queue.snapshot())

    def _handle_respond(self, req_id: str) -> None:
        srv: DeckardServer = self.server  # type: ignore[assignment]

        body = self._read_body()
        if body is None:
            return

        if "response" not in body:
            self._send_json(400, {"error": "missing 'response' field"})
            return

        # Atomic check-and-respond under queue lock (fixes C1 TOCTOU race)
        with srv.queue._lock:
            entry = srv.queue._entries.get(req_id)
            if entry is None:
                self._send_json(404, {"error": "request not found"})
                return

            if entry.event.is_set():
                self._send_json(409, {"error": "already responded"})
                return

            response_text = body["response"]
            reading_ms = body.get("reading_ms")
            composing_ms = body.get("composing_ms")
            responded_at = datetime.now(timezone.utc).isoformat()

            created = datetime.fromisoformat(entry.request.created_at)
            now = datetime.now(timezone.utc)
            duration_ms = int((now - created).total_seconds() * 1000)

            entry.request.response = response_text
            entry.request.responded_at = responded_at
            entry.request.status = Status.COMPLETED
            entry.duration_ms = duration_ms
            entry.reading_ms = reading_ms
            entry.composing_ms = composing_ms

            entry.event.set()

        self._send_json(200, {"status": "ok", "id": req_id})

    # ---- helpers ----

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "empty body"})
            return None
        if length > MAX_BODY_SIZE:
            self._send_json(413, {"error": "payload too large", "max_bytes": MAX_BODY_SIZE})
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return None

    def _send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new thread."""
    daemon_threads = True


class DeckardServer(_ThreadedHTTPServer):
    """Deckard HTTP server wrapping HTTPServer with queue and SQLite."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8421,
        db_path: str = "~/.deckard/deckard.db",
        chunk_delay_ms: int = 2,
    ):
        self.config = DeckardConfig(
            host=host,
            port=port,
            db_path=db_path,
            chunk_delay_ms=chunk_delay_ms,
        )
        resolved_db = os.path.expanduser(db_path)
        self.db = _init_db(resolved_db)
        self.db_lock = threading.Lock()
        self.queue = RequestQueue()

        # Recover abandoned requests from previous run
        _abandon_pending(self.db)
        for req in _load_abandoned(self.db):
            self.queue.add(req, abandoned=True)

        super().__init__((host, port), _Handler)

        # If port was 0, record the actual port
        if port == 0:
            self.config.port = self.server_address[1]

        # Expose actual port as attribute for convenience
        self.port = self.server_address[1]

    def serve(self) -> None:
        """Serve until shutdown() is called."""
        self.serve_forever()

    def shutdown(self) -> None:
        """Graceful shutdown: abandon pending, close DB, stop serving. Idempotent."""
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True

        # Unblock any waiting handlers
        abandoned = self.queue.abandon_all()

        # Mark pending/in_progress in DB as abandoned
        _abandon_pending(self.db)

        super().shutdown()
        self.db.close()
