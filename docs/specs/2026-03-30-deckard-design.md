# Deckard: Human-as-LLM Endpoint — Design Spec

*2026-03-30*

**Goal:** A TUI tool where a human responds to LLM API requests. Any OpenAI-compatible client points at deckard instead of an inference server. The client can't tell the difference.

**Dependency:** `textual` (TUI framework). Everything else stdlib.

**Named after:** Rick Deckard (Blade Runner) — the boundary between human and artificial is undecidable at the interface. Bonus: Deckard Cain (Diablo) — "Stay awhile and listen."

---

## Architecture

Four modules:

| File | Responsibility |
|------|---------------|
| `server.py` | HTTP server, request queue, SQLite logging, SSE streaming |
| `client.py` | Textual TUI — request list, detail view, response composition |
| `types.py` | QueuedRequest, DeckardConfig, Status enum |
| `__main__.py` | CLI entry, arg parsing, server discovery |

The server is always headless. The client is always a separate connection to the server. `deckard` (no subcommand) starts the server if needed and opens the client.

```
Any OpenAI-compatible client (curl, openai-python, etc.)
    |
    |  POST /v1/chat/completions
    v
Deckard server (localhost:8421)
    |
    |  Queue request, block HTTP thread
    v
Deckard TUI (client.py)
    |
    |  Human reads prompt, types response
    |  POST /_deckard/queue/{id}/respond
    v
Server unblocks, streams response to original client
```

---

## Server (`server.py`)

### Public Endpoints (OpenAI-compatible)

**`POST /v1/chat/completions`** — Primary endpoint. Accepts OpenAI-format request:

```json
{
  "model": "anything",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": true,
  "temperature": 0.0
}
```

Behavior:
1. Assign UUID, store in queue as `pending`
2. Log to SQLite
3. Block on `threading.Event` — wait for human response
4. When response arrives:
   - If `stream: false` → return standard JSON completion
   - If `stream: true` → stream response as SSE chunks (word-split, 2ms floor delay)
5. If server shuts down while blocking → client gets connection closed

**`GET /v1/models`** — Returns model list:

```json
{
  "object": "list",
  "data": [{"id": "deckard", "object": "model", "owned_by": "human"}]
}
```

**`GET /health`** — Returns `{"status": "ok", "pending": N}`.

### Internal Endpoints (TUI protocol)

Prefixed with `/_deckard/` to separate from the public OpenAI API.

**`GET /_deckard/queue`** — Returns pending and abandoned requests as JSON array. TUI polls this.

**`POST /_deckard/queue/{id}/respond`** — TUI submits human's response. Body: `{"response": "...", "reading_ms": N, "composing_ms": N}`. Sets the `threading.Event`, unblocks the waiting HTTP handler. Returns 200 on success, 404 if request ID not found, 409 if already completed, 400 if malformed JSON.

### SSE Streaming

When `stream: true` in the original request, the response is streamed in OpenAI SSE format:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"deckard","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"deckard","choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"deckard","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`.

Chunking: split response by words. Each word (plus trailing space) is one SSE event.

Delay: 2ms floor between chunks (protocol correctness). `--simulate-latency` increases to ~30ms/chunk (realistic inference speed).

### Queue Mechanics

- In-memory dict keyed by request UUID
- Each entry has a `threading.Event`
- HTTP handler thread blocks on `event.wait()`
- TUI submits response → server sets event → handler wakes, streams/returns
- No lock/claim mechanism — single TUI, single queue
- On shutdown (SIGTERM/SIGINT): pending and in_progress requests marked `abandoned` in SQLite
- `completed` status written to SQLite only after response is fully delivered to the client socket. If socket write fails, response is preserved in the record but status reflects delivery failure. The in-memory Event unblocks the handler immediately; the durable record reflects ground truth.

### SQLite Logging

Database: `~/.deckard/deckard.db`

```sql
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
```

**Token estimation:**
- `estimated_prompt_tokens`: `sum(len(msg["content"]) for msg in messages) // 4`
- `estimated_completion_tokens`: `len(response.split())` (word count)

**Conversation ID:** SHA-256 of concatenated message contents (minus last user message). Same history → same ID.

**Duration fields:**
- `duration_ms`: wall clock from request arrival to response submission
- `reading_ms`: active time viewing the request detail before first keystroke in response input (idle-guarded)
- `composing_ms`: active time typing the response until submit (idle-guarded)
- Idle guard: 2 minutes of no keystrokes pauses the timer. Any keystroke resumes.

---

## Client TUI (`client.py`)

### Layout

```
┌──────────────────────────────────────────────────┐
│ deckard  ● connected  ─  3 pending  ─  12 total  │  StatusBar
├──────────────────────────────────────────────────┤
│ [1] 14:32  gpt-4    "What is the capital of…"    │  RequestList
│ [2] 14:33  deckard  "Summarize the following…"   │  (scrollable)
│ [3] 14:33  deckard  "Generate a Dafny spec…"     │
├──────────────────────────────────────────────────┤
│ ┃ system: You are a helpful assistant.            │  RequestDetail
│ ┃──────────────────────────────────────────────── │  (colored margins)
│ ┃ user: What is the capital of France?            │
├──────────────────────────────────────────────────┤
│ > Paris.                                          │  ResponseInput
│                                                   │  (TextArea)
└──────────────────────────────────────────────────┘
```

### Message Role Indicators

Colored left-margin bars in the RequestDetail pane:
- **System messages:** dim gray bar
- **User messages:** cyan bar
- **Assistant messages (history):** green bar
- Thin separator line between messages

Full message content displayed — nothing hidden. The visual structure helps the human parse role boundaries quickly.

### Keybindings

| Key | Action |
|-----|--------|
| `↑/↓` or `j/k` | Navigate request list |
| `Enter` | Select request — shows detail + opens response input |
| `Ctrl+Enter` | Submit response |
| `Esc` | Back to request list from detail/input view |
| `q` | Quit TUI |

### Abandoned Request Recovery

On startup, the TUI checks for requests with status `abandoned`. These appear at the top of the request list with a `↻` indicator and a "no client waiting" note (the original HTTP connection is gone). The human can respond for the record (logged but not delivered) or leave them.

### Duration Tracking

- Reading timer starts when request detail is displayed
- Composing timer starts on first keystroke in response input
- Both timers pause after 2 minutes of no keystrokes (idle guard)
- Both timers resume on any keystroke
- Timers sent to server with the response via `POST /_deckard/queue/{id}/respond`
- Diagnostic value: long reading → complex prompt, long composing → difficult question, short both → trivial oracle call

### Polling

TUI polls `GET /_deckard/queue` every 2 seconds for new requests. StatusBar updates pending count. Uses Textual's `@work(thread=True)` pattern for non-blocking HTTP.

---

## Types (`types.py`)

```python
class Status(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass
class QueuedRequest:
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
    host: str = "127.0.0.1"
    port: int = 8421
    db_path: str = "~/.deckard/deckard.db"
    simulate_latency: bool = False
    chunk_delay_ms: int = 2
```

---

## CLI Entry (`__main__.py`)

```
deckard                     — find/start server + open TUI
deckard serve               — headless server, logs to stdout
deckard client              — connect TUI to running server
deckard --port 9000         — override port
deckard --simulate-latency  — 30ms/chunk streaming delay
```

**Server discovery (`deckard` with no subcommand):**
1. Try `GET http://localhost:{port}/health`
2. If responds → server is running, open client
3. If fails → fork server as background process, poll health until ready, open client

---

## Project Structure

```
deckard/
├── deckard/
│   ├── __init__.py
│   ├── __main__.py
│   ├── server.py
│   ├── client.py
│   ├── types.py
│   └── app.tcss
├── tests/
│   ├── test_server.py
│   ├── test_queue.py
│   └── test_types.py
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

### pyproject.toml

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

---

## Future Work (v0.2)

- `$EDITOR` integration: `e` opens `$EDITOR` with prompt as comments, blank section for response (git-commit pattern). Deferred — TextArea handles multi-line editing for v0.1.
- Anthropic `/v1/messages` endpoint (currently 501 stub)
- Adaptive polling (500ms burst after new request, 2s steady state)

## What This Is Not

- Not a proxy. Deckard replaces the inference server, it doesn't sit in front of one.
- Not an Ollama plugin. It's a standalone HTTP server any OpenAI-compatible client can point at.
- Not a testing framework. It's a utility with solid logging that happens to produce useful diagnostic data.
- Not verified software. It's a tool for experiencing verified workflows from the oracle's perspective.
