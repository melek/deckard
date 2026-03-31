# CLAUDE.md

## What This Is

Deckard is a human-as-LLM endpoint. It's an OpenAI-compatible HTTP server where a human reads prompts and types responses via a TUI. Any client that speaks the OpenAI chat completions API can point at deckard instead of an inference server.

## Commands

```bash
# Run (starts server + opens TUI)
python3 -m deckard

# Run server only (headless, logs to stdout)
python3 -m deckard start

# Connect TUI to running server
python3 -m deckard client

# Options
python3 -m deckard --port 9000              # override port (default: 8421)
python3 -m deckard --simulate-latency       # 30ms/chunk streaming (default: 2ms)
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Architecture

- `deckard/server.py` -- HTTP server, request queue, SQLite logging, SSE streaming
- `deckard/client.py` -- Textual TUI
- `deckard/types.py` -- Shared types (Status, QueuedRequest, DeckardConfig)
- `deckard/__main__.py` -- CLI entry with server discovery

Data: `~/.deckard/deckard.db` (SQLite)

## Dependencies

- `textual` (TUI framework). Everything else stdlib.
- Python 3.10+

## TUI Keybindings

- Arrow keys: navigate request list
- Enter: select request (shows prompt + opens response editor)
- Ctrl+J or Ctrl+S: submit response
- Esc: back to request list
- q: quit (only when request list has focus)
