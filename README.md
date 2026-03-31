# Deckard

Human-as-LLM endpoint. An OpenAI-compatible API server where a human reads prompts and types responses.

Point any client at `localhost:8421` instead of an inference server. The client can't tell the difference.

## Install

```bash
uv tool install /path/to/deckard
```

Or for development:

```bash
uv pip install -e /path/to/deckard
```

## Usage

```bash
# Start server + TUI
deckard

# Send a test request from another terminal
deckard request "What is the capital of France?"

# Start headless server only
deckard start

# Connect TUI to a running server
deckard client
```

## Options

```
--port N              Server port (default: 8421)
--simulate-latency    30ms/chunk streaming delay (default: 2ms)
```

## What It's For

- Test any system that makes LLM API calls by putting a human in the loop
- See exactly what a framework sends in its prompts (system messages, tool results, conversation history)
- Produce diagnostic data: response times, reading vs composing duration, token estimates

## Data

SQLite log at `~/.deckard/deckard.db`. Every request and response is recorded with timestamps, duration tracking, and token estimates.

## Dependencies

Python 3.10+. One dependency: [Textual](https://textual.textualize.io/) (TUI framework).
