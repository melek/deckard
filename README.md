# Deckard

Human-as-LLM endpoint. An OpenAI-compatible API server where a human reads prompts and types responses.

Point any client at `localhost:8421` instead of an inference server. The client can't tell the difference.

## Install

```bash
uv tool install git+https://github.com/melek/deckard
```

Or from a local clone:

```bash
uv tool install -e /path/to/deckard
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

## Pointing Other Tools at Deckard

Start deckard, then set the base URL in your tool of choice:

```bash
# Python openai library
export OPENAI_BASE_URL=http://localhost:8421/v1
python -c "from openai import OpenAI; print(OpenAI().chat.completions.create(model='deckard', messages=[{'role':'user','content':'hello'}]).choices[0].message.content)"

# Claude Code
claude --provider openai-compatible --endpoint http://localhost:8421/v1

# curl
curl http://localhost:8421/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deckard","messages":[{"role":"user","content":"hello"}]}'

# Any OpenAI-compatible client
# Just set the base URL to http://localhost:8421/v1
```

Each request appears in the deckard TUI. You read the full prompt (system messages, conversation history, everything) and type your response.

## What It's For

- Test any system that makes LLM API calls by putting a human in the loop
- See exactly what a framework sends in its prompts (system messages, tool results, conversation history)
- Produce diagnostic data: response times, reading vs composing duration, token estimates

## Data

SQLite log at `~/.deckard/deckard.db`. Every request and response is recorded with timestamps, duration tracking, and token estimates.

## Dependencies

Python 3.10+. One dependency: [Textual](https://textual.textualize.io/) (TUI framework).
