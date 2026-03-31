"""Deckard CLI entry point."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import urllib.request


def _check_server(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


def _start_server_background(args: argparse.Namespace) -> bool:
    cmd = [sys.executable, "-m", "deckard", "start", "--port", str(args.port)]
    if args.simulate_latency:
        cmd.append("--simulate-latency")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if _check_server(args.host, args.port):
            return True
        time.sleep(0.1)
    # Server didn't come up — clean up the subprocess
    proc.terminate()
    return False


def _run_start(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if _check_server(args.host, args.port):
        print(f"Server already running at {args.host}:{args.port}.", file=sys.stderr)
        sys.exit(1)
    from deckard.server import DeckardServer
    srv = DeckardServer(
        host=args.host,
        port=args.port,
        chunk_delay_ms=30 if args.simulate_latency else 2,
    )
    try:
        srv.serve()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


def _run_client(args: argparse.Namespace):
    from deckard.client import DeckardClient
    app = DeckardClient(host=args.host, port=args.port)
    app.run()


def _run_request(args: argparse.Namespace):
    """Send a test request and print the response."""
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    body = json.dumps({
        "model": args.model,
        "messages": messages,
        "stream": args.stream,
    }).encode()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    print(f"Sending to {args.host}:{args.port}... (waiting for human response)")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if args.stream:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    elif line == "data: [DONE]":
                        print()
            else:
                data = json.loads(resp.read().decode())
                print(data["choices"][0]["message"]["content"])
    except urllib.error.URLError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Is the deckard server running? Try: deckard serve", file=sys.stderr)
        sys.exit(1)


def _run_stop(args: argparse.Namespace):
    """Send shutdown signal to a running server."""
    url = f"http://{args.host}:{args.port}/_deckard/shutdown"
    try:
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read().decode())
        print(f"Server at {args.host}:{args.port} shutting down.")
    except Exception:
        print(f"No server found at {args.host}:{args.port}.", file=sys.stderr)
        sys.exit(1)


def _run_combined(args: argparse.Namespace):
    started_server = False
    if not _check_server(args.host, args.port):
        print(f"Starting deckard server on {args.host}:{args.port}...")
        if not _start_server_background(args):
            print("Failed to start server.", file=sys.stderr)
            sys.exit(1)
        started_server = True

    _run_client(args)

    # After TUI exits
    if started_server:
        print(f"Server still running at {args.host}:{args.port}.")
        print(f"  Stop it:  deckard stop")
        print(f"  Reattach: deckard client")
    else:
        print(f"Server at {args.host}:{args.port} was already running (not started by this session).")


def main():
    parser = argparse.ArgumentParser(prog="deckard", description="Human-as-LLM endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--simulate-latency", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Run headless server")
    start_parser.add_argument("--host", default="127.0.0.1")
    start_parser.add_argument("--port", type=int, default=8421)
    start_parser.add_argument("--simulate-latency", action="store_true")

    client_parser = subparsers.add_parser("client", help="Connect TUI to running server")
    client_parser.add_argument("--host", default="127.0.0.1")
    client_parser.add_argument("--port", type=int, default=8421)

    stop_parser = subparsers.add_parser("stop", help="Shut down a running server")
    stop_parser.add_argument("--host", default="127.0.0.1")
    stop_parser.add_argument("--port", type=int, default=8421)

    req_parser = subparsers.add_parser("request", help="Send a test request (blocks until response)")
    req_parser.add_argument("prompt", help="The user message to send")
    req_parser.add_argument("--host", default="127.0.0.1")
    req_parser.add_argument("--port", type=int, default=8421)
    req_parser.add_argument("--model", default="deckard")
    req_parser.add_argument("--system", default=None, help="Optional system prompt")
    req_parser.add_argument("--stream", action="store_true", help="Request streaming response")

    args = parser.parse_args()

    if args.command == "start":
        _run_start(args)
    elif args.command == "client":
        _run_client(args)
    elif args.command == "stop":
        _run_stop(args)
    elif args.command == "request":
        _run_request(args)
    else:
        _run_combined(args)


if __name__ == "__main__":
    main()
