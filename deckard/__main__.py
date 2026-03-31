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
    cmd = [sys.executable, "-m", "deckard", "serve", "--port", str(args.port)]
    if args.simulate_latency:
        cmd.append("--simulate-latency")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if _check_server(args.host, args.port):
            return True
        time.sleep(0.1)
    return False


def _run_serve(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from deckard.server import DeckardServer
    srv = DeckardServer(
        host=args.host,
        port=args.port,
        chunk_delay_ms=30 if args.simulate_latency else 2,
    )
    srv.serve()


def _run_client(args: argparse.Namespace):
    from deckard.client import DeckardClient
    app = DeckardClient(host=args.host, port=args.port)
    app.run()


def _run_combined(args: argparse.Namespace):
    if not _check_server(args.host, args.port):
        print(f"Starting deckard server on {args.host}:{args.port}...")
        if not _start_server_background(args):
            print("Failed to start server.", file=sys.stderr)
            sys.exit(1)
    _run_client(args)


def main():
    parser = argparse.ArgumentParser(prog="deckard", description="Human-as-LLM endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--simulate-latency", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run headless server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8421)
    serve_parser.add_argument("--simulate-latency", action="store_true")

    client_parser = subparsers.add_parser("client", help="Connect TUI to running server")
    client_parser.add_argument("--host", default="127.0.0.1")
    client_parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "client":
        _run_client(args)
    else:
        _run_combined(args)


if __name__ == "__main__":
    main()
