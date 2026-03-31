"""Integration test: server + programmatic client interaction."""
import json
import threading
import time
import urllib.request
import urllib.error
import sqlite3
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


def _post_json(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _get_json(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def test_full_roundtrip(server):
    """Client sends request → appears in queue → respond → client gets answer."""
    base = f"http://127.0.0.1:{server.port}"

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

    # Wait for queue
    for _ in range(50):
        queue = _get_json(f"{base}/_deckard/queue")
        if queue:
            break
        time.sleep(0.1)

    assert len(queue) == 1
    assert queue[0]["messages"][1]["content"] == "What is 2+2?"

    # Respond
    req_id = queue[0]["id"]
    _post_json(f"{base}/_deckard/queue/{req_id}/respond", {
        "response": "4",
        "reading_ms": 100,
        "composing_ms": 200,
    })

    t.join(timeout=10)
    assert result["response"]["choices"][0]["message"]["content"] == "4"
    assert result["response"]["usage"]["completion_tokens"] == 1


def test_abandoned_on_shutdown(server):
    """Requests survive server shutdown as abandoned."""
    base = f"http://127.0.0.1:{server.port}"

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
            pass

    t = threading.Thread(target=client_request)
    t.start()

    for _ in range(50):
        queue = _get_json(f"{base}/_deckard/queue")
        if queue:
            break
        time.sleep(0.1)
    assert len(queue) == 1

    server.shutdown()
    t.join(timeout=5)

    # Check SQLite
    conn = sqlite3.connect(server.config.db_path)
    rows = conn.execute("SELECT status FROM requests").fetchall()
    assert any(r[0] == "abandoned" for r in rows)
    conn.close()
