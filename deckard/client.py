"""Deckard TUI client — human-as-LLM response interface.

Connects to the deckard server over HTTP (no internal server imports).
Displays pending requests, shows message detail, and submits human responses.

Usage:
    python -m deckard.client                          # default
    python -m deckard.client --host 127.0.0.1 --port 8421
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime

logger = logging.getLogger(__name__)

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, ListView, ListItem, Label, RichLog, Static, TextArea


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDLE_THRESHOLD_S = 120  # seconds of idle before gap is excluded


def _truncate(text: str, length: int) -> str:
    """Truncate text to length, adding ellipsis if needed."""
    if len(text) <= length:
        return text
    return text[: length - 1] + "\u2026"


def _format_time(iso: str) -> str:
    """Extract HH:MM from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return "??:??"


def _last_user_message(messages: list[dict]) -> str:
    """Return the content of the last user message, or fallback."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    # No user message — use last message regardless of role
    if messages:
        return messages[-1].get("content", "")
    return "(empty)"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class StatusBar(Static):
    """Top bar: connection dot, pending count, total count."""

    def __init__(self, server_addr: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.connected = False
        self.pending = 0
        self.total = 0
        self.server_addr = server_addr
        self._render_content()

    def _render_content(self) -> None:
        if self.connected:
            dot = "[green]\u25cf[/green] connected"
        else:
            dot = "[red]\u25cf[/red] disconnected"
        self.update(
            f" deckard  {dot}  \u2500  {self.pending} pending  \u2500  {self.total} total  \u2500  {self.server_addr}"
        )

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        self._render_content()

    def set_counts(self, pending: int, total: int) -> None:
        self.pending = pending
        self.total = total
        self._render_content()


class RequestDetail(RichLog):
    """Shows full message array with colored left-margin bars."""
    pass


class ResponseInput(TextArea):
    """Multi-line text editor for composing a response."""
    pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class DeckardClient(App):
    """Deckard TUI — human-as-LLM response interface."""

    CSS_PATH = "app.tcss"
    TITLE = "deckard"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("escape", "back", "Back", show=True),
        # Ctrl+Enter: some terminals send ctrl+j, others ctrl+enter.
        # Bind both plus ctrl+s as a reliable fallback.
        Binding("ctrl+j", "submit", "Submit (Ctrl+Enter)", show=True),
        Binding("ctrl+s", "submit", "Submit", show=False),
    ]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8421,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

        # Queue state
        self._queue_items: list[dict] = []
        self._selected_id: str | None = None

        # Duration tracking
        self._reading_start: float | None = None
        self._composing_start: float | None = None
        self._composing_accumulated_ms: int = 0
        self._last_keystroke: float | None = None

    def compose(self) -> ComposeResult:
        yield StatusBar(server_addr=f"{self.host}:{self.port}", id="status-bar")
        yield ListView(id="request-list")
        yield RequestDetail(
            highlight=True, markup=True, wrap=True, id="request-detail"
        )
        yield ResponseInput(id="response-input")
        yield Footer()

    def on_mount(self) -> None:
        # Hide detail and input on mount
        self.query_one("#request-detail").display = False
        self.query_one("#response-input").display = False
        self._poll_queue()
        self.set_interval(2, self._poll_queue)
        self.query_one("#request-list", ListView).focus()

    # ---- polling ----

    @work(exclusive=True, thread=True, group="poll")
    def _poll_queue(self) -> None:
        """Poll server for pending requests."""
        status_bar = self.query_one("#status-bar", StatusBar)
        try:
            req = urllib.request.Request(f"{self.base_url}/_deckard/queue")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            self.call_from_thread(self._update_queue, data)
            self.call_from_thread(status_bar.set_connected, True)
        except (urllib.error.URLError, OSError, ConnectionError):
            self.call_from_thread(status_bar.set_connected, False)
        except Exception as e:
            logger.debug("poll error: %s", e)
            self.call_from_thread(status_bar.set_connected, False)

    def _update_queue(self, items: list[dict]) -> None:
        """Update the request list from polled data."""
        status_bar = self.query_one("#status-bar", StatusBar)
        list_view = self.query_one("#request-list", ListView)

        pending = sum(1 for i in items if i.get("status") == "pending")
        status_bar.set_counts(pending, len(items))

        # Skip rebuild if queue hasn't changed (but always build on first poll)
        old_ids = [i.get("id") for i in self._queue_items]
        new_ids = [i.get("id") for i in items]
        if old_ids == new_ids and list_view.children:
            self._queue_items = items
            return

        self._queue_items = items

        # Preserve selection
        old_index = list_view.index

        list_view.clear()
        if not items:
            list_view.append(
                ListItem(Label("[dim]Waiting for requests... Point a client at this server.[/dim]", markup=True))
            )
        else:
            for idx, item in enumerate(items):
                ts = _format_time(item.get("created_at", ""))
                model = _truncate(item.get("model", ""), 8)
                last_msg = _last_user_message(item.get("messages", []))
                preview = _truncate(last_msg, 40)
                abandoned = item.get("abandoned", False)

                prefix = "\u21bb " if abandoned else ""
                note = "  [dim](no client waiting)[/dim]" if abandoned else ""
                label_text = (
                    f"[dim][{idx + 1}][/dim] {ts}  {model:8s}  "
                    f'{prefix}"{preview}"{note}'
                )
                list_view.append(ListItem(Label(label_text, markup=True)))

        # Restore selection if valid
        if old_index is not None and 0 <= old_index < len(items):
            list_view.index = old_index

    # ---- request selection ----

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Show detail and input when a request is selected."""
        # Don't switch requests while composing a response (CA-4.2 guard)
        if self._selected_id is not None:
            return

        idx = event.list_view.index
        if idx is None or idx >= len(self._queue_items):
            return

        item = self._queue_items[idx]
        self._selected_id = item["id"]

        # Show detail
        detail = self.query_one("#request-detail", RequestDetail)
        detail.display = True
        detail.clear()

        # Abandoned banner
        if item.get("abandoned", False):
            detail.write("[yellow]\u21bb recovered \u2014 no client waiting[/yellow]")
            detail.write("")

        # Render messages with colored margin bars
        messages = item.get("messages", [])
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                bar = "[dim]\u2503[/dim]"
                role_label = "[dim]system:[/dim]"
            elif role == "user":
                bar = "[cyan]\u2503[/cyan]"
                role_label = "[cyan]user:[/cyan]"
            elif role == "assistant":
                bar = "[green]\u2503[/green]"
                role_label = "[green]assistant:[/green]"
            else:
                bar = "[dim]\u2503[/dim]"
                role_label = f"[dim]{role}:[/dim]"

            detail.write(f" {bar} {role_label} {content}")

            # Separator between messages (but not after the last)
            if i < len(messages) - 1:
                detail.write(f" {bar}{'─' * 50}")

        # Show input
        text_area = self.query_one("#response-input", ResponseInput)
        text_area.display = True
        text_area.clear()
        text_area.focus()

        # Start duration tracking
        self._reading_start = time.monotonic()
        self._composing_start = None
        self._composing_accumulated_ms = 0
        self._last_keystroke = None

    # ---- response composition ----

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Track composing duration on keystrokes."""
        now = time.monotonic()

        if self._composing_start is None:
            # First keystroke
            self._composing_start = now
            self._last_keystroke = now
            return

        # Idle guard: if gap since last keystroke > threshold, don't count the gap
        if self._last_keystroke is not None:
            gap = now - self._last_keystroke
            if gap <= _IDLE_THRESHOLD_S:
                self._composing_accumulated_ms += int(gap * 1000)
            # else: idle gap, skip it

        self._last_keystroke = now

    # ---- submit ----

    def action_submit(self) -> None:
        """Submit the response (Ctrl+S)."""
        if self._selected_id is None:
            return

        text_area = self.query_one("#response-input", ResponseInput)
        response_text = text_area.text.strip()
        if not response_text:
            return

        # Calculate reading_ms
        reading_ms = 0
        if self._reading_start is not None:
            reading_end = self._composing_start or time.monotonic()
            reading_ms = int((reading_end - self._reading_start) * 1000)

        # Composing_ms: accumulated active time
        composing_ms = self._composing_accumulated_ms

        self._send_response(self._selected_id, response_text, reading_ms, composing_ms)

    @work(exclusive=True, thread=True, group="submit")
    def _send_response(
        self, req_id: str, response: str, reading_ms: int, composing_ms: int
    ) -> None:
        """POST the response to the server."""
        try:
            body = json.dumps({
                "response": response,
                "reading_ms": reading_ms,
                "composing_ms": composing_ms,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/_deckard/queue/{req_id}/respond",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read().decode())

            # Success — hide detail/input and go back to list
            self.call_from_thread(self._after_submit_success)

        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode())
                err_msg = err_body.get("error", f"HTTP {e.code}")
            except (json.JSONDecodeError, TypeError):
                err_msg = f"HTTP {e.code}"
            detail = self.query_one("#request-detail", RequestDetail)
            self.call_from_thread(detail.write, f"[red]Error: {err_msg}[/red]")

        except Exception as e:
            detail = self.query_one("#request-detail", RequestDetail)
            self.call_from_thread(detail.write, f"[red]Error: {e}[/red]")

    def _after_submit_success(self) -> None:
        """Clean up UI after successful submit."""
        self._selected_id = None
        self._reading_start = None
        self._composing_start = None
        self._composing_accumulated_ms = 0
        self._last_keystroke = None

        self.query_one("#request-detail").display = False
        self.query_one("#response-input").display = False
        self.query_one("#request-list", ListView).focus()

        # Trigger immediate re-poll
        self._poll_queue()

    # ---- navigation ----

    def action_back(self) -> None:
        """Escape: hide detail/input, return to list."""
        self._selected_id = None
        self._reading_start = None
        self._composing_start = None
        self._composing_accumulated_ms = 0
        self._last_keystroke = None

        self.query_one("#request-detail").display = False
        self.query_one("#response-input").display = False
        self.query_one("#request-list", ListView).focus()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="deckard-client", description="Deckard TUI client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8421, help="Server port (default: 8421)")
    args = parser.parse_args(argv)

    app = DeckardClient(host=args.host, port=args.port)
    app.run()


if __name__ == "__main__":
    main()
