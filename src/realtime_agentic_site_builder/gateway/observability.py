"""Request correlation and the gateway's own metrics.

Every HTTP request and WebSocket handshake gets an id: the caller's ``X-Request-ID`` if it is a
plausible one, or a fresh one otherwise. It is echoed on HTTP responses, attached to every problem
document as ``trace_id``, and set on every log record emitted while the request is handled. Tasks
inherit it, so the log lines of a long-lived socket carry the id of the handshake that opened it.
"""

import contextvars
import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Final

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER: Final = "x-request-id"
_ACCEPTABLE_ID: Final = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rasb_request_id", default=None
)


def current_request_id() -> str | None:
    return _request_id.get()


class RequestIdMiddleware:
    """Outermost middleware, so that even a refusal before routing carries the id."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return
        offered = dict(scope["headers"]).get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        # An untrusted id is replaced rather than echoed, so it cannot forge or inject log lines.
        request_id = offered if _ACCEPTABLE_ID.match(offered) else uuid.uuid4().hex
        token = _request_id.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_id)
        finally:
            _request_id.reset(token)


_installed = False


def install_log_correlation() -> None:
    """Give every log record a ``request_id`` attribute, for formatters to use. Idempotent.

    A record factory rather than a filter, because a filter on a logger never sees its children's
    records, and the handlers belong to whoever runs the process.
    """
    global _installed
    if _installed:
        return
    previous: Callable[..., logging.LogRecord] = logging.getLogRecordFactory()

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = _request_id.get() or "-"
        return record

    logging.setLogRecordFactory(factory)
    _installed = True


class GatewayMetrics:
    """Counters the gateway keeps about itself, rendered in the Prometheus text format."""

    def __init__(self) -> None:
        self.closes: Counter[str] = Counter()
        self.refusals: Counter[str] = Counter()
        self.exports: Counter[str] = Counter()

    def render(self, gauges: Iterable[tuple[str, str, float]]) -> str:
        lines: list[str] = []
        for name, help_text, value in gauges:
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name} {value:g}"]
        for name, label, help_text, counts in (
            ("rasb_ws_closed_total", "reason", "Sockets the server closed.", self.closes),
            ("rasb_refused_total", "limit", "Requests refused by a limit.", self.refusals),
            ("rasb_exports_total", "outcome", "Project exports.", self.exports),
        ):
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
            lines += [f'{name}{{{label}="{key}"}} {count}' for key, count in sorted(counts.items())]
        return "\n".join(lines) + "\n"
