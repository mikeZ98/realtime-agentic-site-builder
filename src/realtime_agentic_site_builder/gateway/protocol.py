"""What crosses the socket. Client messages describe intent and can never claim an identity.

Every model forbids undeclared fields, so a message carrying ``agent_id`` or ``user_id`` is rejected
outright rather than having the field silently ignored.
"""

from typing import Annotated, Any, Final, Literal

from pydantic import Field, JsonValue, TypeAdapter

from realtime_agentic_site_builder.domain.base import MAX_TEXT_LENGTH, DomainModel, Identifier
from realtime_agentic_site_builder.domain.swarm import (
    ElementLock,
    PatchOperation,
    SwarmSnapshot,
    ViewportX,
    ViewportY,
)

RequestId = Annotated[str, Field(max_length=64)]
PromptText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]


class _ClientMessage(DomainModel):
    request_id: RequestId | None = None


class PingMessage(_ClientMessage):
    type: Literal["ping"]


class CursorMessage(_ClientMessage):
    type: Literal["cursor"]
    x: ViewportX
    y: ViewportY


class SelectMessage(_ClientMessage):
    type: Literal["select"]
    target_id: Identifier | None = None


class LockAcquireMessage(_ClientMessage):
    type: Literal["lock.acquire"]
    target_id: Identifier


class LockReleaseMessage(_ClientMessage):
    type: Literal["lock.release"]
    target_id: Identifier


class DeltaMessage(_ClientMessage):
    """A patch to commit. The server supplies who it is from and where it falls in sequence."""

    type: Literal["delta"]
    target_domain: Literal["frontend", "backend"]
    target_id: Identifier
    patches: Annotated[tuple[PatchOperation, ...], Field(min_length=1)]


class PromptMessage(_ClientMessage):
    type: Literal["prompt"]
    text: PromptText


ClientMessage = (
    PingMessage
    | CursorMessage
    | SelectMessage
    | LockAcquireMessage
    | LockReleaseMessage
    | DeltaMessage
    | PromptMessage
)
CLIENT_MESSAGE: Final[TypeAdapter[ClientMessage]] = TypeAdapter(
    Annotated[ClientMessage, Field(discriminator="type")]
)


class SyncFrame(DomainModel):
    """The first frame on every connection. Any thoughts being replayed follow it directly."""

    kind: Literal["sync"] = "sync"
    session_id: Identifier
    user_id: Identifier
    agent_id: Identifier
    snapshot: SwarmSnapshot
    latest_thought_seq: int | None
    replayed_thoughts: int
    thought_gap: bool


class CanvasHello(DomainModel):
    """The first frame of a canvas view, which then only receives composited visual frames."""

    kind: Literal["canvas.hello"] = "canvas.hello"
    session_id: Identifier
    user_id: Identifier
    agent_id: Identifier
    is_owner: bool
    latest_thought_seq: int | None
    replayed_thoughts: int
    thought_gap: bool


class PongFrame(DomainModel):
    kind: Literal["pong"] = "pong"
    request_id: RequestId | None = None
    server_time_ms: int


class AckFrame(DomainModel):
    kind: Literal["ack"] = "ack"
    request_id: RequestId | None = None
    type: str
    lock: ElementLock | None = None
    frame_seq: int | None = None


class ErrorFrame(DomainModel):
    kind: Literal["error"] = "error"
    request_id: RequestId | None = None
    problem: dict[str, JsonValue]


def client_request_id(raw: Any) -> str | None:
    """Pull a correlation id out of a message that failed validation, if it has a usable one."""
    candidate = raw.get("request_id") if isinstance(raw, dict) else None
    return candidate if isinstance(candidate, str) and len(candidate) <= 64 else None
