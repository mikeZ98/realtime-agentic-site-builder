"""One client's socket: an ingress loop for what it asks and an egress loop for what it receives."""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from pydantic import JsonValue, ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from realtime_agentic_site_builder.core.hub import SessionListener, SwarmSessionHub
from realtime_agentic_site_builder.domain.errors import SessionNotFoundError, SiteBuilderError
from realtime_agentic_site_builder.domain.swarm import StreamDelta
from realtime_agentic_site_builder.gateway.auth import Identity
from realtime_agentic_site_builder.gateway.protocol import (
    CLIENT_MESSAGE,
    AckFrame,
    ClientMessage,
    CursorMessage,
    DeltaMessage,
    ErrorFrame,
    LockAcquireMessage,
    LockReleaseMessage,
    PingMessage,
    PongFrame,
    SelectMessage,
    client_request_id,
)

if TYPE_CHECKING:
    from realtime_agentic_site_builder.gateway.service import GatewayService

logger = logging.getLogger(__name__)

WS_NORMAL: Final = 1000
WS_GOING_AWAY: Final = 1001
WS_POLICY_VIOLATION: Final = 1008
WS_MESSAGE_TOO_BIG: Final = 1009
WS_INTERNAL_ERROR: Final = 1011
WS_SESSION_NOT_FOUND: Final = 4404
# A person outranks the swarm's coder but not its architect, and cannot choose differently.
HUMAN_LOCK_PRIORITY: Final = 80
_CLOSE_TIMEOUT_S: Final = 1.0
_MAX_PENDING_REPLIES: Final = 256


@dataclass(frozen=True, slots=True)
class CloseReason:
    code: int
    reason: str


SLOW_CONSUMER: Final = CloseReason(WS_POLICY_VIOLATION, "Slow Consumer")
SESSION_ENDED: Final = CloseReason(WS_NORMAL, "Session closed")
SERVER_DRAINING: Final = CloseReason(WS_GOING_AWAY, "Server shutting down")


class ClientConnection:
    """Runs until the client leaves, misbehaves or stalls, or its session ends.

    Egress never overlaps with itself and only asks for the next event when the last was sent, so a
    slow client's cursors conflate in its listener while state and thoughts wait, undropped. That
    waiting is bounded by ``ws_catchup_deadline_ms``: a client must drain back to empty within the
    window, or it is evicted. A deadline on each send alone would not do, because a client that
    takes just under the limit for every message would otherwise fall behind forever.
    """

    def __init__(
        self,
        service: "GatewayService",
        websocket: WebSocket,
        identity: Identity,
        session_id: str,
        listener: SessionListener,
        initial_frames: tuple[str, ...],
    ) -> None:
        self._service = service
        self._hub: SwarmSessionHub = service.hub
        self._websocket = websocket
        self.identity = identity
        self.session_id = session_id
        self._listener = listener
        self._initial_frames = initial_frames
        self._replies: deque[str] = deque()
        self._end_reason = SESSION_ENDED
        self._x = 0
        self._y = 0
        self._selected: str | None = None
        settings = self._hub.settings
        self._deadline_s = settings.ws_catchup_deadline_ms / 1000
        self._tokens = float(settings.ws_max_messages_per_second)
        self._token_refill_at = service.clock()
        self._caught_up_at = asyncio.get_running_loop().time()

    def drain(self) -> None:
        """End the connection after what is already buffered has been delivered."""
        self._end_reason = SERVER_DRAINING
        self._listener.close()

    async def run(self) -> None:
        egress = asyncio.create_task(self._egress())
        ingress = asyncio.create_task(self._ingress())
        try:
            done, _ = await asyncio.wait({egress, ingress}, return_when=asyncio.FIRST_COMPLETED)
            close = (egress if egress in done else ingress).result()
        except Exception:
            logger.exception("connection failed for %s", self.identity.user_id)
            close = CloseReason(WS_INTERNAL_ERROR, "Internal error")
        finally:
            egress.cancel()
            ingress.cancel()
            # Not gather: if this task is cancelled too, gather raises a new CancelledError, which
            # an anyio scope in the ASGI host would no longer recognize as its own.
            await asyncio.wait({egress, ingress})
            self._listener.close()
        if close is not None:
            await self._close(close)

    async def _close(self, close: CloseReason) -> None:
        # The peer may be why we are closing, so even this is not allowed to hang.
        try:
            await asyncio.wait_for(
                self._websocket.close(close.code, close.reason), _CLOSE_TIMEOUT_S
            )
        except (TimeoutError, RuntimeError, WebSocketDisconnect):
            logger.info("could not deliver close %d to %s", close.code, self.identity.user_id)

    async def _egress(self) -> CloseReason | None:
        try:
            for frame in self._initial_frames:
                await self._send(frame, catching_up=True)
            while True:
                if self._replies:
                    await self._send(self._replies.popleft())
                elif (event := self._listener.poll()) is not None:
                    await self._send(event.model_dump_json())
                elif self._listener.closed:
                    return self._end_reason
                else:
                    await self._listener.wait()
                    # Stamped on waking: time spent with nothing to send is not time spent behind.
                    self._caught_up_at = asyncio.get_running_loop().time()
        except TimeoutError:
            logger.warning("evicting slow consumer %s", self.identity.user_id)
            return SLOW_CONSUMER
        except (WebSocketDisconnect, RuntimeError):
            return None

    async def _send(self, frame: str, *, catching_up: bool = False) -> None:
        now = asyncio.get_running_loop().time()
        if not catching_up and not self._listener.lossless_pending and not self._replies:
            self._caught_up_at = now
        remaining = self._deadline_s - (now - self._caught_up_at)
        await asyncio.wait_for(self._websocket.send_text(frame), remaining)

    def _reply(self, frame: str) -> CloseReason | None:
        # Replies are queued rather than sent here, so a stalled reader cannot wedge the ingress
        # loop. Reading from a client that never reads its replies must still be bounded.
        if len(self._replies) >= _MAX_PENDING_REPLIES:
            return SLOW_CONSUMER
        self._replies.append(frame)
        self._listener.wake()
        return None

    def _take_token(self) -> bool:
        settings = self._hub.settings
        now = self._service.clock()
        elapsed_s = max(now - self._token_refill_at, 0) / 1000
        self._token_refill_at = now
        limit = settings.ws_max_messages_per_second
        self._tokens = min(self._tokens + elapsed_s * limit, float(limit))
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True

    def _receive_timeout(self) -> tuple[float, CloseReason]:
        settings = self._hub.settings
        heartbeat_s = settings.ws_heartbeat_timeout_seconds
        expiry_s = self.identity.expires_at_s - self._hub.now_ms() / 1000
        if expiry_s < heartbeat_s:
            return max(expiry_s, 0.0), CloseReason(WS_POLICY_VIOLATION, "Token expired")
        return heartbeat_s, CloseReason(WS_POLICY_VIOLATION, "Heartbeat timeout")

    async def _ingress(self) -> CloseReason | None:
        settings = self._hub.settings
        while True:
            timeout_s, on_timeout = self._receive_timeout()
            try:
                message = await asyncio.wait_for(self._websocket.receive(), timeout_s)
            except TimeoutError:
                return on_timeout
            if message["type"] == "websocket.disconnect":
                return None
            text = message.get("text")
            if not self._take_token():
                return CloseReason(WS_POLICY_VIOLATION, "Rate limit exceeded")
            if text is None:
                close = self._reply(_invalid(None, "Binary frames are not supported."))
            elif len(text.encode()) > settings.ws_max_message_bytes:
                return CloseReason(WS_MESSAGE_TOO_BIG, "Message too big")
            else:
                close = await self._handle_text(text)
            if close is not None:
                return close

    async def _handle_text(self, text: str) -> CloseReason | None:
        try:
            message = CLIENT_MESSAGE.validate_json(text)
        except ValidationError as error:
            first = error.errors(include_url=False, include_context=False, include_input=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            return self._reply(_invalid(_request_id_of(text), f"{location}: {first['msg']}"))
        try:
            self._hub.touch(self.session_id)
            reply = await self._dispatch(message)
        except SiteBuilderError as error:
            problem = cast("dict[str, JsonValue]", error.to_problem().to_json_dict())
            reply = ErrorFrame(request_id=message.request_id, problem=problem).model_dump_json()
        return None if reply is None else self._reply(reply)

    async def _dispatch(self, message: ClientMessage) -> str | None:
        """Carry out a message, and return its reply if it needs one."""
        request_id = message.request_id
        ack = AckFrame(request_id=request_id, type=message.type)
        if isinstance(message, PingMessage):
            pong = PongFrame(request_id=request_id, server_time_ms=self._hub.now_ms())
            return pong.model_dump_json()
        if isinstance(message, CursorMessage | SelectMessage):
            if isinstance(message, CursorMessage):
                self._x, self._y = message.x, message.y
            else:
                self._selected = message.target_id
            await self._publish_presence()
            # Cursors arrive many times a second, so they are only answered if asked.
            return None if request_id is None else ack.model_dump_json()
        if isinstance(message, LockAcquireMessage):
            lock = await self._hub.lock_manager(self.session_id).acquire(
                message.target_id, self.identity.agent_id, HUMAN_LOCK_PRIORITY, self._hub.now_ms()
            )
            self._service.note_lock(self.session_id, self.identity, message.target_id)
            ack = ack.model_copy(update={"lock": lock})
        elif isinstance(message, LockReleaseMessage):
            await self._hub.lock_manager(self.session_id).release(
                message.target_id, self.identity.agent_id
            )
            self._service.forget_lock(self.session_id, self.identity, message.target_id)
        elif isinstance(message, DeltaMessage):
            ack = ack.model_copy(update={"frame_seq": await self._commit(message)})
        else:
            self._service.start_run(self.session_id, message.text)
        return ack.model_dump_json()

    async def _commit(self, message: DeltaMessage) -> int:
        delta = StreamDelta(
            session_id=self.session_id,
            agent_id=self.identity.agent_id,
            delta_seq=self._hub.reserve_delta_seq(self.session_id),
            target_domain=message.target_domain,
            target_id=message.target_id,
            patches=message.patches,
            emitted_at_ms=self._hub.now_ms(),
        )
        return (await self._hub.apply_delta(self.session_id, delta)).frame_seq

    async def _publish_presence(self, *, active: bool = True) -> None:
        await self._hub.broadcast_presence(
            self.session_id,
            self.identity.user_id,
            x=self._x,
            y=self._y,
            selected_id=self._selected,
            active=active,
        )

    async def announce_departure(self) -> None:
        try:
            await self._publish_presence(active=False)
        except SessionNotFoundError:
            logger.debug("session %s ended before its client left", self.session_id)


def _invalid(request_id: str | None, detail: str) -> str:
    problem: dict[str, JsonValue] = {"status": 422, "title": "Invalid message", "detail": detail}
    return ErrorFrame(request_id=request_id, problem=problem).model_dump_json()


def _request_id_of(text: str) -> str | None:
    try:
        return client_request_id(json.loads(text))
    except ValueError:
        return None
