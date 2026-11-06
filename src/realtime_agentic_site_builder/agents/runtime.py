"""What every node shares: agent identities, lock scopes, telemetry and delta commits."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self

from pydantic import TypeAdapter, ValidationError

from realtime_agentic_site_builder.agents.provider import (
    LLMProvider,
    LLMRequest,
    LLMResult,
)
from realtime_agentic_site_builder.agents.schemas import AgentOutput
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.core.hub import SwarmSessionHub
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import MAX_TEXT_LENGTH
from realtime_agentic_site_builder.domain.blueprint import ComponentNode
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    ElementLockConflictError,
    LLMProviderError,
    LockAcquisitionTimeoutError,
    element_resource_uri,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    LockState,
    StreamDelta,
)

LOCK_POLL_S: Final = 0.05
# Renew well before a lease ends, so that one slow turn of the event loop cannot lose it.
RENEWALS_PER_LEASE: Final = 3
WIREFRAME_PREFIX: Final = "/* wireframe */"
_DELTA: Final = TypeAdapter(StreamDelta)

# A later role's work builds on an earlier one's, so the earlier role wins a deadlock.
_IDENTITY_ROWS: Final = (
    (AgentRole.ARCHITECT, "agent-architect", "Architect", "#7c3aed", 90),
    (AgentRole.BACKEND_ENGINEER, "agent-backend", "Backend Engineer", "#059669", 70),
    (AgentRole.FRONTEND_CODER, "agent-coder", "Frontend Coder", "#2563eb", 60),
    (AgentRole.STYLER, "agent-styler", "Styler", "#db2777", 40),
    (AgentRole.CRITIC_LINTER, "agent-critic", "Critic", "#d97706", 30),
)


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    role: AgentRole
    agent_id: str
    name: str
    color: str
    priority: int


IDENTITIES: Final[Mapping[AgentRole, AgentIdentity]] = {
    row[0]: AgentIdentity(*row) for row in _IDENTITY_ROWS
}


def wireframe_code(component_id: str) -> str:
    return f'{WIREFRAME_PREFIX} <section data-component="{component_id}" />'


def is_wireframe(component: ComponentNode) -> bool:
    return component.code.startswith(WIREFRAME_PREFIX)


class ElementGoneError(Exception):
    """An element a node was working on was removed by someone else in the meantime."""


class SwarmRuntime:
    def __init__(
        self,
        hub: SwarmSessionHub,
        provider: LLMProvider,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.hub = hub
        self.provider = provider
        self.sleep = sleep


class LockScope:
    """Holds element locks for one unit of work, renewing them until the block ends.

    Generation can outlast the longest lease, so holds are renewed in the background. A hold lost
    anyway, to preemption, is not requested again: the next commit is rejected, which is the
    correct signal to the agent. Leaving the block releases everything, including on failure.
    """

    _heartbeat: asyncio.Task[None]

    def __init__(self, context: "NodeContext") -> None:
        self._context = context
        self._hub = context.runtime.hub
        self._manager = self._hub.lock_manager(context.session_id)
        self._agent_id = context.identity.agent_id
        self._priority = context.identity.priority
        self._held: dict[str, ElementLock] = {}
        self._requested: set[str] = set()

    async def __aenter__(self) -> Self:
        self._heartbeat = asyncio.create_task(self._renew_forever())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._heartbeat.cancel()
        await asyncio.gather(self._heartbeat, return_exceptions=True)
        for target_id in sorted(self._requested):
            await self._manager.release(target_id, self._agent_id)

    def lock_on(self, target_id: str) -> ElementLock | None:
        return self._held.get(target_id)

    async def acquire(self, *target_ids: str) -> None:
        """Take every target, in sorted order so that agents using scopes cannot form a cycle."""
        for target_id in sorted(set(target_ids) - self._held.keys()):
            await self._acquire_one(target_id)

    async def _acquire_one(self, target_id: str) -> None:
        wait_ms = self._hub.settings.agent_lock_wait_ms
        deadline = self._hub.now_ms() + wait_ms
        self._requested.add(target_id)
        announced = False
        while True:
            now_ms = self._hub.now_ms()
            lock = await self._manager.acquire(target_id, self._agent_id, self._priority, now_ms)
            if lock.state is LockState.ACQUIRED:
                self._held[target_id] = lock
                return
            if lock.state is LockState.CONFLICT_RESOLVED:
                raise ElementLockConflictError(
                    competing_agent_ids=(self._agent_id,),
                    resource_uri=element_resource_uri(target_id),
                )
            if now_ms >= deadline:
                raise LockAcquisitionTimeoutError(
                    agent_id=self._agent_id, target_id=target_id, timeout_ms=wait_ms
                )
            if not announced:
                announced = True
                await self._context.emit(
                    AgentAction.BLOCKED_WAITING,
                    f"Waiting for the lock on {target_id}.",
                    tool_name="lock.wait",
                    tool_args={"target_id": target_id},
                )
            await self._context.runtime.sleep(LOCK_POLL_S)

    async def _renew_forever(self) -> None:
        interval_s = self._hub.settings.lock_lease_ms / 1000 / RENEWALS_PER_LEASE
        while True:
            await self._context.runtime.sleep(interval_s)
            now_ms = self._hub.now_ms()
            for target_id in tuple(self._held):
                if self._manager.live_holder(target_id, now_ms) != self._agent_id:
                    del self._held[target_id]
                else:
                    self._held[target_id] = await self._manager.acquire(
                        target_id, self._agent_id, self._priority, now_ms
                    )


class NodeContext:
    """One node run: gathers what it emits, for the graph to merge into state afterwards."""

    def __init__(self, runtime: SwarmRuntime, state: AgentWorkflowState, role: AgentRole) -> None:
        self.runtime = runtime
        self.state = state
        self.identity = IDENTITIES[role]
        self.session_id = state["session_id"]
        self.iteration = state["iteration"]
        self.thoughts: list[AgentThoughtStream] = []
        self.deltas: list[StreamDelta] = []
        self.feedback: list[str] = []

    def bundle(self) -> FullstackProjectBundle:
        return self.runtime.hub.latest_snapshot(self.session_id).bundle_state

    def component_index(self, component_id: str) -> int:
        for index, component in enumerate(self.bundle().frontend.components):
            if component.id == component_id:
                return index
        raise ElementGoneError(f"component {component_id} no longer exists")

    def locks(self) -> LockScope:
        return LockScope(self)

    def _thought(
        self,
        text: str,
        tool_name: str | None,
        tool_args: Mapping[str, Any] | None,
        tokens_used: int,
        latency_ms: float,
    ) -> AgentThoughtStream:
        common: dict[str, Any] = {
            "agent_id": self.identity.agent_id,
            "role": self.identity.role,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tokens_used": tokens_used,
            "execution_latency_ms": latency_ms,
            "timestamp_ms": self.runtime.hub.now_ms(),
        }
        try:
            return AgentThoughtStream(thought=text[:MAX_TEXT_LENGTH], **common)
        except ValidationError:
            # Model text can sanitize to nothing, and that must not abort the agent's work.
            return AgentThoughtStream(thought=f"{self.identity.name} is working.", **common)

    async def emit(
        self,
        action: AgentAction,
        text: str,
        *,
        tool_name: str | None = None,
        tool_args: Mapping[str, Any] | None = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        target_id: str | None = None,
        lock: ElementLock | None = None,
    ) -> None:
        """Publish a thought with any tool badge, then move this agent's cursor to match.

        The thought goes on the lossless channel. The frame repeats it only as the caption for a
        cursor, and frames conflate, so consumers wanting the full stream read ``ThoughtEvent``.
        """
        thought = self._thought(text, tool_name, tool_args, tokens_used, latency_ms)
        self.thoughts.append(thought)
        await self.runtime.hub.broadcast_thought(self.session_id, thought)
        x, y = self._cursor_for(target_id)
        frame = AgentCursorFrame(
            agent_id=self.identity.agent_id,
            name=self.identity.name,
            role=self.identity.role,
            color=self.identity.color,
            x=x,
            y=y,
            current_action=action,
            current_thought=thought,
            active_lock=lock,
        )
        await self.runtime.hub.broadcast_frame(self.session_id, frame)

    def _cursor_for(self, target_id: str | None) -> tuple[int, int]:
        for component in self.bundle().frontend.components:
            if component.id == target_id:
                box = component.box
                return box.x + box.width // 2, box.y + box.height // 2
        return 0, 0

    async def generate[T: AgentOutput](
        self, system: str, prompt: str, schema: type[T], *, target_id: str | None = None
    ) -> LLMResult[T]:
        """Call this role's model tier, with its reasoning and usage on the thought stream."""
        request = LLMRequest(
            role=self.identity.role, system=system, prompt=prompt, output_schema=schema
        )
        client = self.runtime.provider.client_for(self.identity.role)
        result = await client.generate_structured(request)
        await self.emit(
            AgentAction.THINKING,
            result.output.rationale or f"{self.identity.name} finished thinking.",
            tool_name="llm.generate",
            tool_args={"model": result.model, "schema": schema.__name__},
            tokens_used=result.usage.total_tokens,
            latency_ms=result.latency_ms,
            target_id=target_id,
        )
        return result

    async def commit(
        self,
        domain: str,
        target_id: str,
        patches: list[dict[str, Any]],
        *,
        lock: ElementLock | None = None,
    ) -> None:
        """Commit a patch through the hub, which enforces this agent's locks on every pointer."""
        hub = self.runtime.hub
        delta = _DELTA.validate_python(
            {
                "session_id": self.session_id,
                "agent_id": self.identity.agent_id,
                # Reserved and submitted with no await between, so numbers follow commit order.
                "delta_seq": hub.reserve_delta_seq(self.session_id),
                "target_domain": domain,
                "target_id": target_id,
                "patches": patches,
                "emitted_at_ms": hub.now_ms(),
            }
        )
        await hub.apply_delta(self.session_id, delta)
        self.deltas.append(delta)
        await self.emit(
            AgentAction.PATCHING_AST,
            f"Patched {target_id}.",
            tool_name="hub.apply_delta",
            tool_args={"target_id": target_id, "operations": len(patches)},
            target_id=target_id,
            lock=lock,
        )

    @contextlib.contextmanager
    def recoverable(self, target: str) -> Iterator[None]:
        """Turn one element's failure into feedback, so it cannot sink the rest of the run.

        The critic sees the element as unfinished and schedules a retry.
        """
        try:
            yield
        except (
            ASTSyntaxCompilationError,
            ElementLockConflictError,
            LockAcquisitionTimeoutError,
            LLMProviderError,
        ) as error:
            self.fail(target, error.detail)
        except ValidationError as error:
            first = error.errors(include_url=False, include_context=False, include_input=False)[0]
            self.fail(target, str(first["msg"]))
        except ElementGoneError as error:
            self.fail(target, str(error))

    def fail(self, target: str, problem: str) -> None:
        self.feedback.append(
            f"[iteration {self.iteration}] {self.identity.role.value} on {target}: {problem}"
        )

    def feedback_for(self, target_id: str) -> str:
        """Return earlier feedback naming this target, to brief the agent that retries it."""
        lines = [line for line in self.state["critic_feedback"] if target_id in line]
        return "\n".join(lines)

    def update(self) -> dict[str, Any]:
        return {
            "agent_thoughts": self.thoughts,
            "pending_deltas": self.deltas,
            "critic_feedback": self.feedback,
        }
