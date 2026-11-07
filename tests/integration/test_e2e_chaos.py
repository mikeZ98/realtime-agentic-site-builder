"""Contention, loss, stalls and teardown, each driven through the real application."""

import asyncio
import itertools
import statistics
from typing import TYPE_CHECKING, Any

import pytest
from e2e_support import (
    OWNER,
    SESSIONS,
    VIEWER,
    Frame,
    GatedProvider,
    Gateway,
    Message,
    Socket,
    deadline,
    gateway,
    settings,
)

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.domain.swarm import AgentRole, AgentThoughtStream

if TYPE_CHECKING:
    from realtime_agentic_site_builder.core.hub import SessionListener

ARCHITECT = IDENTITIES[AgentRole.ARCHITECT]
CODER = IDENTITIES[AgentRole.FRONTEND_CODER]
STYLER = IDENTITIES[AgentRole.STYLER]
HUMAN = "user:alice"
HERO = "hero"
DOCUMENT = "frontend:document"


def staged_bundle() -> dict[str, Any]:
    box = {"x": 0, "width": 1920, "height": 400}
    return {
        "project_id": "chaos",
        "frontend": {
            "site_id": "site",
            "title": "Chaos",
            "theme": "dark",
            "components": [
                _component(HERO, {**box, "y": 0}),
                _component("pricing", {**box, "y": 500}),
            ],
        },
        "backend": {"service_name": "api", "database_dialect": "SQLITE"},
    }


def _component(component_id: str, box: dict[str, int]) -> dict[str, Any]:
    from realtime_agentic_site_builder.domain.blueprint import (
        BoundingBox,
        ComponentNode,
        ComponentType,
    )

    node = ComponentNode.create(
        id=component_id,
        component_type=ComponentType.FEATURES,
        title=component_id.title(),
        code="export default function S() { return <section />; }",
        box=BoundingBox(**box),
    )
    return node.model_dump(mode="json")


async def owned_session(harness: Gateway) -> tuple[str, Socket]:
    session_id = await harness.create_session(OWNER, bundle=staged_bundle())
    owner = harness.socket(session_id, OWNER)
    assert (await owner.connect())["kind"] == "sync"
    return session_id, owner


async def acquire(socket: Socket, target_id: str, request_id: str) -> Frame:
    socket.send({"type": "lock.acquire", "target_id": target_id, "request_id": request_id})
    while (frame := await socket.next_of("ack", "error"))["request_id"] != request_id:
        pass
    return frame


async def edit(socket: Socket, target_id: str, path: str, value: str, request_id: str) -> Frame:
    socket.send(
        {
            "type": "delta",
            "request_id": request_id,
            "target_domain": "frontend",
            "target_id": target_id,
            "patches": [{"op": "replace", "path": path, "value": value}],
        }
    )
    while (frame := await socket.next_of("ack", "error"))["request_id"] != request_id:
        pass
    return frame


async def lock_event(
    socket: Socket, kind: str, target_id: str, agent_id: str | None = None
) -> Frame:
    """Wait for a lock notice of a kind on a target, from a given agent or from anyone."""
    while True:
        event: Frame = (await socket.next_of("lock"))["event"]
        lock = event["lock"]
        if (event["kind"], lock["target_id"]) == (kind, target_id) and agent_id in (
            None,
            lock["locked_by_agent_id"],
        ):
            return event


# --- contention -----------------------------------------------------------------------------

CONTENDERS = (ARCHITECT, CODER, STYLER)


@pytest.mark.parametrize(
    "arrival",
    list(itertools.permutations(CONTENDERS)),
    ids=lambda order: ">".join(identity.name.split()[0] for identity in order),
)
@deadline(10)
async def test_simultaneous_contenders_are_served_by_priority_whatever_order_they_arrive_in(
    arrival: tuple[Any, ...],
) -> None:
    async with gateway() as harness:
        session_id, owner = await owned_session(harness)
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        assert (await acquire(owner, HERO, "hold"))["lock"]["state"] == "ACQUIRED"

        queued = await asyncio.gather(
            *(
                manager.acquire(HERO, identity.agent_id, identity.priority, hub.now_ms())
                for identity in arrival
            )
        )
        assert {lock.state.value for lock in queued} == {"WAITING"}

        owner.send({"type": "lock.release", "target_id": HERO})
        served: list[str] = []
        for _ in CONTENDERS:
            holder = (await lock_event(owner, "PROMOTED", HERO))["lock"]["locked_by_agent_id"]
            served.append(holder)
            await manager.release(HERO, holder)
        assert served == [ARCHITECT.agent_id, CODER.agent_id, STYLER.agent_id]
        assert manager.active_locks() == ()
        await owner.drop()


@deadline(10)
async def test_a_deadlock_with_a_higher_ranked_agent_costs_the_person_their_locks() -> None:
    """Architect outranks a person: the person is preempted, and can no longer edit the hero."""
    async with gateway() as harness:
        session_id, owner = await owned_session(harness)
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        assert (await acquire(owner, HERO, "hero"))["lock"]["state"] == "ACQUIRED"
        await manager.acquire(DOCUMENT, ARCHITECT.agent_id, ARCHITECT.priority, hub.now_ms())
        assert (await acquire(owner, DOCUMENT, "document"))["lock"]["state"] == "WAITING"

        won = await manager.acquire(HERO, ARCHITECT.agent_id, ARCHITECT.priority, hub.now_ms())

        assert won.state.value == "ACQUIRED"
        preempted = await lock_event(owner, "PREEMPTED", HERO, HUMAN)
        assert preempted["problem"]["status"] == 409
        assert manager.live_holder(HERO, hub.now_ms()) == ARCHITECT.agent_id
        assert not any(lock.locked_by_agent_id == HUMAN for lock in manager.active_locks())
        refused = await edit(owner, HERO, "/frontend/components/0/title", "Mine", "late")
        assert refused["problem"]["status"] == 409
        await owner.drop()


@deadline(10)
async def test_a_deadlock_with_a_lower_ranked_agent_is_won_by_the_person() -> None:
    """A person outranks the coder: the coder is preempted, and the person's edits go through."""
    async with gateway() as harness:
        session_id, owner = await owned_session(harness)
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        assert (await acquire(owner, HERO, "hero"))["lock"]["state"] == "ACQUIRED"
        await manager.acquire(DOCUMENT, CODER.agent_id, CODER.priority, hub.now_ms())
        assert (await acquire(owner, DOCUMENT, "document"))["lock"]["state"] == "WAITING"

        lost = await manager.acquire(HERO, CODER.agent_id, CODER.priority, hub.now_ms())

        assert lost.state.value == "CONFLICT_RESOLVED"
        await lock_event(owner, "PROMOTED", DOCUMENT, HUMAN)
        title = await edit(owner, DOCUMENT, "/frontend/title", "Renamed", "title")
        hero = await edit(owner, HERO, "/frontend/components/0/title", "Hero!", "component")
        assert (title["kind"], hero["kind"]) == ("ack", "ack")
        bundle = hub.latest_snapshot(session_id).bundle_state
        assert bundle.frontend.title == "Renamed"
        assert bundle.frontend.components[0].title == "Hero!"
        assert [delta.agent_id for delta in hub.timeline(session_id).deltas] == [HUMAN, HUMAN]
        await owner.drop()


@deadline(10)
async def test_locks_are_path_level_so_one_lock_never_covers_another_element() -> None:
    async with gateway() as harness:
        session_id, owner = await owned_session(harness)
        assert (await acquire(owner, HERO, "hero"))["lock"]["state"] == "ACQUIRED"
        # Holding the hero says nothing about the document's own fields, or another component.
        title = await edit(owner, HERO, "/frontend/title", "Sneaky", "title")
        other = await edit(owner, HERO, "/frontend/components/1/title", "Sneaky", "pricing")
        assert title["problem"]["status"] == other["problem"]["status"] == 409
        assert "frontend%3Adocument" in title["problem"]["resource_uri"]
        assert harness.service.hub.timeline(session_id).deltas == ()
        await owner.drop()


# --- loss and catch-up ----------------------------------------------------------------------


async def flood(
    harness: Gateway, session_id: str, count: int, *, per_pause: int = 5, pause_s: float = 0.0
) -> None:
    for index in range(count):
        await harness.service.hub.broadcast_thought(
            session_id,
            AgentThoughtStream(
                agent_id=CODER.agent_id,
                role=AgentRole.FRONTEND_CODER,
                thought=f"Thought {index}",
                tool_name="llm.generate",
                timestamp_ms=1,
            ),
        )
        if index % per_pause == 0:
            await asyncio.sleep(pause_s)


async def thought_seqs(socket: Socket, count: int) -> list[int]:
    return [(await socket.next_of("canvas.thought"))["seq"] for _ in range(count)]


@deadline(15)
async def test_a_dropped_canvas_client_resumes_with_every_thought_it_missed_in_order() -> None:
    total = 600
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=staged_bundle())
        first = harness.socket(session_id, VIEWER, query="view=canvas")
        await first.connect()
        # About 2,500 a second, spread over a quarter of a second, so the drop lands inside it.
        stream = asyncio.create_task(flood(harness, session_id, total, pause_s=0.002))
        seen = await thought_seqs(first, 120)
        await first.drop()  # Mid-stream, with the flood still running.
        assert not stream.done(), "the drop has to happen while thoughts are still arriving"
        await stream

        resumed = harness.socket(
            session_id, VIEWER, query=f"view=canvas&last_seen_thought_seq={seen[-1]}"
        )
        hello = await resumed.connect()
        assert hello["thought_gap"] is False
        assert hello["latest_thought_seq"] == total - 1
        replayed = await thought_seqs(resumed, hello["replayed_thoughts"])

        assert seen + replayed == list(range(total)), "nothing lost, nothing repeated"
        await resumed.drop()


@deadline(15)
async def test_resuming_after_more_than_the_buffer_holds_says_so_rather_than_pretending() -> None:
    async with gateway(settings(thought_backlog_size=100)) as harness:
        session_id = await harness.create_session(OWNER, bundle=staged_bundle())
        first = harness.socket(session_id, VIEWER, query="view=canvas")
        await first.connect()
        await flood(harness, session_id, 10)
        seen = await thought_seqs(first, 10)
        await first.drop()
        await flood(harness, session_id, 400)

        resumed = harness.socket(
            session_id, VIEWER, query=f"view=canvas&last_seen_thought_seq={seen[-1]}"
        )
        hello = await resumed.connect()
        assert (hello["thought_gap"], hello["replayed_thoughts"]) == (True, 100)
        assert await thought_seqs(resumed, 100) == list(range(310, 410))
        await resumed.drop()


# --- slow consumers -------------------------------------------------------------------------


@deadline(20)
async def test_a_stalled_client_is_evicted_while_healthy_ones_keep_their_frame_rate() -> None:
    fps = 60
    stall = asyncio.Event()
    delivered = 0

    async def stalls(_message: Message) -> None:
        nonlocal delivered
        delivered += 1
        if delivered > 3:
            await stall.wait()

    config = settings(broadcast_fps=fps, ws_catchup_deadline_ms=300)
    async with gateway(config) as harness:
        session_id = await harness.create_session(OWNER, bundle=staged_bundle())
        stalled = harness.socket(session_id, "stalled", send_gate=stalls)
        healthy_canvas = [
            harness.socket(session_id, f"viewer-{index}", query="view=canvas") for index in range(2)
        ]
        healthy_stream = harness.socket(session_id, VIEWER)
        await stalled.connect()
        for socket in (*healthy_canvas, healthy_stream):
            await socket.connect()

        loop = asyncio.get_running_loop()
        arrivals: dict[int, list[float]] = {0: [], 1: []}
        thoughts: list[int] = []

        async def watch_canvas(index: int, socket: Socket) -> None:
            while True:
                frame = await socket.receive()
                if frame["kind"] == "canvas":
                    arrivals[index].append(loop.time())

        async def watch_stream() -> None:
            while True:
                frame = await healthy_stream.receive()
                if frame["kind"] == "thought":
                    thoughts.append(frame["seq"])

        watchers = [
            asyncio.create_task(watch_canvas(index, socket))
            for index, socket in enumerate(healthy_canvas)
        ]
        watchers.append(asyncio.create_task(watch_stream()))
        started = loop.time()
        flooding = asyncio.create_task(_steady_flood(harness, session_id, seconds=1.5))

        assert await stalled.closed() == (1008, "Slow Consumer")
        evicted_after = loop.time() - started
        await flooding
        await asyncio.sleep(0.1)
        for watcher in watchers:
            watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)

        assert evicted_after < 1.0, "the deadline, not the flood, decided when it went"
        assert thoughts == list(range(len(thoughts))), "the healthy stream lost nothing"
        assert len(thoughts) > 500
        for times in arrivals.values():
            window = [time for time in times if started <= time <= started + 1.5]
            rate = len(window) / 1.5
            gaps = sorted(later - earlier for earlier, later in itertools.pairwise(window))
            p95_ms = gaps[int(len(gaps) * 0.95)] * 1000
            assert rate > 0.9 * fps, f"{rate:.1f} frames per second"
            assert p95_ms < 2.5 * 1000 / fps, f"p95 interval {p95_ms:.1f} ms"
            assert statistics.median(gaps) * 1000 < 1.2 * 1000 / fps

        stall.set()
        for socket in (*healthy_canvas, healthy_stream):
            await socket.drop()
        await stalled.finished()


async def _steady_flood(harness: Gateway, session_id: str, *, seconds: float) -> None:
    """About 600 thoughts a second, which is far more telemetry than any real run produces."""
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        await flood(harness, session_id, 6, per_pause=10)
        await asyncio.sleep(0.01)


# --- teardown -------------------------------------------------------------------------------


@deadline(15)
async def test_deleting_a_busy_session_leaves_nothing_running_or_referenced() -> None:
    provider = GatedProvider()
    async with gateway(provider=provider) as harness:
        await asyncio.sleep(0)
        baseline = asyncio.all_tasks()
        service, hub = harness.service, harness.service.hub

        session_id = await harness.create_session(OWNER, bundle=staged_bundle(), prompt="Build")
        runtime = service._sessions[session_id]
        run, ticker = runtime.run, runtime.ticker
        assert run is not None
        assert ticker.running
        owner = harness.socket(session_id, OWNER)
        viewer = harness.socket(session_id, VIEWER, query="view=canvas")
        await owner.connect()
        await viewer.connect()
        assert (await acquire(owner, "pricing", "hold"))["lock"]["state"] == "ACQUIRED"
        while provider.calls == 0:
            await asyncio.sleep(0.01)  # The run is now parked inside a model call.
        listeners: set[SessionListener] = set(hub._sessions[session_id].listeners)
        assert len(listeners) == 2
        assert len(asyncio.all_tasks()) > len(baseline)

        deleted = await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert deleted.status_code == 204

        assert await owner.closed() == (1000, "Session closed")
        assert await viewer.closed() == (1000, "Session closed")
        await owner.finished()
        await viewer.finished()
        await asyncio.sleep(0)

        assert run.cancelled()
        assert not ticker.running
        assert all(listener.closed for listener in listeners)
        assert hub._sessions == {}
        assert hub._pumps == {}
        assert hub._bus_handlers == {}
        assert harness.bus._handlers == {}
        assert service._sessions == {}
        assert service._connections == set()
        leaked = asyncio.all_tasks() - baseline
        assert leaked == set(), [task.get_coro() for task in leaked]

        again = await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert again.status_code == 404
        provider.gate.set()
    assert asyncio.all_tasks() == {asyncio.current_task()}, "shutdown left nothing behind either"
