"""A runtime monitor for the safety properties the whole engine rests on.

Each property is checked against a session's recorded history, never against what the code meant
to do, so a defect anywhere upstream (the lock manager, the patch engine, the hub, a replica's
adoption of another worker's commits) shows up as a named, located violation:

- ``log-complete``: every frame after the first has exactly one delta that produced it.
- ``frames-contiguous``: frame numbers rise by exactly one, with no gap and no repeat.
- ``deltas-ordered``: delta sequence numbers strictly increase.
- ``mutual-exclusion``: no frame records two holders of one element.
- ``commits-locked``: every delta's author held its target when the delta was committed.
- ``leases-live``: and that hold had not lapsed at the moment of the commit.
- ``replay-exact``: re-applying the log from its first frame reproduces every recorded hash.

The checks are pure and run in time linear in the log, plus one patch application per delta. A
log with frames and deltas out of step is reported, never raised: the other checks pair what
they can, from the start.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Final

from realtime_agentic_site_builder.core.patch_engine import PatchEngine
from realtime_agentic_site_builder.domain.base import DomainModel
from realtime_agentic_site_builder.domain.errors import SiteBuilderError
from realtime_agentic_site_builder.domain.swarm import LockState, ReplayTimeline

MAX_REPORTED_VIOLATIONS: Final = 5


class InvariantResult(DomainModel):
    name: str
    holds: bool
    checked: int
    violations: tuple[str, ...] = ()


class InvariantReport(DomainModel):
    session_id: str
    first_frame: int
    last_frame: int
    holds: bool
    invariants: tuple[InvariantResult, ...]


def _log_complete(timeline: ReplayTimeline) -> Iterator[str]:
    following, deltas = len(timeline.snapshots) - 1, len(timeline.deltas)
    if deltas != following:
        yield f"{following} frames follow the first, but {deltas} deltas were recorded"


def _frames_contiguous(timeline: ReplayTimeline) -> Iterator[str]:
    for earlier, later in pairwise(timeline.snapshots):
        if later.frame_seq != earlier.frame_seq + 1:
            yield f"frame {later.frame_seq} follows frame {earlier.frame_seq}"


def _deltas_ordered(timeline: ReplayTimeline) -> Iterator[str]:
    for earlier, later in pairwise(timeline.deltas):
        if later.delta_seq <= earlier.delta_seq:
            yield f"delta {later.delta_seq} follows delta {earlier.delta_seq}"


def _mutual_exclusion(timeline: ReplayTimeline) -> Iterator[str]:
    for frame in timeline.snapshots:
        held = Counter(
            lock.target_id for lock in frame.active_locks if lock.state is LockState.ACQUIRED
        )
        for target_id, holders in sorted(held.items()):
            if holders > 1:
                yield f"frame {frame.frame_seq} records {holders} holders of {target_id}"


def _commits(timeline: ReplayTimeline, *, require_live: bool) -> Iterator[str]:
    for delta, frame in zip(timeline.deltas, timeline.snapshots[1:], strict=False):
        hold = next(
            (
                lock
                for lock in frame.active_locks
                if lock.target_id == delta.target_id
                and lock.locked_by_agent_id == delta.agent_id
                and lock.state is LockState.ACQUIRED
            ),
            None,
        )
        if hold is None:
            if not require_live:
                yield (
                    f"frame {frame.frame_seq}: {delta.agent_id} committed to {delta.target_id}"
                    " without holding it"
                )
        elif require_live and hold.is_expired(frame.timestamp_ms):
            yield (
                f"frame {frame.frame_seq}: {delta.agent_id} committed to {delta.target_id}"
                " on a lapsed lease"
            )


def _replay_exact(timeline: ReplayTimeline, engine: PatchEngine) -> Iterator[str]:
    bundle = timeline.snapshots[0].bundle_state
    if bundle.state_hash() != timeline.snapshots[0].bundle_hash:
        yield f"frame {timeline.snapshots[0].frame_seq} does not match its own hash"
        return
    for delta, frame in zip(timeline.deltas, timeline.snapshots[1:], strict=False):
        try:
            bundle, state_hash = engine.apply_patch(bundle, delta)
        except SiteBuilderError as error:
            yield f"frame {frame.frame_seq}: delta {delta.delta_seq} no longer applies: {error}"
            return
        if state_hash != frame.bundle_hash:
            yield f"frame {frame.frame_seq}: replay reaches a different state"
            return


def verify_timeline(timeline: ReplayTimeline, engine: PatchEngine | None = None) -> InvariantReport:
    """Check every invariant over a replay log, and report each with what violated it."""
    patcher = engine or PatchEngine()
    frames, deltas = len(timeline.snapshots), len(timeline.deltas)
    checks: tuple[tuple[str, int, Callable[[], Iterator[str]]], ...] = (
        ("log-complete", frames, lambda: _log_complete(timeline)),
        ("frames-contiguous", frames, lambda: _frames_contiguous(timeline)),
        ("deltas-ordered", deltas, lambda: _deltas_ordered(timeline)),
        ("mutual-exclusion", frames, lambda: _mutual_exclusion(timeline)),
        ("commits-locked", deltas, lambda: _commits(timeline, require_live=False)),
        ("leases-live", deltas, lambda: _commits(timeline, require_live=True)),
        ("replay-exact", deltas, lambda: _replay_exact(timeline, patcher)),
    )
    results = []
    for name, checked, check in checks:
        found = list(check())
        results.append(
            InvariantResult(
                name=name,
                holds=not found,
                checked=checked,
                violations=tuple(found[:MAX_REPORTED_VIOLATIONS]),
            )
        )
    return InvariantReport(
        session_id=timeline.session_id,
        first_frame=timeline.snapshots[0].frame_seq,
        last_frame=timeline.snapshots[-1].frame_seq,
        holds=all(result.holds for result in results),
        invariants=tuple(results),
    )
