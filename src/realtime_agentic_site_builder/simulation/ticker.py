"""Runs a simulator in real time and publishes its frames to a session."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Final, Self

from realtime_agentic_site_builder.core.clock import Clock
from realtime_agentic_site_builder.core.hub import SwarmSessionHub
from realtime_agentic_site_builder.domain.errors import SessionNotFoundError
from realtime_agentic_site_builder.simulation.controller import CursorSimulator

logger = logging.getLogger(__name__)

# A longer gap means the loop was starved or suspended. Cursors should resume, not leap.
MAX_STEP_MS: Final = 250


def monotonic_ms() -> int:
    """Measure ticks independently of the wall clock, which can step during a run."""
    return time.monotonic_ns() // 1_000_000


class SimulationTicker:
    """Publishes one session's cursors at ``broadcast_fps``, until stopped or the session ends.

    Frames do not count as session activity. A ticker that runs forever must not stop an abandoned
    session from expiring, and when it does expire, the ticker simply ends.
    """

    def __init__(
        self,
        hub: SwarmSessionHub,
        session_id: str,
        simulator: CursorSimulator | None = None,
        *,
        clock: Clock = monotonic_ms,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._hub = hub
        self._session_id = session_id
        self.simulator = simulator or CursorSimulator()
        self._clock = clock
        self._sleep = sleep
        self._interval_ms = 1000 / hub.settings.broadcast_fps
        self._task: asyncio.Task[None] | None = None
        self.ticks = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Begin ticking. Starting a running ticker changes nothing."""
        if not self.running:
            self._task = asyncio.create_task(self._run())
            self._task.add_done_callback(self._report_failure)

    async def stop(self) -> None:
        """Stop ticking and wait for it to finish. Safe to call at any time, and more than once.

        The wait is shielded: if whoever called this is itself cancelled, the shutdown still
        completes, and it is only the wait that is abandoned.
        """
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    @staticmethod
    def _report_failure(task: asyncio.Task[None]) -> None:
        # Nobody may be awaiting this task, and frozen cursors alone would not explain themselves.
        if not task.cancelled() and task.exception() is not None:
            logger.error("simulation ticker failed", exc_info=task.exception())

    async def _run(self) -> None:
        last_ms = self._clock()
        next_ms = last_ms + self._interval_ms
        try:
            while True:
                await self._sleep(max(next_ms - self._clock(), 0.0) / 1000)
                now_ms = self._clock()
                # Clamped both ways: the clock source is injectable, and may be a wall clock.
                await self._tick(min(max(now_ms - last_ms, 0), MAX_STEP_MS))
                last_ms = now_ms
                next_ms += self._interval_ms
                if next_ms < now_ms:
                    # Fell behind. Skip the missed frames rather than firing them in a burst.
                    next_ms = now_ms + self._interval_ms
        except SessionNotFoundError:
            logger.info("simulation ticker ended with session %s", self._session_id)

    async def _tick(self, delta_ms: int) -> None:
        self.simulator.sync(
            self._hub.latest_snapshot(self._session_id).bundle_state,
            self._hub.lock_manager(self._session_id).active_locks(),
        )
        for frame in self.simulator.step(delta_ms):
            await self._hub.broadcast_frame(self._session_id, frame, touch=False)
        self.ticks += 1
