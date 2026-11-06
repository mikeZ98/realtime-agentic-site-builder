"""The injectable time source, which keeps lease and frame timing deterministic under test."""

import time
from collections.abc import Callable

Clock = Callable[[], int]


def system_now_ms() -> int:
    return time.time_ns() // 1_000_000
