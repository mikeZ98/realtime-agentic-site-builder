"""Keeps a viewer's copy of the layout exactly equal to the server's, with RFC 6902 alone.

A viewer mirrors the site by applying each commit's patches to its own copy. The patches are not
the whole story, though: after applying them, the engine stamps every component's checksum and
version, and validation normalizes text (NFKC, entity decoding, injection stripping). A browser
cannot reproduce either exactly, so the server works out how the patched copy differs from what it
committed and sends that difference as more patches. Applied after the commit's own, they make the
viewer's copy equal to the committed state, value for value, with no logic duplicated in the page.

The difference is usually a ``replace`` of the version and checksum of each changed component.
"""

import copy
from collections.abc import Iterable, Iterator
from typing import Any

from realtime_agentic_site_builder.core.patch_engine import replay_operations
from realtime_agentic_site_builder.domain.swarm import PatchOperation

FRONTEND_POINTER = "/frontend"


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _replace(pointer: str, value: Any) -> PatchOperation:
    return PatchOperation.model_validate({"op": "replace", "path": pointer, "value": value})


def json_diff(old: Any, new: Any, pointer: str) -> Iterator[PatchOperation]:
    """Patches that turn ``old`` into ``new``, descending wherever the two share a shape.

    Where they do not (different keys, lengths or types) the whole value is replaced, which is
    rare here: stamping and normalization change values, not the structure around them.
    """
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        for key in old:
            yield from json_diff(old[key], new[key], f"{pointer}/{_escape(key)}")
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (before, after) in enumerate(zip(old, new, strict=True)):
            yield from json_diff(before, after, f"{pointer}/{index}")
    elif type(old) is not type(new) or old != new:
        yield _replace(pointer, new)


def corrections(
    before: dict[str, Any], patches: Iterable[PatchOperation], after: dict[str, Any]
) -> tuple[PatchOperation, ...]:
    """What a viewer must apply after ``patches`` to turn ``before`` into ``after``.

    Both layouts are JSON dumps of the frontend, and ``patches`` are what took one to the other.
    """
    document = {"frontend": copy.deepcopy(before)}
    replay_operations(document, patches)
    return tuple(json_diff(document["frontend"], after, FRONTEND_POINTER))
