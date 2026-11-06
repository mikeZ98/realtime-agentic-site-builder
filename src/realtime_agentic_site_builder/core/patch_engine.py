"""Atomic RFC 6902 patch execution over project bundles, and deterministic snapshot generation."""

import copy
import re
from typing import Any, Final

from pydantic import TypeAdapter, ValidationError

from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import CodePayload, canonical_json, thaw_json
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    SiteBlueprint,
    compute_state_checksum,
)
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    EndpointBindingNotFoundError,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentCursorFrame,
    ElementLock,
    JsonPatchOp,
    PatchOperation,
    StreamDelta,
    SwarmSnapshot,
)

DEFAULT_MAX_PATCH_SIZE_BYTES: Final = 65536

# RFC 6901 forbids leading zeros. The digit cap keeps int() clear of its conversion limit, and no
# real array comes near that length.
_ARRAY_INDEX: Final = re.compile(r"0|[1-9][0-9]{0,15}")
_END_OF_ARRAY: Final = "-"
_ENGINE_OWNED_FIELDS: Final = frozenset({"ast_version", "state_checksum"})
_CODE: Final = TypeAdapter(CodePayload)


class _PatchFailure(Exception):
    """An operation cannot apply. Translated to a domain error with the operation's position."""


def parse_pointer(pointer: str) -> tuple[str, ...]:
    """Split an RFC 6901 pointer. ``~1`` decodes first, so ``~01`` means a literal ``~1``."""
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:])


def _array_index(array: list[Any], token: str, *, for_insert: bool = False) -> int:
    if for_insert and token == _END_OF_ARRAY:
        return len(array)
    if _ARRAY_INDEX.fullmatch(token) is None:
        raise _PatchFailure(f"{token!r} is not a valid array index")
    index = int(token)
    if index > len(array) or (index == len(array) and not for_insert):
        raise _PatchFailure(f"array index {index} is out of bounds for length {len(array)}")
    return index


def _resolve(document: Any, tokens: tuple[str, ...]) -> Any:
    node = document
    for token in tokens:
        if isinstance(node, dict):
            if token not in node:
                raise _PatchFailure(f"member {token!r} does not exist")
            node = node[token]
        elif isinstance(node, list):
            node = node[_array_index(node, token)]
        else:
            raise _PatchFailure(f"cannot traverse into a scalar at {token!r}")
    return node


def _insert(document: Any, tokens: tuple[str, ...], value: Any) -> None:
    parent = _resolve(document, tokens[:-1])
    token = tokens[-1]
    if isinstance(parent, dict):
        parent[token] = value
    elif isinstance(parent, list):
        parent.insert(_array_index(parent, token, for_insert=True), value)
    else:
        raise _PatchFailure(f"cannot add {token!r} to a scalar")


def _delete(document: Any, tokens: tuple[str, ...]) -> Any:
    removed = _resolve(document, tokens)
    parent = _resolve(document, tokens[:-1])
    if isinstance(parent, dict):
        del parent[tokens[-1]]
    else:
        del parent[_array_index(parent, tokens[-1])]
    return removed


def _json_equal(left: Any, right: Any) -> bool:
    """Compare as RFC 6902 ``test`` requires. Python's ``True == 1`` is not JSON equality."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(_json_equal, left, right))
    return bool(left == right)


def _encoded_size(value: Any) -> int:
    return len(canonical_json(value).encode())


def _require_in_domain(operation: PatchOperation, domain: str) -> None:
    root = f"/{domain}"
    for pointer in (operation.path, operation.from_):
        if pointer is not None and pointer != root and not pointer.startswith(f"{root}/"):
            raise _PatchFailure(f"{pointer!r} is outside the delta's {domain} domain")


def _apply_operation(document: Any, operation: PatchOperation, budget: int) -> int:
    """Apply one operation in place and return the size budget left for the rest of the delta."""
    tokens = parse_pointer(operation.path)
    source = parse_pointer(operation.from_ or "")
    if operation.op is JsonPatchOp.ADD:
        _insert(document, tokens, thaw_json(operation.value))
    elif operation.op is JsonPatchOp.REMOVE:
        _delete(document, tokens)
    elif operation.op is JsonPatchOp.REPLACE:
        _delete(document, tokens)
        _insert(document, tokens, thaw_json(operation.value))
    elif operation.op is JsonPatchOp.MOVE:
        _insert(document, tokens, _delete(document, source))
    elif operation.op is JsonPatchOp.COPY:
        value = _resolve(document, source)
        # Each copy can double the document, so a small patch could otherwise exhaust memory.
        budget -= _encoded_size(value)
        if budget < 0:
            raise _PatchFailure("copied data exceeds the patch size limit")
        _insert(document, tokens, copy.deepcopy(value))
    elif not _json_equal(_resolve(document, tokens), thaw_json(operation.value)):
        raise _PatchFailure(f"test failed at {operation.path!r}")
    return budget


def _stamp_component(raw: dict[str, Any], previous: ComponentNode | None) -> None:
    """Derive the engine-owned fields of a patched component.

    A writer that left ``state_checksum`` alone gets it recomputed. One that supplied a new value
    has it verified, because a mismatch means the writer and the server disagree about the state.
    ``ast_version`` is always derived, so a patch cannot move it backwards. A removed component's
    history goes with it, so an id that is added again restarts at version 1.
    """
    checksum = compute_state_checksum(
        _CODE.validate_python(raw.get("code")), BoundingBox.model_validate(raw.get("box"))
    )
    supplied = raw.get("state_checksum")
    untouched = previous is not None and supplied == previous.state_checksum
    if supplied is not None and not untouched and supplied != checksum:
        raise _PatchFailure(f"state_checksum drift on component {raw.get('id')!r}")
    raw["state_checksum"] = checksum

    version = 1
    if previous is not None:
        version = previous.ast_version
        before = previous.model_dump(mode="json")
        if any(raw.get(k) != v for k, v in before.items() if k not in _ENGINE_OWNED_FIELDS):
            version += 1
    raw["ast_version"] = version


def _stamp_components(document: dict[str, Any], prior: SiteBlueprint) -> None:
    """Stamp every well-formed component, leaving malformed shapes for schema validation."""
    frontend = document.get("frontend")
    components = frontend.get("components") if isinstance(frontend, dict) else None
    if not isinstance(components, list):
        return
    known = {component.id: component for component in prior.components}
    for raw in components:
        if isinstance(raw, dict):
            component_id = raw.get("id")
            _stamp_component(
                raw, known.get(component_id) if isinstance(component_id, str) else None
            )


def _describe(error: ValidationError) -> str:
    first = error.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"patched bundle violates its schema at {location or 'root'}: {first['msg']}"


class PatchEngine:
    """Applies deltas to a private serialized copy of the bundle.

    Bundles are immutable and the copy is only rebuilt into a bundle once every operation and the
    schema have passed, so a rejected delta has nothing to roll back.
    """

    def __init__(
        self,
        *,
        max_patch_size_bytes: int = DEFAULT_MAX_PATCH_SIZE_BYTES,
        clock: Clock = system_now_ms,
    ) -> None:
        self._max_patch_size_bytes = max_patch_size_bytes
        self._clock = clock

    def apply_patch(
        self, bundle: FullstackProjectBundle, delta: StreamDelta
    ) -> tuple[FullstackProjectBundle, str]:
        """Return the patched bundle and its state hash, or raise ``ASTSyntaxCompilationError``.

        ``line_number`` on the error is the 1-based position of the failing operation. Failures
        found after all operations ran report the last position and no ``syntax_token``.
        """
        patch = [operation.model_dump() for operation in delta.patches]
        encoded = canonical_json(patch)
        budget = self._max_patch_size_bytes - len(encoded.encode())
        if budget < 0:
            raise ASTSyntaxCompilationError(
                f"patch exceeds the {self._max_patch_size_bytes} byte limit",
                line_number=1,
                syntax_token=None,
                code_snippet=encoded,
            )

        document = bundle.model_dump(mode="json")
        for position, (operation, raw) in enumerate(zip(delta.patches, patch, strict=True), 1):
            try:
                _require_in_domain(operation, delta.target_domain)
                budget = _apply_operation(document, operation, budget)
            # A chain of nested adds can build a document too deep for recursive traversal.
            except (_PatchFailure, RecursionError) as error:
                raise ASTSyntaxCompilationError(
                    str(error),
                    line_number=position,
                    syntax_token=operation.path,
                    code_snippet=canonical_json(raw),
                ) from error

        try:
            _stamp_components(document, bundle.frontend)
            patched = FullstackProjectBundle.model_validate(document)
        except (ValidationError, _PatchFailure, EndpointBindingNotFoundError) as error:
            reason = _describe(error) if isinstance(error, ValidationError) else str(error)
            raise ASTSyntaxCompilationError(
                reason, line_number=len(delta.patches), syntax_token=None, code_snippet=encoded
            ) from error
        return patched, patched.state_hash()

    def generate_snapshot(
        self,
        bundle: FullstackProjectBundle,
        frame_seq: int,
        step_name: str,
        agents: tuple[AgentCursorFrame, ...],
        locks: tuple[ElementLock, ...],
        *,
        timestamp_ms: int | None = None,
    ) -> SwarmSnapshot:
        """Record a frame. Equal arguments and an explicit timestamp yield an equal snapshot."""
        return SwarmSnapshot.capture(
            frame_seq=frame_seq,
            step_name=step_name,
            timestamp_ms=self._clock() if timestamp_ms is None else timestamp_ms,
            bundle_state=bundle,
            active_agents=agents,
            active_locks=locks,
        )
