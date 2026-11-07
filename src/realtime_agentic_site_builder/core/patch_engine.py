"""Atomic RFC 6902 patch execution over project bundles, and deterministic snapshot generation."""

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import (
    CodePayload,
    canonical_json,
    sha256_hex,
    thaw_json,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    SiteBlueprint,
    compute_state_checksum,
)
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    BundleTooLargeError,
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
# Lock targets for what belongs to no single element: site or service fields, and element order.
# The colon keeps them distinct: component ids, model names and endpoint ids cannot contain one.
FRONTEND_DOCUMENT_LOCK: Final = "frontend:document"
BACKEND_DOCUMENT_LOCK: Final = "backend:document"
_DOCUMENT_LOCKS: Final = {"frontend": FRONTEND_DOCUMENT_LOCK, "backend": BACKEND_DOCUMENT_LOCK}
# Each domain's lockable collections, and the member that identifies an element of each.
_ELEMENT_KEYS: Final = {
    "frontend": {"components": "id"},
    "backend": {"models": "name", "endpoints": "id"},
}
_INSERTING_OPS: Final = frozenset({JsonPatchOp.ADD, JsonPatchOp.MOVE, JsonPatchOp.COPY})
_VALUE_WRITING_OPS: Final = frozenset({JsonPatchOp.ADD, JsonPatchOp.REPLACE})

Authorizer = Callable[[frozenset[str]], None]

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


def _element_id(item: Any, key: str) -> set[str]:
    element_id = item.get(key) if isinstance(item, dict) else None
    return {element_id} if isinstance(element_id, str) else set()


def _collection_ids(items: Any, key: str) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {element_id for item in items for element_id in _element_id(item, key)}


def _domain_ids(section: Any, domain: str) -> set[str]:
    members = section if isinstance(section, dict) else {}
    return {
        element_id
        for name, key in _ELEMENT_KEYS[domain].items()
        for element_id in _collection_ids(members.get(name), key)
    }


def _existing_elements(
    document: dict[str, Any], tokens: tuple[str, ...], *, inserting: bool
) -> set[str]:
    """Map a pointer to the lock targets it addresses in the document as it stands.

    Inserting at an array index shifts the element there without changing it, so it does not touch
    that element. A pointer that resolves to nothing touches nothing, and fails when applied.
    """
    domain, *rest = tokens
    document_lock = _DOCUMENT_LOCKS[domain]
    section = document.get(domain)
    if not rest:
        return {document_lock} | _domain_ids(section, domain)
    name, *inner = rest
    key = _ELEMENT_KEYS[domain].get(name)
    if key is None:
        return {document_lock}
    items = section.get(name) if isinstance(section, dict) else None
    if not inner:
        return {document_lock} | _collection_ids(items, key)
    if (
        (inserting and len(inner) == 1)
        or not isinstance(items, list)
        or _ARRAY_INDEX.fullmatch(inner[0]) is None
        or int(inner[0]) >= len(items)
    ):
        return set()
    return _element_id(items[int(inner[0])], key)


def _written_elements(tokens: tuple[str, ...], value: Any) -> set[str]:
    """Return the ids that writing a value at a pointer would bring into existence."""
    domain, *rest = tokens
    if not rest:
        return _domain_ids(value, domain)
    name, *inner = rest
    key = _ELEMENT_KEYS[domain].get(name)
    if key is None:
        return set()
    if not inner:
        return _collection_ids(value, key)
    if len(inner) == 1:
        return _element_id(value, key)
    if inner[1:] == [key] and isinstance(value, str):
        return {value}
    return set()


def elements_touched(document: dict[str, Any], operation: PatchOperation) -> frozenset[str]:
    """Return every lock target one operation reads or writes, including ``test`` and ``from``.

    ``document`` must be the working copy as the previous operations left it, because pointers
    address array positions and those shift during a delta.
    """
    tokens = parse_pointer(operation.path)
    touched = _existing_elements(document, tokens, inserting=operation.op in _INSERTING_OPS)
    if operation.from_ is not None:
        source = parse_pointer(operation.from_)
        touched |= _existing_elements(document, source, inserting=False)
        touched |= _written_elements(tokens, _resolve(document, source))
    elif operation.op in _VALUE_WRITING_OPS:
        touched |= _written_elements(tokens, thaw_json(operation.value))
    return frozenset(touched)


def _diff_collection(
    before: Mapping[str, BaseModel], after: Mapping[str, BaseModel]
) -> tuple[set[str], bool]:
    """Return the ids that were added, removed or edited, and whether survivors changed order."""
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    reordered = [key for key in before if key in after] != [key for key in after if key in before]
    return changed, reordered


def changed_elements(
    before: FullstackProjectBundle, after: FullstackProjectBundle
) -> frozenset[str]:
    """Return the lock targets a transition between two bundles touched.

    This compares outcomes by stable id rather than reading patch paths, because pointers address
    array positions: after ``remove /frontend/components/0``, index 0 names a different component.
    """
    components, components_moved = _diff_collection(
        {node.id: node for node in before.frontend.components},
        {node.id: node for node in after.frontend.components},
    )
    models, models_moved = _diff_collection(
        {model.name: model for model in before.backend.models},
        {model.name: model for model in after.backend.models},
    )
    endpoints, endpoints_moved = _diff_collection(
        {endpoint.id: endpoint for endpoint in before.backend.endpoints},
        {endpoint.id: endpoint for endpoint in after.backend.endpoints},
    )
    changed = components | models | endpoints

    frontend_fields = {"components"}
    if components_moved or before.frontend.model_dump(
        exclude=frontend_fields
    ) != after.frontend.model_dump(exclude=frontend_fields):
        changed.add(FRONTEND_DOCUMENT_LOCK)
    backend_fields = {"models", "endpoints"}
    if (
        models_moved
        or endpoints_moved
        or before.backend.model_dump(exclude=backend_fields)
        != after.backend.model_dump(exclude=backend_fields)
    ):
        changed.add(BACKEND_DOCUMENT_LOCK)
    return frozenset(changed)


@dataclass(frozen=True, slots=True)
class PatchResult:
    bundle: FullstackProjectBundle
    state_hash: str
    size_bytes: int


def bundle_size(bundle: FullstackProjectBundle) -> int:
    """Measure a bundle as its canonical form, which is what hashing and replay serialize."""
    return len(canonical_json(bundle.model_dump(mode="json")).encode())


class PatchEngine:
    """Applies deltas to a private serialized copy of the bundle.

    Bundles are immutable and the copy is only rebuilt into a bundle once every operation and the
    schema have passed, so a rejected delta has nothing to roll back.
    """

    def __init__(
        self,
        *,
        max_patch_size_bytes: int = DEFAULT_MAX_PATCH_SIZE_BYTES,
        max_bundle_bytes: int | None = None,
        clock: Clock = system_now_ms,
    ) -> None:
        self._max_patch_size_bytes = max_patch_size_bytes
        self._max_bundle_bytes = max_bundle_bytes
        self._clock = clock

    def apply_patch(
        self,
        bundle: FullstackProjectBundle,
        delta: StreamDelta,
        *,
        authorize: Authorizer = lambda _elements: None,
    ) -> tuple[FullstackProjectBundle, str]:
        """Return the patched bundle and its state hash. See :meth:`apply`."""
        result = self.apply(bundle, delta, authorize=authorize)
        return result.bundle, result.state_hash

    def apply(
        self,
        bundle: FullstackProjectBundle,
        delta: StreamDelta,
        *,
        authorize: Authorizer = lambda _elements: None,
    ) -> PatchResult:
        """Apply a delta, or raise ``ASTSyntaxCompilationError`` or ``BundleTooLargeError``.

        ``authorize`` is given each operation's lock targets before that operation runs, and rejects
        the delta by raising. It defaults to allowing everything, so a replay of an already
        authorized log needs no lock state.

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
                authorize(elements_touched(document, operation))
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
        # One serialization yields both the hash and the size, so the limit costs nothing extra.
        canonical = canonical_json(patched.model_dump(mode="json"))
        size_bytes = len(canonical.encode())
        if self._max_bundle_bytes is not None and size_bytes > self._max_bundle_bytes:
            raise BundleTooLargeError(size_bytes=size_bytes, limit_bytes=self._max_bundle_bytes)
        return PatchResult(patched, sha256_hex(canonical), size_bytes)

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
