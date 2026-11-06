"""Shared primitives: the strict model base, deep immutability, hashing and text sanitization."""

import hashlib
import html
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Annotated, Any, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PlainSerializer

# Largest integer a JSON peer such as a browser can represent without losing precision.
MAX_SAFE_INTEGER: Final = 2**53 - 1
MAX_TITLE_LENGTH: Final = 256
MAX_TEXT_LENGTH: Final = 4096
MAX_CODE_LENGTH: Final = 1_000_000
MAX_JSON_DEPTH: Final = 64
MAX_SANITIZE_PASSES: Final = 8

_SCRIPT_BLOCK: Final = re.compile(
    r"<\s*(script|style)\b[^>]*>.*?(?:<\s*/\s*\1\s*>|$)", re.IGNORECASE | re.DOTALL
)
_TAG: Final = re.compile(r"<[a-zA-Z/!?][^>]*(?:>|$)")
_ACTIVE_SCHEME: Final = re.compile(
    r"(?:java|vb|live)script\s*:|data\s*:\s*text/html", re.IGNORECASE
)
_EVENT_HANDLER: Final = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)
_INJECTION_PATTERNS: Final = (_SCRIPT_BLOCK, _TAG, _ACTIVE_SCHEME, _EVENT_HANDLER)
_LINE_BREAKING_CONTROLS: Final = frozenset("\t\n\r")
_INVISIBLE_CATEGORIES: Final = frozenset({"Cc", "Cf"})


class DomainModel(BaseModel):
    """Base for every domain value: immutable, closed to undeclared fields, whitespace-stripped."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class FrozenMap[V](Mapping[str, V]):
    """Read-only, hashable mapping.

    ``frozen=True`` only guards attribute assignment, so a plain dict field could still be mutated
    in place, invalidating its model's validators and hashes after construction.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[str, V]) -> None:
        self._items = dict(items)

    def __getitem__(self, key: str) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._items!r})"


def _require_encodable(text: str) -> None:
    """Reject lone surrogates, which validate as ``str`` but crash when serialized to JSON."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("JSON strings must be valid Unicode") from error


def freeze_json(value: object, *, _depth: int = 0) -> object:
    """Recursively convert JSON-compatible data to immutable containers, rejecting anything else."""
    if _depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON value is nested deeper than {MAX_JSON_DEPTH} levels")
    if isinstance(value, str):
        _require_encodable(value)
        return value
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for key in value:
            _require_encodable(key)
        return FrozenMap({k: freeze_json(v, _depth=_depth + 1) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item, _depth=_depth + 1) for item in value)
    raise ValueError(f"{type(value).__name__} is not JSON-compatible")


def thaw_json(value: object) -> object:
    """Inverse of :func:`freeze_json`, producing plain containers for serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Serialize to one byte-stable form: sorted keys, no insignificant whitespace, no NaN."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    # surrogatepass keeps hashing total, leaving rejection of such text to model validation.
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def require_unique(values: Iterable[str], label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")


def _strip_once(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value))
    text = "".join(
        " " if char in _LINE_BREAKING_CONTROLS else char
        for char in text
        if char in _LINE_BREAKING_CONTROLS
        or unicodedata.category(char) not in _INVISIBLE_CATEGORIES
    )
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def sanitize_text(value: str) -> str:
    """Strip control characters and script injection vectors from display text.

    Repeats until stable, so that removing one vector cannot assemble another
    (``<scr<script>ipt>``) and so that sanitizing is idempotent, which JSON roundtrips rely on.
    Passes are capped because each one peels a single layer, which would make crafted input cost
    quadratic time on the event loop. Genuine text settles within a few passes.
    """
    current = value
    for _ in range(MAX_SANITIZE_PASSES):
        cleaned = _strip_once(current)
        if cleaned == current:
            return current
        current = cleaned
    raise ValueError("text is too deeply obfuscated to sanitize")


def _sanitizer(max_length: int) -> AfterValidator:
    def validate(value: str) -> str:
        cleaned = sanitize_text(value)
        if not cleaned:
            raise ValueError("text is empty after sanitization")
        # Unicode normalization can expand text, and an over-long result would fail on re-parse.
        if len(cleaned) > max_length:
            raise ValueError(f"text exceeds {max_length} characters after sanitization")
        return cleaned

    return AfterValidator(validate)


NonNegativeInt = Annotated[int, Field(strict=True, ge=0, le=MAX_SAFE_INTEGER)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=MAX_SAFE_INTEGER)]
UnixMillis = NonNegativeInt
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
]
SanitizedTitle = Annotated[str, Field(max_length=MAX_TITLE_LENGTH), _sanitizer(MAX_TITLE_LENGTH)]
SanitizedText = Annotated[str, Field(max_length=MAX_TEXT_LENGTH), _sanitizer(MAX_TEXT_LENGTH)]
CodePayload = Annotated[str, Field(min_length=1, max_length=MAX_CODE_LENGTH)]
FrozenJson = Annotated[Any, AfterValidator(freeze_json)]
FrozenJsonObject = Annotated[
    Mapping[str, Any],
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=dict[str, Any]),
]
