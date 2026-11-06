"""Frontend AST: the component tree of a generated site."""

from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from realtime_agentic_site_builder.domain.base import (
    CodePayload,
    DomainModel,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    SanitizedTitle,
    Sha256Hex,
    canonical_json,
    require_unique,
    sha256_hex,
)

# A canonical lowercase UUIDv7 also satisfies this pattern.
ComponentId = Annotated[str, Field(max_length=128, pattern=r"^[a-z0-9-]+$")]
Theme = Literal["dark", "light", "cyberpunk", "minimal", "enterprise"]


class ComponentType(StrEnum):
    NAVBAR = "NAVBAR"
    HERO = "HERO"
    FEATURES = "FEATURES"
    PRICING = "PRICING"
    TESTIMONIALS = "TESTIMONIALS"
    CONTACT_FORM = "CONTACT_FORM"
    FOOTER = "FOOTER"
    CUSTOM_DYNAMIC = "CUSTOM_DYNAMIC"


class BoundingBox(DomainModel):
    x: NonNegativeInt
    y: NonNegativeInt
    width: NonNegativeInt
    height: NonNegativeInt

    @model_validator(mode="after")
    def _require_non_zero_area(self) -> Self:
        if self.width == 0 or self.height == 0:
            raise ValueError("bounding box must have a non-zero area")
        return self

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def overlaps(self, other: "BoundingBox") -> bool:
        """Report a positive-area intersection. Boxes that share only an edge do not overlap."""
        return (
            self.x < other.right
            and other.x < self.right
            and self.y < other.bottom
            and other.y < self.bottom
        )


def normalize_code(code: str) -> str:
    """Remove line-ending and trailing-whitespace differences that do not change the code."""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def compute_state_checksum(code: str, box: BoundingBox) -> str:
    """Hash normalized code and coordinates, encoded so the two cannot run into each other."""
    return sha256_hex(
        canonical_json({"box": [box.x, box.y, box.width, box.height], "code": normalize_code(code)})
    )


class ComponentNode(DomainModel):
    id: ComponentId
    component_type: ComponentType
    title: SanitizedTitle
    code: CodePayload
    box: BoundingBox
    ast_version: PositiveInt
    connected_endpoint_id: Identifier | None = None
    state_checksum: Sha256Hex

    @model_validator(mode="after")
    def _detect_drift(self) -> Self:
        expected = compute_state_checksum(self.code, self.box)
        if self.state_checksum != expected:
            raise ValueError(f"state_checksum drift on component {self.id}: expected {expected}")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        component_type: ComponentType,
        title: str,
        code: str,
        box: BoundingBox,
        ast_version: int = 1,
        connected_endpoint_id: str | None = None,
    ) -> Self:
        """Build a node, deriving its checksum from the given code and box."""
        return cls(
            id=id,
            component_type=component_type,
            title=title,
            code=code,
            box=box,
            ast_version=ast_version,
            connected_endpoint_id=connected_endpoint_id,
            state_checksum=compute_state_checksum(code, box),
        )


def _require_spatial_order(components: Sequence[ComponentNode]) -> None:
    """Require reading order (top to bottom, then left to right) and no overlapping boxes.

    Vertical stacks and flex or grid rows both satisfy this. Components must already be sorted by
    ``y``, which lets the overlap scan stop at the first component below the current one.
    """
    for previous, current in pairwise(components):
        if (current.box.y, current.box.x) < (previous.box.y, previous.box.x):
            raise ValueError(f"component {current.id} is out of order after {previous.id}")
    for index, upper in enumerate(components):
        for lower in components[index + 1 :]:
            if lower.box.y >= upper.box.bottom:
                break
            if upper.box.overlaps(lower.box):
                raise ValueError(f"components {upper.id} and {lower.id} overlap")


class SiteBlueprint(DomainModel):
    site_id: Identifier
    title: SanitizedTitle
    theme: Theme
    components: tuple[ComponentNode, ...] = ()

    @model_validator(mode="after")
    def _validate_components(self) -> Self:
        require_unique((component.id for component in self.components), "component ids")
        _require_spatial_order(self.components)
        return self
