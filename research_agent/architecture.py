"""Reviewed, bounded architecture language for FM-hybrid experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


COMPOSED_PREFIX = "composed:v1:"
REVIEWED_OPERATORS = ("embedding_mlp", "bi_interaction_mlp", "cross_network")
REVIEWED_FUSIONS = ("add", "learned_gate")
REVIEWED_WIDTHS = (16, 32, 64)
REVIEWED_DEPTHS = (1, 2, 3)
REVIEWED_DROPOUTS = (0.0, 0.1, 0.2)
REVIEWED_CROSS_LAYERS = (1, 2, 3)


@dataclass(frozen=True)
class ReviewedArchitectureSpec:
    """A safe model graph assembled only from pre-implemented operators."""

    interaction_paths: tuple[str, ...]
    fusion: str
    hidden_width: int
    hidden_depth: int
    dropout: float
    cross_layers: int

    def __post_init__(self) -> None:
        if not 1 <= len(self.interaction_paths) <= 2:
            raise ValueError("architecture requires one or two interaction paths")
        if len(set(self.interaction_paths)) != len(self.interaction_paths):
            raise ValueError("architecture interaction paths must be unique")
        unknown = set(self.interaction_paths) - set(REVIEWED_OPERATORS)
        if unknown:
            raise ValueError(f"unreviewed architecture operators: {sorted(unknown)}")
        canonical_paths = tuple(
            operator for operator in REVIEWED_OPERATORS if operator in self.interaction_paths
        )
        if self.interaction_paths != canonical_paths:
            raise ValueError(f"interaction paths must use canonical order {canonical_paths}")
        if self.fusion not in REVIEWED_FUSIONS:
            raise ValueError(f"unreviewed architecture fusion: {self.fusion}")
        if self.fusion == "learned_gate" and len(self.interaction_paths) < 2:
            raise ValueError("learned_gate requires two interaction paths")
        if self.hidden_width not in REVIEWED_WIDTHS:
            raise ValueError(f"hidden_width must be one of {REVIEWED_WIDTHS}")
        if self.hidden_depth not in REVIEWED_DEPTHS:
            raise ValueError(f"hidden_depth must be one of {REVIEWED_DEPTHS}")
        if self.dropout not in REVIEWED_DROPOUTS:
            raise ValueError(f"dropout must be one of {REVIEWED_DROPOUTS}")
        if self.cross_layers not in REVIEWED_CROSS_LAYERS:
            raise ValueError(f"cross_layers must be one of {REVIEWED_CROSS_LAYERS}")
        has_mlp = bool({"embedding_mlp", "bi_interaction_mlp"} & set(self.interaction_paths))
        if not has_mlp and (self.hidden_width, self.hidden_depth, self.dropout) != (32, 2, 0.1):
            raise ValueError("MLP settings must use canonical defaults when no MLP path is selected")
        if "cross_network" not in self.interaction_paths and self.cross_layers != 2:
            raise ValueError("cross_layers must use canonical default 2 when no cross path is selected")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewedArchitectureSpec":
        try:
            return cls(
                interaction_paths=tuple(str(item) for item in value["interaction_paths"]),
                fusion=str(value["fusion"]),
                hidden_width=int(value["hidden_width"]),
                hidden_depth=int(value["hidden_depth"]),
                dropout=float(value["dropout"]),
                cross_layers=int(value["cross_layers"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reviewed architecture specification: {exc}") from exc

    @property
    def architecture_id(self) -> str:
        paths = "+".join(self.interaction_paths)
        return (
            f"{COMPOSED_PREFIX}{paths}:{self.fusion}:w{self.hidden_width}:"
            f"d{self.hidden_depth}:p{self.dropout:g}:c{self.cross_layers}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "language_version": 1,
            "family": "fm_hybrid",
            "base": "immutable_fm_second_order",
            "interaction_paths": list(self.interaction_paths),
            "fusion": self.fusion,
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "dropout": self.dropout,
            "cross_layers": self.cross_layers,
            "residual_to_fm": True,
            "structural_diff_from_fm": {
                "added_modules": [*self.interaction_paths, self.fusion],
                "removed_modules": [],
            },
        }


def parse_architecture_id(architecture: str) -> ReviewedArchitectureSpec | None:
    """Parse a canonical composed ID; return None for legacy architecture aliases."""
    if not architecture.startswith(COMPOSED_PREFIX):
        return None
    parts = architecture[len(COMPOSED_PREFIX):].split(":")
    if len(parts) != 6:
        raise ValueError("invalid composed architecture identifier")
    paths, fusion, width, depth, dropout, cross_layers = parts
    if not (width.startswith("w") and depth.startswith("d") and dropout.startswith("p") and cross_layers.startswith("c")):
        raise ValueError("invalid composed architecture identifier fields")
    spec = ReviewedArchitectureSpec(
        interaction_paths=tuple(paths.split("+")),
        fusion=fusion,
        hidden_width=int(width[1:]),
        hidden_depth=int(depth[1:]),
        dropout=float(dropout[1:]),
        cross_layers=int(cross_layers[1:]),
    )
    if spec.architecture_id != architecture:
        raise ValueError("composed architecture identifier is not canonical")
    return spec


def controlled_single_path_ablations(architecture: str) -> tuple[str, ...]:
    """Return canonical one-path children for a reviewed two-path composition."""
    spec = parse_architecture_id(architecture)
    if spec is None or len(spec.interaction_paths) != 2:
        return ()
    ablations: list[str] = []
    for retained_path in spec.interaction_paths:
        has_mlp = retained_path in {"embedding_mlp", "bi_interaction_mlp"}
        ablation = ReviewedArchitectureSpec(
            interaction_paths=(retained_path,),
            fusion="add",
            hidden_width=spec.hidden_width if has_mlp else 32,
            hidden_depth=spec.hidden_depth if has_mlp else 2,
            dropout=spec.dropout if has_mlp else 0.1,
            cross_layers=spec.cross_layers if retained_path == "cross_network" else 2,
        )
        ablations.append(ablation.architecture_id)
    return tuple(ablations)


def architecture_schema() -> dict[str, Any]:
    """Strict JSON-schema fragment used by the generic implementer for architecture work."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "interaction_paths", "fusion", "hidden_width", "hidden_depth", "dropout", "cross_layers",
        ],
        "properties": {
            "interaction_paths": {
                "type": "array", "minItems": 1, "maxItems": 2,
                "items": {"type": "string", "enum": list(REVIEWED_OPERATORS)},
            },
            "fusion": {"type": "string", "enum": list(REVIEWED_FUSIONS)},
            "hidden_width": {"type": "integer", "enum": list(REVIEWED_WIDTHS)},
            "hidden_depth": {"type": "integer", "enum": list(REVIEWED_DEPTHS)},
            "dropout": {"type": "number", "enum": list(REVIEWED_DROPOUTS)},
            "cross_layers": {"type": "integer", "enum": list(REVIEWED_CROSS_LAYERS)},
        },
    }
