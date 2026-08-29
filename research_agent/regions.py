"""Explicit, lightweight management of independent research regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .planner import ResearchDirection
from .review import ReviewDecision
from .search import SearchState
from .state import ResearchState


@dataclass(frozen=True)
class RegionSnapshot:
    """Evidence summary for one region of the approved search catalogue."""

    region_id: str
    direction_id: str
    attempts: int
    best_primary: float | None
    status: str


class SearchRegionManager:
    """Keeps searches diverse without introducing another optimisation dependency.

    A region is a line of evidence for one high-level research direction.  This
    first implementation deliberately uses history rather than a surrogate
    model: it is reproducible, easy to inspect, and leaves Optuna/TPE as a
    reviewed future replacement rather than silently adding a dependency.
    """

    def snapshots(self, history: Sequence[Mapping[str, Any]]) -> list[RegionSnapshot]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in history:
            direction_id = str(item.get("direction_id") or "unassigned")
            grouped.setdefault(direction_id, []).append(item)
        snapshots: list[RegionSnapshot] = []
        for direction_id, items in sorted(grouped.items()):
            values = [
                float(item.get("metrics", {}).get("primary"))
                for item in items
                if item.get("metrics", {}).get("primary") is not None
            ]
            last = items[-1]
            snapshots.append(
                RegionSnapshot(
                    region_id=str(last.get("search_region_id") or self.region_id(direction_id)),
                    direction_id=direction_id,
                    attempts=len(items),
                    best_primary=max(values) if values else None,
                    status="PROMISING" if any(item.get("decision") == "accepted" for item in items) else "EXPLORING",
                )
            )
        return snapshots

    @staticmethod
    def region_id(direction_id: str, *, restart_index: int = 0) -> str:
        suffix = "" if restart_index == 0 else f"_restart_{restart_index:02d}"
        return f"region_{direction_id}{suffix}"

    def choose_search_state(
        self,
        direction: ResearchDirection,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        review: ReviewDecision,
    ) -> SearchState:
        prior = [item for item in history if item.get("direction_id") == direction.direction_id]
        if review.action == "restart":
            restart_index = 1 + sum(
                str(item.get("search_region_id", "")).startswith(self.region_id(direction.direction_id, restart_index=1).rsplit("_", 1)[0])
                for item in prior
            )
            return SearchState(
                status="RESTARTING",
                region_id=self.region_id(direction.direction_id, restart_index=restart_index),
                strategy="diverse_restart",
            )
        if any(item.get("decision") == "accepted" for item in prior):
            return SearchState(status="EXPLOITING", region_id=self.region_id(direction.direction_id), strategy="local_refinement")
        return SearchState(status="EXPLORING", region_id=self.region_id(direction.direction_id), strategy=direction.strategy)
