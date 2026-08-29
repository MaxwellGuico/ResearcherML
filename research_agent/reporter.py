"""Renders human-readable reports only from structured run records."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import ArtifactStore


class MarkdownReporter:
    """Generates the readable log from the append-only artifact store."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def write(self, destination: str | Path | None = None) -> Path:
        target = Path(destination) if destination else self.store.root / "research_log.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return target

    def render(self) -> str:
        lines = ["# Research Run Log", ""]
        iterations = self.store.read_iterations()
        if not iterations:
            lines.extend(["No completed iterations have been recorded.", ""])
        for record in iterations:
            lines.extend(self._render_iteration(record))
        lines.extend(self._render_interventions())
        return "\n".join(lines).rstrip() + "\n"

    def _render_iteration(self, record: dict[str, Any]) -> list[str]:
        metrics = record.get("metrics", {})
        lines = [
            f"## {record['experiment_id']}",
            "",
            f"- Decision: {record['decision']}",
            f"- Hypothesis: {record['hypothesis']}",
            f"- Rationale: {record['rationale']}",
            f"- GAUC: {metrics.get('GAUC', 'unavailable')}",
            f"- nDCG@5: {metrics.get('nDCG@5', 'unavailable')}",
            f"- Primary: {metrics.get('primary', 'unavailable')}",
            f"- Delta primary: {record.get('delta_primary', 'unavailable')}",
            f"- Runtime seconds: {record.get('runtime_seconds', 'unavailable')}",
        ]
        if record.get("error"):
            lines.append(f"- Error: {record['error']}")
        if record.get("recovery"):
            lines.append(f"- Recovery: {record['recovery']}")
        lines.append("")
        return lines

    def _render_interventions(self) -> list[str]:
        interventions = self.store.read_interventions()
        lines = ["## Manual Intervention Summary", "", f"Manual interventions: {len(interventions)}", ""]
        for index, item in enumerate(interventions, start=1):
            lines.extend(
                [
                    f"{index}. {item['timestamp']} — {item.get('experiment_id') or 'run-wide'}",
                    f"   - Action: {item['description']}",
                    f"   - Reason: {item['reason']}",
                    f"   - Effect: {item['effect']}",
                    "",
                ]
            )
        return lines
