"""Deterministic safety checks for proposed research experiments."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract


@dataclass(frozen=True)
class ExperimentProposal:
    """A declarative candidate that can be checked before any training begins."""

    experiment_id: str
    hypothesis: str
    rationale: str
    config: Mapping[str, Any]
    changed_factors: tuple[str, ...]
    parent_experiment_id: str | None = None
    training_split: str = "train"
    selection_split: str = "valid"
    external_data_sources: tuple[str, ...] = ()
    uses_test_labels: bool = False
    loads_raw_csv: bool = False
    modified_files: tuple[str, ...] = ()
    requested_dependencies: tuple[str, ...] = ()
    model_family: str = "fm"
    human_reviewed: bool = False
    runtime_budget_seconds: float = 600.0
    research_direction_id: str | None = None
    search_strategy: str = ""
    search_region_id: str = ""
    portfolio_role: str | None = None

    def config_fingerprint(self) -> str:
        """Stable representation used to reject exact duplicate candidates."""
        return json.dumps(dict(self.config), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class SafetyReport:
    passed: bool
    violations: tuple[str, ...] = ()


class SafetyValidator:
    """Enforces benchmark and project rules before a runner is invoked."""

    def __init__(
        self,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        max_runtime_seconds: float = 600.0,
        allowed_dependencies: frozenset[str] = frozenset({"numpy", "torch"}),
        approved_model_families: frozenset[str] = frozenset({"fm"}),
        max_worker_threads: int = 2,
    ) -> None:
        self.contract = contract
        self.max_runtime_seconds = max_runtime_seconds
        self.allowed_dependencies = allowed_dependencies
        self.approved_model_families = approved_model_families
        self.max_worker_threads = max_worker_threads

    def validate(
        self,
        proposal: ExperimentProposal,
        *,
        historical_configs: Sequence[Mapping[str, Any]] = (),
        parent_config: Mapping[str, Any] | None = None,
    ) -> SafetyReport:
        violations: list[str] = []
        if not proposal.hypothesis.strip():
            violations.append("proposal must state a hypothesis")
        if not proposal.rationale.strip():
            violations.append("proposal must state a rationale")
        if len(proposal.changed_factors) != 1:
            violations.append("proposal must change exactly one main factor")
        if proposal.portfolio_role not in {
            None, "single_worker", "incumbent_exploit", "independent_explore"
        }:
            violations.append("proposal has an unsupported portfolio role")
        if parent_config is not None:
            actual_changes = self._changed_config_keys(parent_config, proposal.config)
            declared_changes = set(proposal.changed_factors)
            expected_changes = set() if proposal.search_strategy == "promotion" else declared_changes
            if actual_changes != expected_changes:
                violations.append(
                    "declared changed_factors do not match actual configuration diff: "
                    f"declared={sorted(expected_changes)}, actual={sorted(actual_changes)}"
                )
        if proposal.training_split != self.contract.train_split:
            violations.append("training must use only the train split")
        if proposal.selection_split != self.contract.selection_split:
            violations.append("candidate selection must use only the valid split")
        if proposal.external_data_sources:
            violations.append("external datasets are not permitted")
        if proposal.uses_test_labels:
            violations.append("test labels must not be used for optimization")
        if proposal.loads_raw_csv:
            violations.append("controllers, runners, and models must use data.py rather than raw CSV files")
        protected = set(proposal.modified_files) & set(self.contract.protected_modules)
        if protected:
            violations.append(f"protected benchmark files may not be modified: {', '.join(sorted(protected))}")
        unknown_dependencies = set(proposal.requested_dependencies) - self.allowed_dependencies
        if unknown_dependencies:
            violations.append(
                "unapproved dependencies requested: " + ", ".join(sorted(unknown_dependencies))
            )
        if proposal.model_family not in self.approved_model_families and not proposal.human_reviewed:
            violations.append("a substantially different model family requires human review")
        if not 0 < proposal.runtime_budget_seconds <= self.max_runtime_seconds:
            violations.append(
                f"runtime budget must be greater than 0 and at most {self.max_runtime_seconds:g} seconds"
            )
        worker_threads = proposal.config.get("worker_threads", 1)
        if not isinstance(worker_threads, int) or not 1 <= worker_threads <= self.max_worker_threads:
            violations.append(
                f"worker_threads must be an integer from 1 to {self.max_worker_threads}"
            )
        if proposal.config_fingerprint() in {self._fingerprint(config) for config in historical_configs}:
            violations.append("proposal duplicates an existing experiment configuration")
        return SafetyReport(passed=not violations, violations=tuple(violations))

    @staticmethod
    def _fingerprint(config: Mapping[str, Any]) -> str:
        return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _changed_config_keys(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> set[str]:
        # Fidelity and epoch count are evaluation-budget metadata, not the
        # conceptual factor under study.
        ignored = {"epochs", "fidelity", "worker_threads", "seed"}
        keys = set(parent) | set(candidate)
        return {key for key in keys if key not in ignored and parent.get(key) != candidate.get(key)}


def has_measured_validation_evidence(record: Mapping[str, Any]) -> bool:
    """A configuration is consumed only after a finite validation primary exists."""
    primary = record.get("metrics", {}).get("primary")
    return (
        isinstance(primary, (int, float))
        and not isinstance(primary, bool)
        and math.isfinite(float(primary))
    )


def measured_historical_configs(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return unique configs backed by measured validation evidence."""
    configs: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        config = record.get("config")
        if not isinstance(config, Mapping) or not has_measured_validation_evidence(record):
            continue
        fingerprint = SafetyValidator._fingerprint(config)
        if fingerprint not in seen:
            seen.add(fingerprint)
            configs.append(config)
    return configs
