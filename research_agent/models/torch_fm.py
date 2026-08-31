"""PyTorch Factorization Machine candidates for approved research directions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import auxiliary_labels, encode_candidate

from ..metrics import evaluate_predictions
from ..architecture import ReviewedArchitectureSpec, parse_architecture_id
from ..runner import CandidateOutput, PreparedData


class TorchFM(nn.Module):
    """Second-order FM matching the official baseline's feature formulation."""

    def __init__(self, feature_dim: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(feature_dim, embedding_dim)
        self.linear = nn.Embedding(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        summed = embedded.sum(dim=1)
        interactions = 0.5 * ((summed.square()).sum(dim=1) - embedded.square().sum(dim=(1, 2)))
        return self.bias + self.linear(features).squeeze(-1).sum(dim=1) + interactions


class DeepFM(TorchFM):
    """FM plus a residual MLP over concatenated field embeddings."""

    def __init__(self, feature_dim: int, field_count: int, embedding_dim: int = 16) -> None:
        super().__init__(feature_dim, embedding_dim)
        self.deep = nn.Sequential(
            nn.Linear(field_count * embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.deep[-1].weight)
        nn.init.zeros_(self.deep[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        return super().forward(features) + self.deep(embedded.flatten(start_dim=1)).squeeze(-1)


class ResidualNFM(TorchFM):
    """FM plus a nonlinear residual over its vector-valued bi-interaction."""

    def __init__(self, feature_dim: int, embedding_dim: int = 16) -> None:
        super().__init__(feature_dim, embedding_dim)
        self.interaction_network = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.interaction_network[-1].weight)
        nn.init.zeros_(self.interaction_network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        summed = embedded.sum(dim=1)
        bi_interaction = 0.5 * (summed.square() - embedded.square().sum(dim=1))
        return super().forward(features) + self.interaction_network(bi_interaction).squeeze(-1)


class _CrossNetwork(nn.Module):
    """Bounded DCN-style explicit feature crosses followed by a scalar head."""

    def __init__(self, input_dim: int, layers: int) -> None:
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(layers)])
        self.output = nn.Linear(input_dim, 1)
        for weight in self.weights:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        initial = value
        crossed = value
        for weight, bias in zip(self.weights, self.biases):
            crossed = initial * torch.sum(crossed * weight, dim=1, keepdim=True) + bias + crossed
        return self.output(crossed).squeeze(-1)


def _mlp(input_dim: int, width: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, width), nn.ReLU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = width
    layers.append(nn.Linear(current, 1))
    network = nn.Sequential(*layers)
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class ComposedFM(TorchFM):
    """FM with reviewed residual interaction paths compiled from a bounded spec."""

    def __init__(
        self,
        feature_dim: int,
        field_count: int,
        embedding_dim: int,
        spec: ReviewedArchitectureSpec,
    ) -> None:
        super().__init__(feature_dim, embedding_dim)
        self.spec = spec
        flat_dim = field_count * embedding_dim
        paths: dict[str, nn.Module] = {}
        for operator in spec.interaction_paths:
            if operator == "embedding_mlp":
                paths[operator] = _mlp(flat_dim, spec.hidden_width, spec.hidden_depth, spec.dropout)
            elif operator == "bi_interaction_mlp":
                paths[operator] = _mlp(embedding_dim, spec.hidden_width, spec.hidden_depth, spec.dropout)
            elif operator == "cross_network":
                paths[operator] = _CrossNetwork(flat_dim, spec.cross_layers)
        self.paths = nn.ModuleDict(paths)
        self.gate = (
            nn.Linear(flat_dim, len(spec.interaction_paths))
            if spec.fusion == "learned_gate" else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        flattened = embedded.flatten(start_dim=1)
        summed = embedded.sum(dim=1)
        bi_interaction = 0.5 * (summed.square() - embedded.square().sum(dim=1))
        outputs: list[torch.Tensor] = []
        for operator in self.spec.interaction_paths:
            inputs = bi_interaction if operator == "bi_interaction_mlp" else flattened
            outputs.append(self.paths[operator](inputs).squeeze(-1))
        stacked = torch.stack(outputs, dim=1)
        if self.gate is None:
            residual = stacked.sum(dim=1)
        else:
            residual = (stacked * torch.softmax(self.gate(flattened), dim=1)).sum(dim=1)
        return super().forward(features) + residual


MULTITASK_OBJECTIVES: Mapping[str, tuple[str, float]] = {
    "multitask_click_w0.05": ("is_click", 0.05),
    "multitask_click_w0.1": ("is_click", 0.1),
    "multitask_click_w0.2": ("is_click", 0.2),
}


class MultiTaskSharedBackbone(nn.Module):
    """Keep the ranking model intact while sharing its embeddings with an auxiliary head."""

    def __init__(self, base_model: nn.Module, field_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.auxiliary_head = nn.Sequential(
            nn.Linear(field_count * embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.auxiliary_head[-1].weight)
        nn.init.zeros_(self.auxiliary_head[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_model(features)

    def forward_tasks(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        primary = self.base_model(features)
        embedding = getattr(self.base_model, "embedding")
        shared = embedding(features).flatten(start_dim=1)
        auxiliary = self.auxiliary_head(shared).squeeze(-1)
        return primary, auxiliary


ARCHITECTURE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "fm": {
        "family": "fm",
        "interaction_path": "scalar second-order factorization-machine interaction",
        "hidden_layers": [],
        "residual_to_fm": False,
        "structural_diff_from_fm": {"added_modules": [], "removed_modules": []},
    },
    "deepfm": {
        "family": "fm_hybrid",
        "interaction_path": "FM plus MLP over concatenated field embeddings",
        "hidden_layers": [64, 32],
        "dropout": 0.1,
        "residual_to_fm": True,
        "structural_diff_from_fm": {
            "added_modules": ["flattened_field_embedding_mlp"],
            "removed_modules": [],
        },
    },
    "nfm_residual": {
        "family": "fm_hybrid",
        "interaction_path": "FM plus MLP over vector-valued bi-interactions",
        "hidden_layers": [32],
        "dropout": 0.1,
        "residual_to_fm": True,
        "structural_diff_from_fm": {
            "added_modules": ["bi_interaction_vector", "nonlinear_interaction_residual"],
            "removed_modules": [],
        },
    },
}

MAX_ARCHITECTURE_OVERHEAD_PARAMETERS = 500_000
MAX_TOTAL_TRAINABLE_PARAMETERS = 10_000_000


def resolve_architecture_spec(architecture: str) -> dict[str, Any]:
    """Resolve a legacy alias or a canonical reviewed composition to evidence metadata."""
    if architecture in ARCHITECTURE_SPECS:
        return dict(ARCHITECTURE_SPECS[architecture])
    composed = parse_architecture_id(architecture)
    if composed is None:
        raise ValueError(f"unsupported model architecture: {architecture}")
    return composed.as_dict()


def build_model(
    architecture: str,
    *,
    feature_dim: int,
    field_count: int,
    embedding_dim: int,
) -> nn.Module:
    """Compile a reviewed declarative architecture into an executable module."""
    if architecture == "fm":
        model: nn.Module = TorchFM(feature_dim, embedding_dim)
    elif architecture == "deepfm":
        model = DeepFM(feature_dim, field_count, embedding_dim)
    elif architecture == "nfm_residual":
        model = ResidualNFM(feature_dim, embedding_dim)
    else:
        spec = parse_architecture_id(architecture)
        if spec is None:
            raise ValueError(f"unsupported model architecture: {architecture}")
        model = ComposedFM(feature_dim, field_count, embedding_dim, spec)
    total = sum(parameter.numel() for parameter in model.parameters())
    baseline = feature_dim * (embedding_dim + 1) + 1
    if total - baseline > MAX_ARCHITECTURE_OVERHEAD_PARAMETERS:
        raise ValueError("reviewed architecture exceeds the parameter-overhead safety bound")
    if total > MAX_TOTAL_TRAINABLE_PARAMETERS:
        raise ValueError("reviewed architecture exceeds the total-parameter safety bound")
    return model


def resolve_training_objective(value: str) -> tuple[str | None, float]:
    if value in {"", "pointwise"}:
        return None, 0.0
    if value not in MULTITASK_OBJECTIVES:
        raise ValueError(f"unsupported training objective: {value}")
    return MULTITASK_OBJECTIVES[value]


def build_candidate_model(
    config: Mapping[str, Any], *, feature_dim: int, field_count: int, embedding_dim: int
) -> nn.Module:
    """Reconstruct the inference graph, including any training-time auxiliary head."""
    model = build_model(
        str(config.get("architecture", "fm")),
        feature_dim=feature_dim,
        field_count=field_count,
        embedding_dim=embedding_dim,
    )
    auxiliary_name, _ = resolve_training_objective(str(config.get("training_objective", "pointwise")))
    if auxiliary_name is not None:
        model = MultiTaskSharedBackbone(model, field_count, embedding_dim)
    if sum(parameter.numel() for parameter in model.parameters()) > MAX_TOTAL_TRAINABLE_PARAMETERS:
        raise ValueError("candidate model exceeds the total-parameter safety bound")
    return model


@dataclass(frozen=True)
class TrainingSummary:
    best_epoch: int
    best_primary: float
    best_metrics: Mapping[str, Any]


def run_torch_fm_candidate(
    prepared: PreparedData,
    config: Mapping[str, Any],
    run_dir: Path,
) -> CandidateOutput:
    """Train a validation-only FM candidate using data prepared by data.py."""
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    feature_variant = str(config.get("feature_variant", "baseline"))
    encoded, feature_dim = encode_candidate(
        {"train": prepared.train_rows, "valid": prepared.validation_rows},
        feature_variant=feature_variant,
    )
    x_train, y_train, train_users = encoded["train"]
    x_valid, y_valid, valid_users = encoded["valid"]

    architecture = str(config.get("architecture", "fm"))
    architecture_spec = resolve_architecture_spec(architecture)
    architecture_spec["architecture"] = architecture
    architecture_spec["field_count"] = int(x_train.shape[1])
    architecture_spec["embedding_dim"] = int(config.get("embedding_dim", 16))
    training_objective = str(config.get("training_objective", "pointwise"))
    auxiliary_name, auxiliary_weight = resolve_training_objective(training_objective)
    architecture_spec["training_objective"] = training_objective
    architecture_spec["auxiliary_head"] = (
        {"label": auxiliary_name, "loss": "binary_cross_entropy", "weight": auxiliary_weight}
        if auxiliary_name is not None else None
    )
    architecture_spec_path = run_dir / "architecture_spec.json"
    architecture_spec_path.write_text(
        json.dumps(architecture_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = build_candidate_model(
        config,
        feature_dim=feature_dim,
        field_count=int(x_train.shape[1]),
        embedding_dim=int(config.get("embedding_dim", 16)),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["l2"]),
    )
    train_x = torch.as_tensor(x_train, dtype=torch.long)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    auxiliary_train = (
        torch.as_tensor(auxiliary_labels(prepared.train_rows, auxiliary_name), dtype=torch.float32)
        if auxiliary_name is not None else None
    )
    valid_x = torch.as_tensor(x_valid, dtype=torch.long)
    batch_size = int(config.get("batch_size", 8192))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 3))
    loss_name = str(config["loss"])
    pair_groups = _pair_groups(train_users, y_train) if loss_name == "pairwise" else None
    if loss_name not in {"pointwise", "pairwise"}:
        raise ValueError(f"unsupported loss: {loss_name}")
    if auxiliary_name is not None and loss_name != "pointwise":
        raise ValueError("multi-task supervision currently requires pointwise primary loss")
    if pair_groups is not None and not pair_groups:
        raise ValueError("pairwise loss requires at least one user with positive and negative training rows")

    best_primary = float("-inf")
    best_metrics: Mapping[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = _train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            batch_size=batch_size,
            seed=seed + epoch,
            loss_name=loss_name,
            pair_groups=pair_groups,
            auxiliary_labels_tensor=auxiliary_train,
            auxiliary_weight=auxiliary_weight,
        )
        scores = _predict(model, valid_x)
        metrics = evaluate_predictions(valid_users, y_valid, scores).as_dict()
        validation_logloss = float(np.mean(np.logaddexp(0.0, scores) - y_valid * scores))
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": epoch_losses["total"],
            "primary_train_loss": epoch_losses["primary"],
            "auxiliary_train_loss": epoch_losses["auxiliary"],
            "validation_logloss": validation_logloss,
            **metrics,
        })
        if float(metrics["primary"]) > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("candidate did not produce a valid validation state")
    model.load_state_dict(best_state)
    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": best_state,
            "feature_dim": feature_dim,
            "config": dict(config),
            "summary": {"best_epoch": best_epoch, "best_primary": best_primary, "metrics": dict(best_metrics)},
        },
        checkpoint_path,
    )
    final_scores = _predict(model, valid_x)
    stopped_early = len(epoch_rows) < epochs
    diagnostics = {
        "training": {
            "epochs_budgeted": epochs,
            "epochs_ran": len(epoch_rows),
            "best_epoch": best_epoch,
            "patience": patience,
            "stop_reason": "early_stopping_patience" if stopped_early else "fidelity_budget_exhausted",
            "loss_objective": loss_name,
            "training_objective": training_objective,
            "auxiliary_task": auxiliary_name,
            "auxiliary_loss_weight": auxiliary_weight,
            "auxiliary_train_positive_rate": (
                float(auxiliary_train.mean()) if auxiliary_train is not None else None
            ),
        },
        "epoch_curve": [
            {
                key: row[key]
                for key in (
                    "epoch", "train_loss", "primary_train_loss", "auxiliary_train_loss",
                    "validation_logloss", "GAUC", "nDCG@5", "primary",
                )
            }
            for row in epoch_rows
        ],
        "score_distribution": _score_diagnostics(final_scores),
        "feature_coverage": _feature_diagnostics(x_train, x_valid, feature_variant),
        "user_segments": _user_segment_diagnostics(valid_users, y_valid, final_scores),
        "stratified_validation": _stratified_validation_diagnostics(
            x_train=x_train,
            x_valid=x_valid,
            train_users=train_users,
            valid_users=valid_users,
            labels=y_valid,
            scores=final_scores,
        ),
        "model": {
            "family": architecture_spec["family"],
            "architecture": architecture,
            "architecture_spec": architecture_spec,
            "feature_dim": feature_dim,
            "field_count": int(x_train.shape[1]),
            "embedding_dim": int(config.get("embedding_dim", 16)),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "multi_task_shared_backbone": auxiliary_name is not None,
        },
    }
    return CandidateOutput(
        user_ids=valid_users,
        labels=y_valid,
        scores=final_scores,
        row_ids=np.arange(len(y_valid), dtype=np.int64),
        metadata={
            "framework": "pytorch",
            "model": architecture,
            "loss": loss_name,
            "training_objective": training_objective,
            "feature_variant": feature_variant,
            "best_epoch": best_epoch,
            "best_metrics": dict(best_metrics),
            "checkpoint_path": str(checkpoint_path),
            "architecture_spec_path": str(architecture_spec_path),
            "diagnostics": diagnostics,
        },
    )


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    loss_name: str,
    pair_groups: Sequence[tuple[np.ndarray, np.ndarray]] | None,
    auxiliary_labels_tensor: torch.Tensor | None = None,
    auxiliary_weight: float = 0.0,
) -> dict[str, float | None]:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    primary_losses: list[float] = []
    auxiliary_losses: list[float] = []
    if loss_name == "pointwise":
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            if auxiliary_labels_tensor is None:
                primary_loss = F.binary_cross_entropy_with_logits(model(features[index]), labels[index])
                auxiliary_loss = None
                loss = primary_loss
            else:
                if not isinstance(model, MultiTaskSharedBackbone):
                    raise TypeError("multi-task objective requires MultiTaskSharedBackbone")
                primary_logits, auxiliary_logits = model.forward_tasks(features[index])
                primary_loss = F.binary_cross_entropy_with_logits(primary_logits, labels[index])
                auxiliary_loss = F.binary_cross_entropy_with_logits(
                    auxiliary_logits, auxiliary_labels_tensor[index]
                )
                loss = primary_loss + auxiliary_weight * auxiliary_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            primary_losses.append(float(primary_loss.detach()))
            if auxiliary_loss is not None:
                auxiliary_losses.append(float(auxiliary_loss.detach()))
    else:
        assert pair_groups is not None
        rng = np.random.default_rng(seed)
        steps = math.ceil(len(labels) / batch_size)
        for _ in range(steps):
            selected = rng.integers(len(pair_groups), size=batch_size)
            positives = np.fromiter(
                (groups[0][rng.integers(len(groups[0]))] for groups in (pair_groups[i] for i in selected)),
                dtype=np.int64,
                count=batch_size,
            )
            negatives = np.fromiter(
                (groups[1][rng.integers(len(groups[1]))] for groups in (pair_groups[i] for i in selected)),
                dtype=np.int64,
                count=batch_size,
            )
            optimizer.zero_grad()
            difference = model(features[torch.from_numpy(positives)]) - model(features[torch.from_numpy(negatives)])
            loss = -F.logsigmoid(difference).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            primary_losses.append(float(loss.detach()))
    return {
        "total": float(np.mean(losses)),
        "primary": float(np.mean(primary_losses)),
        "auxiliary": float(np.mean(auxiliary_losses)) if auxiliary_losses else None,
    }


def _pair_groups(users: Sequence[Any], labels: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[Any, tuple[list[int], list[int]]] = {}
    for index, (user, label) in enumerate(zip(users, labels)):
        positive, negative = grouped.setdefault(user, ([], []))
        (positive if int(label) == 1 else negative).append(index)
    return [
        (np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64))
        for positive, negative in grouped.values()
        if positive and negative
    ]


def _predict(model: TorchFM, features: torch.Tensor, batch_size: int = 200_000) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            output.append(model(features[start : start + batch_size]).cpu().numpy())
    return np.concatenate(output)


def _write_epoch_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "epoch", "train_loss", "primary_train_loss", "auxiliary_train_loss",
        "validation_logloss", "GAUC", "nDCG@5", "primary", "users", "rows",
        "evaluator_sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_diagnostics(scores: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(scores)
    valid = scores[finite]
    if not len(valid):
        return {"count": int(len(scores)), "finite_fraction": 0.0}
    return {
        "count": int(len(scores)),
        "finite_fraction": float(finite.mean()),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "p01": float(np.quantile(valid, 0.01)),
        "p50": float(np.quantile(valid, 0.50)),
        "p99": float(np.quantile(valid, 0.99)),
    }


def _feature_diagnostics(
    train: np.ndarray,
    valid: np.ndarray,
    feature_variant: str,
) -> dict[str, Any]:
    names = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
    if train.shape[1] > len(names):
        names.append(feature_variant)
    fields = []
    for index in range(train.shape[1]):
        train_values = np.unique(train[:, index])
        valid_values = np.unique(valid[:, index])
        unseen_mask = ~np.isin(valid[:, index], train_values)
        fields.append({
            "field_index": index,
            "field_name": names[index] if index < len(names) else f"field_{index}",
            "train_unique": int(len(train_values)),
            "validation_unique": int(len(valid_values)),
            "validation_unseen_fraction": float(unseen_mask.mean()) if len(valid) else 0.0,
        })
    return {"feature_variant": feature_variant, "fields": fields}


def _user_segment_diagnostics(
    users: Sequence[Any],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    counts: dict[Any, int] = {}
    for user in users:
        counts[user] = counts.get(user, 0) + 1
    definitions = {
        "few_exposures_1_5": lambda count: count <= 5,
        "medium_exposures_6_20": lambda count: 6 <= count <= 20,
        "many_exposures_21_plus": lambda count: count >= 21,
    }
    output: dict[str, Any] = {}
    user_array = np.asarray(users, dtype=object)
    for name, predicate in definitions.items():
        mask = np.fromiter((predicate(counts[user]) for user in users), dtype=bool, count=len(users))
        if not mask.any():
            continue
        metrics = evaluate_predictions(user_array[mask].tolist(), labels[mask], scores[mask]).as_dict()
        output[name] = {
            "rows": int(mask.sum()),
            "users": len({users[index] for index in np.flatnonzero(mask)}),
            "positive_rate": float(labels[mask].mean()),
            "GAUC": metrics["GAUC"],
            "nDCG@5": metrics["nDCG@5"],
            "primary": metrics["primary"],
        }
    return output


def _stratified_validation_diagnostics(
    *,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    train_users: Sequence[Any],
    valid_users: Sequence[Any],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Slice validation metrics using boundaries derived only from training data."""
    train_activity: dict[Any, int] = {}
    for user in train_users:
        train_activity[user] = train_activity.get(user, 0) + 1
    activity_counts = np.fromiter(
        (train_activity.get(user, 0) for user in valid_users),
        dtype=np.int64,
        count=len(valid_users),
    )
    activity_masks = {
        "cold_start_0": activity_counts == 0,
        "low_activity_1_20": (activity_counts >= 1) & (activity_counts <= 20),
        "medium_activity_21_100": (activity_counts >= 21) & (activity_counts <= 100),
        "high_activity_101_plus": activity_counts >= 101,
    }

    seen_by_field = np.column_stack([
        np.isin(x_valid[:, index], np.unique(x_train[:, index]))
        for index in range(x_train.shape[1])
    ])
    unseen_counts = (~seen_by_field).sum(axis=1)
    coverage_masks = {
        "all_fields_seen": unseen_counts == 0,
        "one_unseen_field": unseen_counts == 1,
        "multiple_unseen_fields": unseen_counts >= 2,
    }
    vocabulary_hashes = {
        str(index): hashlib.sha256(
            np.unique(x_train[:, index]).astype("<i8", copy=False).tobytes()
        ).hexdigest()
        for index in range(x_train.shape[1])
    }
    user_activity = {
        name: _validation_slice(valid_users, labels, scores, mask)
        for name, mask in activity_masks.items()
        if mask.any()
    }
    feature_coverage = {
        name: _validation_slice(valid_users, labels, scores, mask)
        for name, mask in coverage_masks.items()
        if mask.any()
    }
    eligible = [
        (f"user_activity:{name}", value)
        for name, value in user_activity.items()
        if value.get("metrics_available") and value.get("rows", 0) >= 100
    ] + [
        (f"feature_coverage:{name}", value)
        for name, value in feature_coverage.items()
        if value.get("metrics_available") and value.get("rows", 0) >= 100
    ]
    weakest = min(eligible, key=lambda item: float(item[1]["primary"]), default=None)
    return {
        "selection_split": "valid",
        "test_data_used": False,
        "boundary_source": "training_only",
        "user_activity_boundaries": [0, 1, 20, 21, 100, 101],
        "training_vocabulary_sha256_by_field": vocabulary_hashes,
        "minimum_rows_for_weakest_stratum": 100,
        "user_activity": user_activity,
        "feature_coverage": feature_coverage,
        "weakest_statistically_eligible_stratum": (
            {"stratum": weakest[0], **weakest[1]} if weakest else None
        ),
    }


def _validation_slice(
    users: Sequence[Any],
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    sliced_users = [users[index] for index in indices]
    output: dict[str, Any] = {
        "rows": int(len(indices)),
        "users": len(set(sliced_users)),
        "positive_rate": float(labels[mask].mean()),
    }
    try:
        metrics = evaluate_predictions(
            sliced_users,
            labels[mask],
            scores[mask],
            split="valid",
        ).as_dict()
    except (ValueError, ZeroDivisionError) as exc:
        output.update({"metrics_available": False, "reason": str(exc)})
        return output
    if not all(math.isfinite(float(metrics[name])) for name in ("GAUC", "nDCG@5", "primary")):
        output.update({"metrics_available": False, "reason": "non-finite slice metrics"})
        return output
    output.update({
        "metrics_available": True,
        "GAUC": metrics["GAUC"],
        "nDCG@5": metrics["nDCG@5"],
        "primary": metrics["primary"],
    })
    return output
