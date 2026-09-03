"""A compact NumPy MLP for dual-aware arc scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    positive_weight: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("hidden_dim, epochs, and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.positive_weight is not None and self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    losses: tuple[float, ...]
    positive_weight: float


class NumpyMLPArcScorer:
    """Two-layer ReLU classifier trained with weighted BCE and Adam.

    The small implementation keeps the learning component auditable and avoids a
    heavyweight runtime dependency. It is a baseline, not a claim of matching a
    paper-scale attention or reinforcement-learning architecture.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, *, seed: int = 0) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mean = np.zeros(input_dim, dtype=float)
        self.scale = np.ones(input_dim, dtype=float)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=float)
        self.metadata: dict[str, object] = {}

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape (n, {self.input_dim})")
        return (features - self.mean) / self.scale

    def _forward(self, standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_linear = standardized @ self.w1 + self.b1
        hidden = np.maximum(hidden_linear, 0.0)
        logits = (hidden @ self.w2 + self.b2).reshape(-1)
        return hidden_linear, hidden, logits

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        _, _, logits = self._forward(self._standardize(features))
        return self._sigmoid(logits)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: TrainingConfig | None = None,
    ) -> TrainingHistory:
        config = config or TrainingConfig(hidden_dim=self.hidden_dim)
        if config.hidden_dim != self.hidden_dim:
            raise ValueError("config.hidden_dim must match the model hidden_dim")
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] != y.shape[0]:
            raise ValueError("features and labels have incompatible shapes")
        if x.shape[0] == 0:
            raise ValueError("training data must be nonempty")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must be finite")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("labels must be binary")

        self.mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        x_standardized = self._standardize(x)
        positives = max(float(np.sum(y)), 1.0)
        negatives = max(float(y.size - np.sum(y)), 1.0)
        positive_weight = config.positive_weight or negatives / positives

        parameters = [self.w1, self.b1, self.w2, self.b2]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        step = 0
        rng = np.random.default_rng(config.seed)
        losses: list[float] = []

        for _ in range(config.epochs):
            order = rng.permutation(x_standardized.shape[0])
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                xb = x_standardized[indices]
                yb = y[indices]
                hidden_linear, hidden, logits = self._forward(xb)
                probabilities = self._sigmoid(logits)
                weights = np.where(yb > 0.5, positive_weight, 1.0)
                dlogits = weights * (probabilities - yb) / max(len(indices), 1)

                grad_w2 = hidden.T @ dlogits[:, None] + config.weight_decay * self.w2
                grad_b2 = np.asarray([np.sum(dlogits)])
                grad_hidden = dlogits[:, None] @ self.w2.T
                grad_hidden_linear = grad_hidden * (hidden_linear > 0.0)
                grad_w1 = xb.T @ grad_hidden_linear + config.weight_decay * self.w1
                grad_b1 = np.sum(grad_hidden_linear, axis=0)
                gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

                step += 1
                for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients, strict=True)
                ):
                    first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * gradient
                    second_moments[index] = (
                        beta2 * second_moments[index] + (1.0 - beta2) * gradient * gradient
                    )
                    first_hat = first_moments[index] / (1.0 - beta1**step)
                    second_hat = second_moments[index] / (1.0 - beta2**step)
                    parameter -= config.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

            probabilities = np.clip(self.predict_proba(x), 1e-9, 1.0 - 1e-9)
            sample_weights = np.where(y > 0.5, positive_weight, 1.0)
            loss = -np.mean(
                sample_weights
                * (y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))
            )
            loss += (
                0.5
                * config.weight_decay
                * (float(np.sum(self.w1 * self.w1)) + float(np.sum(self.w2 * self.w2)))
            )
            losses.append(float(loss))

        return TrainingHistory(losses=tuple(losses), positive_weight=float(positive_weight))

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined_metadata = dict(self.metadata)
        if metadata:
            combined_metadata.update(metadata)
        np.savez_compressed(
            path,
            input_dim=np.asarray([self.input_dim], dtype=int),
            hidden_dim=np.asarray([self.hidden_dim], dtype=int),
            mean=self.mean,
            scale=self.scale,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            metadata_json=np.asarray([json.dumps(combined_metadata, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> NumpyMLPArcScorer:
        with np.load(Path(path), allow_pickle=False) as payload:
            input_dim = int(payload["input_dim"][0])
            hidden_dim = int(payload["hidden_dim"][0])
            model = cls(input_dim=input_dim, hidden_dim=hidden_dim, seed=0)
            model.mean = np.asarray(payload["mean"], dtype=float)
            model.scale = np.asarray(payload["scale"], dtype=float)
            model.w1 = np.asarray(payload["w1"], dtype=float)
            model.b1 = np.asarray(payload["b1"], dtype=float)
            model.w2 = np.asarray(payload["w2"], dtype=float)
            model.b2 = np.asarray(payload["b2"], dtype=float)
            metadata_text = str(payload["metadata_json"][0])
            model.metadata = cast(dict[str, object], json.loads(metadata_text))
        return model


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have the same shape")
    predicted = p >= threshold
    truth = y >= 0.5
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    true_negative = int(np.sum(~predicted & ~truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / max(len(y), 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(truth)) if len(y) else 0.0,
    }
