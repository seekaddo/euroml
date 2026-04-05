from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.features import feature_columns


@dataclass
class EnsembleWeights:
    prior: float = 0.20
    linear: float = 0.45
    tree: float = 0.35


def _window_label(window: int | None) -> str:
    return "all" if window is None else str(window)


def _slice_last_draws(frame: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if window is None:
        return frame

    draw_ordinals = sorted(int(value) for value in frame["draw_ordinal"].unique())
    if len(draw_ordinals) <= window:
        return frame

    start_ordinal = draw_ordinals[-window]
    return frame[frame["draw_ordinal"] >= start_ordinal]


class InclusionProbabilityModel:
    def __init__(
        self,
        random_state: int = 7,
        weights: EnsembleWeights | None = None,
    ) -> None:
        self.random_state = random_state
        self.weights = weights or EnsembleWeights()
        self._manual_weights = weights is not None
        self.columns = feature_columns()
        self.base_rate = 0.0
        self._constant_probability: float | None = None
        self.diagnostics: dict[str, float | int | dict[str, float]] = {}
        self.linear_model = self._build_linear_model()
        self.tree_model = self._build_tree_model()

    def _build_linear_model(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.5,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _build_tree_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            max_iter=200,
            random_state=self.random_state,
        )

    def fit(self, frame: pd.DataFrame) -> "InclusionProbabilityModel":
        if frame.empty:
            raise ValueError("Cannot fit on an empty frame")

        x_values = frame[self.columns].to_numpy(dtype=float)
        y_values = frame["target"].to_numpy(dtype=int)
        self.base_rate = float(np.mean(y_values))

        unique_targets = np.unique(y_values)
        if len(unique_targets) < 2:
            self._constant_probability = float(unique_targets[0])
            self.diagnostics = {
                "training_rows": int(len(frame)),
                "validation_draw_count": 0,
                "constant_probability": float(self._constant_probability),
                "weights": {
                    "prior": float(self.weights.prior),
                    "linear": float(self.weights.linear),
                    "tree": float(self.weights.tree),
                },
            }
            return self

        if not self._manual_weights:
            self.weights = self._learn_blend_weights(frame)

        self._constant_probability = None
        self.linear_model = self._build_linear_model()
        self.tree_model = self._build_tree_model()
        self.linear_model.fit(x_values, y_values)
        self.tree_model.fit(x_values, y_values)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)

        if self._constant_probability is not None:
            return np.full(len(frame), self._constant_probability, dtype=float)

        x_values = frame[self.columns].to_numpy(dtype=float)
        linear_probs = self.linear_model.predict_proba(x_values)[:, 1]
        tree_probs = self.tree_model.predict_proba(x_values)[:, 1]
        prior_probs = np.full(len(frame), self.base_rate, dtype=float)

        blended = (
            self.weights.prior * prior_probs
            + self.weights.linear * linear_probs
            + self.weights.tree * tree_probs
        )
        return np.clip(blended, 1e-6, 1.0 - 1e-6)

    def _learn_blend_weights(self, frame: pd.DataFrame) -> EnsembleWeights:
        train_frame, validation_frame, validation_draw_count = self._validation_split(frame)
        if train_frame is None or validation_frame is None:
            self.diagnostics = {
                "training_rows": int(len(frame)),
                "validation_draw_count": 0,
                "weights": {
                    "prior": float(self.weights.prior),
                    "linear": float(self.weights.linear),
                    "tree": float(self.weights.tree),
                },
            }
            return self.weights

        train_targets = train_frame["target"].to_numpy(dtype=int)
        validation_targets = validation_frame["target"].to_numpy(dtype=int)
        if len(np.unique(train_targets)) < 2 or len(np.unique(validation_targets)) < 2:
            self.diagnostics = {
                "training_rows": int(len(frame)),
                "validation_draw_count": int(validation_draw_count),
                "weights": {
                    "prior": float(self.weights.prior),
                    "linear": float(self.weights.linear),
                    "tree": float(self.weights.tree),
                },
            }
            return self.weights

        train_x = train_frame[self.columns].to_numpy(dtype=float)
        validation_x = validation_frame[self.columns].to_numpy(dtype=float)

        probe_linear = self._build_linear_model()
        probe_tree = self._build_tree_model()
        probe_linear.fit(train_x, train_targets)
        probe_tree.fit(train_x, train_targets)

        prior_rate = float(np.mean(train_targets))
        prior_probs = np.full(len(validation_frame), prior_rate, dtype=float)
        linear_probs = probe_linear.predict_proba(validation_x)[:, 1]
        tree_probs = probe_tree.predict_proba(validation_x)[:, 1]

        losses = {
            "prior": float(brier_score_loss(validation_targets, prior_probs)),
            "linear": float(brier_score_loss(validation_targets, linear_probs)),
            "tree": float(brier_score_loss(validation_targets, tree_probs)),
        }
        learned_weights = self._weights_from_losses(losses)
        blended_probs = (
            learned_weights.prior * prior_probs
            + learned_weights.linear * linear_probs
            + learned_weights.tree * tree_probs
        )
        blend_loss = float(brier_score_loss(validation_targets, blended_probs))
        self.diagnostics = {
            "training_rows": int(len(frame)),
            "validation_rows": int(len(validation_frame)),
            "validation_draw_count": int(validation_draw_count),
            "prior_brier": losses["prior"],
            "linear_brier": losses["linear"],
            "tree_brier": losses["tree"],
            "blend_brier": blend_loss,
            "weights": {
                "prior": float(learned_weights.prior),
                "linear": float(learned_weights.linear),
                "tree": float(learned_weights.tree),
            },
        }
        return learned_weights

    def _validation_split(
        self,
        frame: pd.DataFrame,
        min_draws: int = 40,
        min_validation_draws: int = 24,
        validation_fraction: float = 0.15,
        max_validation_draws: int = 104,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int]:
        draw_ordinals = sorted(int(value) for value in frame["draw_ordinal"].unique())
        if len(draw_ordinals) < min_draws:
            return None, None, 0

        validation_draw_count = max(min_validation_draws, int(round(len(draw_ordinals) * validation_fraction)))
        validation_draw_count = min(validation_draw_count, max_validation_draws, len(draw_ordinals) - min_validation_draws)
        if validation_draw_count <= 0:
            return None, None, 0

        validation_start = draw_ordinals[-validation_draw_count]
        train_frame = frame[frame["draw_ordinal"] < validation_start]
        validation_frame = frame[frame["draw_ordinal"] >= validation_start]
        if train_frame.empty or validation_frame.empty:
            return None, None, 0
        return train_frame, validation_frame, validation_draw_count

    def _weights_from_losses(self, losses: dict[str, float]) -> EnsembleWeights:
        inverse_losses = {
            name: 1.0 / max(value, 1e-9)
            for name, value in losses.items()
        }
        total = sum(inverse_losses.values())
        return EnsembleWeights(
            prior=float(inverse_losses["prior"] / total),
            linear=float(inverse_losses["linear"] / total),
            tree=float(inverse_losses["tree"] / total),
        )


class MultiHistoryInclusionProbabilityModel:
    def __init__(
        self,
        random_state: int = 7,
        history_windows: tuple[int | None, ...] = (None, 520, 260, 104),
    ) -> None:
        self.random_state = random_state
        self.history_windows = history_windows
        self.columns = feature_columns()
        self.expert_models: list[tuple[str, float, InclusionProbabilityModel]] = []
        self.diagnostics: dict[str, float | int | str | dict[str, float]] = {}

    def fit(self, frame: pd.DataFrame) -> "MultiHistoryInclusionProbabilityModel":
        if frame.empty:
            raise ValueError("Cannot fit on an empty frame")

        probe_model = InclusionProbabilityModel(random_state=self.random_state)
        train_frame, validation_frame, validation_draw_count = probe_model._validation_split(frame)

        if train_frame is None or validation_frame is None:
            self.expert_models = self._fit_final_experts(frame, self._equal_weights(frame))
            self.diagnostics = {
                "model_type": "multi_history",
                "training_rows": int(len(frame)),
                "validation_draw_count": 0,
                "expert_weights": {label: round(weight, 4) for label, weight, _ in self.expert_models},
            }
            return self

        validation_targets = validation_frame["target"].to_numpy(dtype=int)
        expert_losses: dict[str, float] = {}
        for window in self.history_windows:
            subset = _slice_last_draws(train_frame, window)
            if subset.empty:
                continue
            expert = InclusionProbabilityModel(random_state=self.random_state)
            expert.fit(subset)
            probabilities = expert.predict_proba(validation_frame)
            expert_losses[_window_label(window)] = float(brier_score_loss(validation_targets, probabilities))

        if not expert_losses:
            self.expert_models = self._fit_final_experts(frame, self._equal_weights(frame))
            self.diagnostics = {
                "model_type": "multi_history",
                "training_rows": int(len(frame)),
                "validation_draw_count": int(validation_draw_count),
                "expert_weights": {label: round(weight, 4) for label, weight, _ in self.expert_models},
            }
            return self

        learned_weights = self._weights_from_losses(expert_losses)
        self.expert_models = self._fit_final_experts(frame, learned_weights)
        blended_probabilities = self.predict_proba(validation_frame)
        blend_loss = float(brier_score_loss(validation_targets, blended_probabilities))
        self.diagnostics = {
            "model_type": "multi_history",
            "training_rows": int(len(frame)),
            "validation_rows": int(len(validation_frame)),
            "validation_draw_count": int(validation_draw_count),
            "blend_brier": blend_loss,
            "expert_brier": {label: round(loss, 6) for label, loss in expert_losses.items()},
            "expert_weights": {label: round(weight, 4) for label, weight, _ in self.expert_models},
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)
        if not self.expert_models:
            raise ValueError("Model must be fit before calling predict_proba")

        blended = np.zeros(len(frame), dtype=float)
        for _, weight, model in self.expert_models:
            blended += weight * model.predict_proba(frame)
        return np.clip(blended, 1e-6, 1.0 - 1e-6)

    def _equal_weights(self, frame: pd.DataFrame) -> dict[str, float]:
        labels = []
        for window in self.history_windows:
            subset = _slice_last_draws(frame, window)
            if not subset.empty:
                labels.append(_window_label(window))
        if not labels:
            labels = ["all"]
        weight = 1.0 / len(labels)
        return {label: weight for label in labels}

    def _fit_final_experts(
        self,
        frame: pd.DataFrame,
        weights: dict[str, float],
    ) -> list[tuple[str, float, InclusionProbabilityModel]]:
        experts: list[tuple[str, float, InclusionProbabilityModel]] = []
        for window in self.history_windows:
            label = _window_label(window)
            weight = weights.get(label)
            if weight is None or weight <= 0.0:
                continue
            subset = _slice_last_draws(frame, window)
            if subset.empty:
                continue
            expert = InclusionProbabilityModel(random_state=self.random_state)
            expert.fit(subset)
            experts.append((label, float(weight), expert))

        total_weight = sum(weight for _, weight, _ in experts)
        if total_weight <= 0.0:
            fallback = InclusionProbabilityModel(random_state=self.random_state).fit(frame)
            return [("all", 1.0, fallback)]

        return [
            (label, weight / total_weight, expert)
            for label, weight, expert in experts
        ]

    def _weights_from_losses(self, losses: dict[str, float]) -> dict[str, float]:
        inverse_losses = {
            label: 1.0 / max(loss, 1e-9)
            for label, loss in losses.items()
        }
        total = sum(inverse_losses.values())
        return {
            label: float(value / total)
            for label, value in inverse_losses.items()
        }


ModelFactory = Callable[[int], Any]


class FamilyBlendProbabilityModel:
    def __init__(
        self,
        factories: tuple[tuple[str, ModelFactory], ...],
        random_state: int = 7,
    ) -> None:
        self.factories = factories
        self.random_state = random_state
        self.expert_models: list[tuple[str, float, Any]] = []
        self.diagnostics: dict[str, float | int | str | dict[str, float]] = {}

    def fit(self, frame: pd.DataFrame) -> "FamilyBlendProbabilityModel":
        if frame.empty:
            raise ValueError("Cannot fit on an empty frame")

        probe_model = InclusionProbabilityModel(random_state=self.random_state)
        train_frame, validation_frame, validation_draw_count = probe_model._validation_split(frame)

        if train_frame is None or validation_frame is None:
            self.expert_models = self._fit_final_experts(frame, self._equal_weights())
            self.diagnostics = {
                "model_type": "family_blend",
                "training_rows": int(len(frame)),
                "validation_draw_count": 0,
                "expert_weights": {label: round(weight, 4) for label, weight, _ in self.expert_models},
            }
            return self

        validation_targets = validation_frame["target"].to_numpy(dtype=int)
        expert_losses: dict[str, float] = {}
        for label, factory in self.factories:
            expert = factory(self.random_state)
            expert.fit(train_frame)
            probabilities = expert.predict_proba(validation_frame)
            expert_losses[label] = float(brier_score_loss(validation_targets, probabilities))

        learned_weights = self._weights_from_losses(expert_losses)
        self.expert_models = self._fit_final_experts(frame, learned_weights)
        blended_probabilities = self.predict_proba(validation_frame)
        blend_loss = float(brier_score_loss(validation_targets, blended_probabilities))
        self.diagnostics = {
            "model_type": "family_blend",
            "training_rows": int(len(frame)),
            "validation_rows": int(len(validation_frame)),
            "validation_draw_count": int(validation_draw_count),
            "blend_brier": blend_loss,
            "expert_brier": {label: round(loss, 6) for label, loss in expert_losses.items()},
            "expert_weights": {label: round(weight, 4) for label, weight, _ in self.expert_models},
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)
        if not self.expert_models:
            raise ValueError("Model must be fit before calling predict_proba")

        blended = np.zeros(len(frame), dtype=float)
        for _, weight, model in self.expert_models:
            blended += weight * model.predict_proba(frame)
        return np.clip(blended, 1e-6, 1.0 - 1e-6)

    def _equal_weights(self) -> dict[str, float]:
        if not self.factories:
            return {}
        weight = 1.0 / len(self.factories)
        return {label: weight for label, _ in self.factories}

    def _fit_final_experts(
        self,
        frame: pd.DataFrame,
        weights: dict[str, float],
    ) -> list[tuple[str, float, Any]]:
        experts: list[tuple[str, float, Any]] = []
        for label, factory in self.factories:
            weight = weights.get(label, 0.0)
            if weight <= 0.0:
                continue
            expert = factory(self.random_state)
            expert.fit(frame)
            experts.append((label, float(weight), expert))

        total_weight = sum(weight for _, weight, _ in experts)
        if total_weight <= 0.0:
            label, factory = self.factories[0]
            fallback = factory(self.random_state).fit(frame)
            return [(label, 1.0, fallback)]

        return [
            (label, weight / total_weight, expert)
            for label, weight, expert in experts
        ]

    def _weights_from_losses(self, losses: dict[str, float]) -> dict[str, float]:
        inverse_losses = {
            label: 1.0 / max(loss, 1e-9)
            for label, loss in losses.items()
        }
        total = sum(inverse_losses.values())
        return {
            label: float(value / total)
            for label, value in inverse_losses.items()
        }
