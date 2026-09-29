"""Two churn-risk baselines behind one interface: logistic regression and
histogram gradient boosting. Both fit their preprocessor on the training frame only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..errors import ConfigError
from ..features.builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .preprocess import Preprocessor

MODEL_NAMES = ("logistic", "gbm")


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    logistic_C: float = 1.0
    logistic_max_iter: int = 500
    gbm_max_iter: int = 200
    gbm_learning_rate: float = 0.05
    gbm_max_leaf_nodes: int = 15
    gbm_min_samples_leaf: int = 40

    @classmethod
    def from_mapping(cls, m: Mapping) -> "ModelConfig":
        lg, gb = m.get("logistic", {}), m.get("gbm", {})
        return cls(
            seed=int(m.get("seed", 42)),
            logistic_C=float(lg.get("C", 1.0)),
            logistic_max_iter=int(lg.get("max_iter", 500)),
            gbm_max_iter=int(gb.get("max_iter", 200)),
            gbm_learning_rate=float(gb.get("learning_rate", 0.05)),
            gbm_max_leaf_nodes=int(gb.get("max_leaf_nodes", 15)),
            gbm_min_samples_leaf=int(gb.get("min_samples_leaf", 40)),
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ChurnBaseline:
    def __init__(self, name: str, config: ModelConfig | None = None) -> None:
        if name not in MODEL_NAMES:
            raise ConfigError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
        self.name = name
        self.config = config or ModelConfig()
        self.preprocessor = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        self._model = None

    def _new_model(self):
        c = self.config
        if self.name == "logistic":
            return LogisticRegression(C=c.logistic_C, max_iter=c.logistic_max_iter, random_state=c.seed)
        return HistGradientBoostingClassifier(
            max_iter=c.gbm_max_iter,
            learning_rate=c.gbm_learning_rate,
            max_leaf_nodes=c.gbm_max_leaf_nodes,
            min_samples_leaf=c.gbm_min_samples_leaf,
            random_state=c.seed,
        )

    def fit(self, X: pd.DataFrame, y) -> "ChurnBaseline":
        yv = np.asarray(y).astype(int)
        if len(np.unique(yv)) < 2:
            raise ConfigError("Training labels contain a single class; cannot fit a classifier.")
        self.preprocessor.fit(X)
        self._model = self._new_model().fit(self.preprocessor.transform(X), yv)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted; call fit() on the training partition first.")
        return self._model.predict_proba(self.preprocessor.transform(X))[:, 1]

    def export(self) -> dict:
        """JSON-serialisable description. Logistic weights are exported in full; the
        boosting model is described by its settings (no pickled estimator is written)."""
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        out = {"model": self.name, "config": self.config.to_dict(), "preprocessor": self.preprocessor.fitted_statistics()}
        if self.name == "logistic":
            names = self.preprocessor.feature_names()
            out["coefficients"] = {n: float(w) for n, w in zip(names, self._model.coef_[0])}
            out["intercept"] = float(self._model.intercept_[0])
        else:
            out["n_iter"] = int(self._model.n_iter_)
            out["n_features"] = len(self.preprocessor.feature_names())
        return out
