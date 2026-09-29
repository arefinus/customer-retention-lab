"""Training-only preprocessing: standardise numerics, one-hot categoricals.

Implemented directly rather than through a ColumnTransformer so the fitted statistics
are plain numbers that can be inspected, serialised to JSON and compared in tests.
An unseen category at transform time maps to an all-zero indicator block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import SchemaError


class Preprocessor:
    def __init__(self, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> None:
        self.numeric = tuple(numeric)
        self.categorical = tuple(categorical)
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.categories_: dict[str, list[str]] = {}

    @property
    def is_fitted(self) -> bool:
        return self.means_ is not None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self._check_columns(X)
        num = X[list(self.numeric)].to_numpy(dtype=float)
        if num.shape[0] == 0:
            raise SchemaError("Cannot fit a preprocessor on zero rows.")
        self.means_ = num.mean(axis=0)
        stds = num.std(axis=0)
        self.stds_ = np.where(stds > 0, stds, 1.0)
        self.categories_ = {c: sorted(X[c].astype(str).unique().tolist()) for c in self.categorical}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted; fit on the training partition first.")
        self._check_columns(X)
        num = (X[list(self.numeric)].to_numpy(dtype=float) - self.means_) / self.stds_
        blocks = [num]
        for c in self.categorical:
            cats = self.categories_[c]
            col = X[c].astype(str).to_numpy()
            onehot = np.zeros((len(X), len(cats)), dtype=float)
            index = {v: i for i, v in enumerate(cats)}
            for r, v in enumerate(col):
                j = index.get(v)
                if j is not None:
                    onehot[r, j] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks)

    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for c in self.categorical:
            names += [f"{c}={v}" for v in self.categories_.get(c, [])]
        return names

    def fitted_statistics(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")
        return {
            "means": {n: float(m) for n, m in zip(self.numeric, self.means_)},
            "stds": {n: float(s) for n, s in zip(self.numeric, self.stds_)},
            "categories": {c: list(v) for c, v in self.categories_.items()},
        }

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.numeric + self.categorical if c not in X.columns]
        if missing:
            raise SchemaError(f"Feature frame is missing columns {missing}.")
