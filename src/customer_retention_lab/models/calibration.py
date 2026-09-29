"""Probability calibration fitted on the validation partition only.

`isotonic` uses a monotone step function; `platt` fits a logistic function of the
logit. Both are exported as plain numbers so no estimator is pickled.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..errors import ConfigError

METHODS = ("isotonic", "platt")
EPS = 1e-6


class Calibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in METHODS:
            raise ConfigError(f"unknown calibration method {method!r}; choose from {METHODS}")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._a: float | None = None
        self._b: float | None = None
        self.n_fit_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None or self._a is not None

    def fit(self, p, y) -> "Calibrator":
        pv = np.clip(np.asarray(p, dtype=float).ravel(), EPS, 1 - EPS)
        yv = np.asarray(y).astype(int).ravel()
        if pv.size == 0 or pv.shape != yv.shape:
            raise ConfigError("Calibrator needs equal-length, non-empty p and y.")
        if len(np.unique(yv)) < 2:
            raise ConfigError("Calibration labels contain a single class; cannot calibrate.")
        if self.method == "isotonic":
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pv, yv)
        else:
            z = np.log(pv / (1 - pv)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, yv)
            self._a, self._b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.n_fit_ = int(pv.size)
        return self

    def transform(self, p) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted; fit on the validation partition first.")
        pv = np.clip(np.asarray(p, dtype=float).ravel(), EPS, 1 - EPS)
        if self.method == "isotonic":
            return np.asarray(self._iso.predict(pv), dtype=float)
        z = np.log(pv / (1 - pv))
        return 1.0 / (1.0 + np.exp(-(self._a * z + self._b)))

    def to_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted.")
        out: dict = {"method": self.method, "fitted_on": "validation", "n_fit": self.n_fit_}
        if self.method == "isotonic":
            out["x_thresholds"] = [float(v) for v in self._iso.X_thresholds_]
            out["y_thresholds"] = [float(v) for v in self._iso.y_thresholds_]
        else:
            out["a"], out["b"] = self._a, self._b
        return out
