"""Evaluation metrics. Undefined quantities return None, never NaN or a guess."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    yv = np.asarray(y).astype(int).ravel()
    pv = np.asarray(p, dtype=float).ravel()
    if yv.shape != pv.shape:
        raise ValueError("y and p must have the same length")
    if np.any((pv < 0) | (pv > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return yv, pv


def brier(y, p) -> float | None:
    yv, pv = _arrays(y, p)
    if yv.size == 0:
        return None
    return float(np.mean((pv - yv) ** 2))


def auroc(y, p) -> float | None:
    """None unless both classes are present."""
    yv, pv = _arrays(y, p)
    if yv.size == 0 or len(np.unique(yv)) < 2:
        return None
    return float(roc_auc_score(yv, pv))


def average_precision(y, p) -> float | None:
    """None when there is no positive (AP is undefined without positives)."""
    yv, pv = _arrays(y, p)
    if yv.size == 0 or yv.sum() == 0:
        return None
    return float(average_precision_score(yv, pv))


def reliability_bins(y, p, n_bins: int = 10) -> list[dict]:
    """Equal-width bins on predicted probability with counts. Empty bins are kept with
    count 0 and None rates so a reader sees where predictions do not fall."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    yv, pv = _arrays(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pv, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        out.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": n,
                "mean_predicted": (float(pv[mask].mean()) if n else None),
                "observed_rate": (float(yv[mask].mean()) if n else None),
                "positives": int(yv[mask].sum()),
            }
        )
    return out


def precision_recall_at_k(y, p, k: int) -> dict:
    """Top-k by predicted risk (ties broken by index for determinism)."""
    yv, pv = _arrays(y, p)
    if k < 0:
        raise ValueError("k must be >= 0")
    k_eff = int(min(k, yv.size))
    order = np.lexsort((np.arange(yv.size), -pv))[:k_eff]
    hits = int(yv[order].sum()) if k_eff else 0
    total = int(yv.sum())
    return {
        "k": int(k),
        "k_effective": k_eff,
        "positives_in_top_k": hits,
        "total_positives": total,
        "precision_at_k": (hits / k_eff if k_eff else None),
        "recall_at_k": (hits / total if total else None),
    }


def stability_by_segment(y, p, segments, k: int | None = None) -> dict[str, dict]:
    """Per-segment metrics. A segment with no rows is reported with n = 0 and None
    metrics when it is listed explicitly; segments are otherwise taken from the data."""
    yv, pv = _arrays(y, p)
    seg = np.asarray(segments).astype(str).ravel()
    if seg.shape != yv.shape:
        raise ValueError("segments must align with y")
    out: dict[str, dict] = {}
    for s in sorted(set(seg.tolist())):
        m = seg == s
        entry = {
            "n": int(m.sum()),
            "positives": int(yv[m].sum()),
            "base_rate": (float(yv[m].mean()) if m.any() else None),
            "brier": brier(yv[m], pv[m]) if m.any() else None,
            "auroc": auroc(yv[m], pv[m]) if m.any() else None,
            "average_precision": average_precision(yv[m], pv[m]) if m.any() else None,
        }
        if k is not None and m.any():
            entry["at_k"] = precision_recall_at_k(yv[m], pv[m], k)
        out[s] = entry
    return out


def empty_segment_entry() -> dict:
    return {"n": 0, "positives": 0, "base_rate": None, "brier": None, "auroc": None, "average_precision": None}


def summarise(y, p, segments, k: int, n_bins: int = 10) -> dict:
    yv, pv = _arrays(y, p)
    return {
        "n": int(yv.size),
        "positives": int(yv.sum()),
        "base_rate": (float(yv.mean()) if yv.size else None),
        "brier": brier(yv, pv),
        "auroc": auroc(yv, pv),
        "average_precision": average_precision(yv, pv),
        "at_capacity": precision_recall_at_k(yv, pv, k),
        "reliability": reliability_bins(yv, pv, n_bins),
        "by_segment": stability_by_segment(yv, pv, segments, k),
    }
