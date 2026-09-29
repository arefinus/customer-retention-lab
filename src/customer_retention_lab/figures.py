"""Render docs/figures/*.svg from committed demo output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .errors import MissingDataError
from .svg import PALETTE, Canvas, Frame

NOTE = "Computed by `make demo` on synthetic fixture v1, seed 42. Not a benchmark result."


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise MissingDataError(f"{path} not found. Run `python -m customer_retention_lab demo` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def fig_cohort_curves(cohort_long: pd.DataFrame, out: Path) -> Path:
    """Cohorts grouped by signup quarter, size-weighted, drawn only over offsets that
    every cohort in the group has reached (so no partially censored point is shown)."""
    df = cohort_long.copy()
    df["quarter"] = pd.PeriodIndex(df["cohort"], freq="M").asfreq("Q").astype(str)
    c = Canvas(Frame(), "Cohort retention by signup quarter (synthetic accounts)")
    c.note(NOTE)
    c.axes("months since signup", "share of cohort still active")
    c.yticks(0.0, 1.0, fmt="{:.1f}")
    max_off = int(df["offset"].max())
    for k in range(0, max_off + 1, 3):
        c.xtick(c.x(k, 0, max_off), str(k))
    items = []
    for j, (q, g) in enumerate(df.groupby("quarter", sort=True)):
        n_cohorts = g["cohort"].nunique()
        counts = g.groupby("offset")["cohort"].nunique()
        full = counts[counts == n_cohorts].index
        gg = g[g["offset"].isin(full)]
        w = gg.groupby("offset").apply(lambda x: (x["retention"] * x["size"]).sum() / x["size"].sum())
        pts = [(c.x(int(k), 0, max_off), c.y(float(v), 0, 1)) for k, v in w.items()]
        c.polyline(pts, PALETTE[j % len(PALETTE)])
        items.append((q, PALETTE[j % len(PALETTE)]))
    c.legend(items, x=c.f.left + c.f.plot_w - 110, y=c.f.top + c.f.plot_h - 16 * len(items) - 10)
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_reliability(evaluation: dict, model: str, out: Path) -> Path:
    raw = evaluation["models"][model]["raw"]["reliability"]
    cal = evaluation["models"][model]["calibrated"]["reliability"]
    c = Canvas(Frame(width=520, height=420, left=70), f"Reliability on the test partition: {model}")
    c.note(NOTE)
    c.axes("mean predicted probability (bin)", "observed churn rate (bin)")
    c.yticks(0.0, 1.0, fmt="{:.1f}")
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        c.xtick(c.x(v, 0, 1), f"{v:.1f}")
    c.polyline([(c.x(0, 0, 1), c.y(0, 0, 1)), (c.x(1, 0, 1), c.y(1, 0, 1))], "#a0aec0", 1.0, dash="4,4")
    for j, (label, bins) in enumerate((("raw", raw), ("calibrated", cal))):
        pts = [(c.x(b["mean_predicted"], 0, 1), c.y(b["observed_rate"], 0, 1)) for b in bins if b["count"] > 0]
        c.polyline(pts, PALETTE[j], 1.5)
        for b in bins:
            if b["count"] > 0:
                r = 2.5 + 4.0 * min(1.0, b["count"] / max(1, max(x["count"] for x in bins)))
                c.circle(c.x(b["mean_predicted"], 0, 1), c.y(b["observed_rate"], 0, 1), r, PALETTE[j])
    c.legend([("raw (marker size = bin count)", PALETTE[0]), ("calibrated on validation", PALETTE[1])])
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_capacity_curve(capacity: dict, out: Path) -> Path:
    rows = capacity["realised_at_capacity"]
    ks = [r["capacity"] for r in rows]
    c = Canvas(Frame(), "Precision and recall at review capacity (test partition, per observation date)")
    c.note(NOTE)
    c.axes("review capacity k (accounts per observation date)", "rate")
    c.yticks(0.0, 1.0, fmt="{:.1f}")
    for k in ks:
        c.xtick(c.x(k, ks[0], ks[-1]), str(k))
    for j, key in enumerate(("precision_at_k", "recall_at_k")):
        pts = [(c.x(r["capacity"], ks[0], ks[-1]), c.y(r[key] or 0.0, 0, 1)) for r in rows]
        c.polyline(pts, PALETTE[j])
        for x, y in pts:
            c.circle(x, y, 3, PALETTE[j])
    c.legend([("precision at k", PALETTE[0]), ("recall at k", PALETTE[1])], x=c.f.left + c.f.plot_w - 130)
    out.write_text(c.render(), encoding="utf-8")
    return out


def fig_segment_stability(evaluation: dict, model: str, out: Path) -> Path:
    seg = evaluation["models"][model]["calibrated"]["by_segment"]
    names = sorted(seg)
    c = Canvas(Frame(width=560, height=360), f"AUROC by segment on the test partition: {model}")
    c.note(NOTE)
    c.axes("segment (plan tier)", "AUROC")
    c.yticks(0.0, 1.0, fmt="{:.1f}")
    gw = c.f.plot_w / max(1, len(names))
    for i, s in enumerate(names):
        v = seg[s]["auroc"]
        x0 = c.f.left + i * gw + gw * 0.2
        c.xtick(c.f.left + i * gw + gw / 2, f"{s} (n={seg[s]['n']})")
        if v is None:
            c.parts.append(f'<text x="{c.f.left + i * gw + gw / 2:.1f}" y="{c.f.top + c.f.plot_h - 8}" text-anchor="middle" fill="#718096">undefined</text>')
            continue
        top = c.y(v, 0, 1)
        c.rect(x0, top, gw * 0.6, c.f.top + c.f.plot_h - top, PALETTE[0])
        c.parts.append(f'<text x="{x0 + gw * 0.3:.1f}" y="{top - 4:.1f}" text-anchor="middle" fill="#2d3748">{v:.3f}</text>')
    out.write_text(c.render(), encoding="utf-8")
    return out


def render_all(output_dir: Path, figure_dir: Path, primary_model: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    evaluation = _load_json(output_dir / "evaluation.json")
    capacity = _load_json(output_dir / "capacity_scenario.json")
    cohort_path = output_dir / "cohort_retention_long.csv"
    if not cohort_path.exists():
        raise MissingDataError(f"{cohort_path} not found. Run `python -m customer_retention_lab demo` first.")
    cohort_long = pd.read_csv(cohort_path, dtype={"cohort": str})
    return [
        fig_cohort_curves(cohort_long, figure_dir / "cohort_retention.svg"),
        fig_reliability(evaluation, primary_model, figure_dir / f"reliability_{primary_model}.svg"),
        fig_capacity_curve(capacity, figure_dir / "capacity_curve.svg"),
        fig_segment_stability(evaluation, primary_model, figure_dir / f"segment_stability_{primary_model}.svg"),
    ]
