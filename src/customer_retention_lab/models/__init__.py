"""Churn-risk baselines, calibration, metrics, cohort curves and capacity ranking."""

from __future__ import annotations

from .calibration import Calibrator
from .churn import ChurnBaseline, ModelConfig
from .cohorts import cohort_retention, cohort_table, rfm_summary
from .metrics import auroc, average_precision, brier, precision_recall_at_k, reliability_bins, stability_by_segment, summarise
from .preprocess import Preprocessor
from .ranking import benefit_scenario, rank_at_capacity

__all__ = [
    "Calibrator",
    "ChurnBaseline",
    "ModelConfig",
    "Preprocessor",
    "auroc",
    "average_precision",
    "benefit_scenario",
    "brier",
    "cohort_retention",
    "cohort_table",
    "precision_recall_at_k",
    "rank_at_capacity",
    "reliability_bins",
    "rfm_summary",
    "stability_by_segment",
    "summarise",
]
