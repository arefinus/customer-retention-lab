from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from customer_retention_lab.errors import ConfigError
from customer_retention_lab.features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from customer_retention_lab.models import Calibrator, ChurnBaseline, ModelConfig, Preprocessor
from customer_retention_lab.models.metrics import auroc, average_precision, brier, precision_recall_at_k, reliability_bins, stability_by_segment

FAST = ModelConfig(gbm_max_iter=30)


def test_preprocessor_statistics_do_not_change_when_test_rows_are_perturbed(trained):
    train = trained["split"].train[list(FEATURE_COLUMNS)]
    test = trained["split"].test[list(FEATURE_COLUMNS)].copy()
    p1 = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train)
    test["usage_mean"] *= 1000.0
    test["plan_tier"] = "tier-999"
    p2 = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train)
    assert p1.fitted_statistics() == p2.fitted_statistics()
    np.testing.assert_array_equal(p1.transform(train), p2.transform(train))


def test_unseen_category_is_handled_without_error(trained):
    train = trained["split"].train
    model = ChurnBaseline("logistic", FAST).fit(train[list(FEATURE_COLUMNS)], train["label"])
    row = train[list(FEATURE_COLUMNS)].head(3).copy()
    row["region"] = "region-UNSEEN"
    row["plan_tier"] = "tier-0"
    p = model.predict_proba(row)
    assert p.shape == (3,) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    enc = model.preprocessor.transform(row)
    onehot_block = enc[:, len(NUMERIC_FEATURES):]
    assert np.all(onehot_block == 0.0)


@pytest.mark.parametrize("name", ["logistic", "gbm"])
def test_baselines_give_probabilities_and_are_deterministic(trained, name):
    train = trained["split"].train
    X, y = train[list(FEATURE_COLUMNS)], train["label"]
    a = ChurnBaseline(name, FAST).fit(X, y).predict_proba(X)
    b = ChurnBaseline(name, FAST).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert np.all((a >= 0) & (a <= 1))
    assert auroc(y, a) is not None and auroc(y, a) > 0.5


def test_single_class_training_labels_raise(trained):
    train = trained["split"].train
    with pytest.raises(ConfigError):
        ChurnBaseline("logistic", FAST).fit(train[list(FEATURE_COLUMNS)], np.zeros(len(train)))
    with pytest.raises(ConfigError):
        ChurnBaseline("mystery")


def test_model_export_is_json_without_pickles(trained):
    ex = trained["exports"]
    s = json.dumps(ex)
    assert "coefficients" in ex["models"]["logistic"] and "intercept" in ex["models"]["logistic"]
    assert set(ex["models"]["logistic"]["coefficients"]) >= set(NUMERIC_FEATURES)
    assert ex["models"]["gbm"]["n_iter"] > 0
    assert "pickle" not in s.lower()


def test_calibrator_is_fitted_on_validation_only(trained):
    val, test = trained["split"].validation, trained["split"].test
    for name in ("logistic", "gbm"):
        c = trained["exports"]["calibration"][name]
        assert c["fitted_on"] == "validation" and c["n_fit"] == len(val) and c["n_fit"] != len(test)
    p_val = trained["predictions"]["validation"]["p_gbm_raw"].to_numpy()
    y_val = trained["predictions"]["validation"]["label"].to_numpy()
    p_test = trained["predictions"]["test"]["p_gbm_raw"].to_numpy()
    c1 = Calibrator("isotonic").fit(p_val, y_val)
    c2 = Calibrator("isotonic").fit(p_val, y_val)
    assert c1.to_dict() == c2.to_dict()
    # thresholds come from validation scores, and test scores are not needed to build them
    assert min(c1.to_dict()["x_thresholds"]) >= p_val.min() - 1e-6
    assert np.all(np.diff(c1.transform(np.sort(p_test))) >= -1e-12)


def test_platt_and_isotonic_validation():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 400)
    y = (rng.uniform(size=400) < p**2).astype(int)
    pl = Calibrator("platt").fit(p, y)
    d = pl.to_dict()
    assert np.isfinite(d["a"]) and np.isfinite(d["b"])
    out = pl.transform(p)
    assert np.all((out >= 0) & (out <= 1))
    assert brier(y, out) <= brier(y, p) + 1e-9
    with pytest.raises(ConfigError):
        Calibrator("temperature")
    with pytest.raises(ConfigError):
        Calibrator("isotonic").fit(p, np.zeros(400))
    with pytest.raises(RuntimeError):
        Calibrator("isotonic").transform(p)


def test_brier_and_reliability_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert brier(y, p) == pytest.approx((0.01 + 0.16 + 0.16 + 0.01) / 4)
    bins = reliability_bins(y, p, n_bins=5)
    assert len(bins) == 5 and sum(b["count"] for b in bins) == 4
    assert bins[0]["count"] == 1 and bins[0]["observed_rate"] == 0.0
    assert bins[1]["count"] == 0 and bins[1]["observed_rate"] is None and bins[1]["mean_predicted"] is None
    assert bins[4]["count"] == 1 and bins[4]["positives"] == 1
    with pytest.raises(ValueError):
        brier(y, np.array([0.1, 0.4, 0.6, 1.5]))


def test_auroc_and_ap_are_undefined_when_a_class_is_missing():
    assert auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert average_precision([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert auroc([0, 1], [0.2, 0.8]) == 1.0
    assert average_precision([0, 1], [0.2, 0.8]) == 1.0


def test_precision_and_recall_at_capacity():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    r = precision_recall_at_k(y, p, 3)
    assert r["k_effective"] == 3 and r["positives_in_top_k"] == 2
    assert r["precision_at_k"] == pytest.approx(2 / 3) and r["recall_at_k"] == pytest.approx(2 / 3)
    big = precision_recall_at_k(y, p, 50)
    assert big["k_effective"] == 6 and big["recall_at_k"] == 1.0
    zero = precision_recall_at_k(y, p, 0)
    assert zero["precision_at_k"] is None and zero["positives_in_top_k"] == 0
    assert precision_recall_at_k([0, 0], [0.2, 0.1], 1)["recall_at_k"] is None


def test_stability_by_segment_handles_single_class_and_missing_segments():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.2])
    seg = np.array(["a", "a", "a", "a", "b", "b"])
    out = stability_by_segment(y, p, seg, k=1)
    assert out["a"]["auroc"] == 1.0 and out["a"]["n"] == 4 and out["a"]["at_k"]["k_effective"] == 1
    assert out["b"]["auroc"] is None and out["b"]["average_precision"] is None and out["b"]["positives"] == 0
    assert "c" not in out
    with pytest.raises(ValueError):
        stability_by_segment(y, p, seg[:3])


def test_evaluation_output_lists_empty_bins_and_all_segments(trained):
    from customer_retention_lab import commands

    res = commands.run_evaluate(trained["out"] / "predictions_test.csv", capacity=50, n_bins=10)
    g = res["models"]["gbm"]["calibrated"]
    assert len(g["reliability"]) == 10 and sum(b["count"] for b in g["reliability"]) == g["n"]
    assert set(g["by_segment"]) == set(pd.read_csv(trained["out"] / "predictions_test.csv")["segment"].unique())
    assert g["at_capacity"]["k"] == 50
