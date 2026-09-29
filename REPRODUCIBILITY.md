# Reproducibility

Release date: 2026-09-27. Python 3.10 locally; CI runs 3.10, 3.11 and 3.12.

## Exact commands

```
python -m pip install -e .
python -m pip install pytest

python tools/make_fixture.py --seed 42            # regenerates examples/fixtures/v1/
python -m customer_retention_lab smoke            # examples/output/smoke/ (gitignored), about 10 s
python -m customer_retention_lab demo             # examples/output/ + docs/figures/
python -m customer_retention_lab build-features   # features_panel.csv + split_manifest.json
python -m customer_retention_lab train            # predictions_{validation,test}.csv + models.json
python -m customer_retention_lab evaluate --predictions examples/output/predictions_test.csv
python -m customer_retention_lab rank --capacity 150
python -m pytest -q
```

## Seeds and determinism

| Component | Seed | Notes |
|---|---|---|
| Fixture | 42 | `numpy.random.default_rng(42)`; a test compares the committed accounts file with a fresh build |
| Account partition | 42 (`configs/model.yaml`) | SHA-256 of `"{seed}:{account_id}"`, so the partition of an account never depends on which other accounts exist |
| Logistic regression | 42 | lbfgs is deterministic; seed passed for completeness |
| HistGradientBoosting | 42 | deterministic for a seed; a test fits twice and compares predictions exactly |
| Calibrator | none | isotonic regression and the Platt fit are deterministic given validation predictions |
| Ranking | none | ties broken by `account_id` |

## What is reproduced

Nothing. There is no source paper and no real dataset run. Every number in
`examples/output/`, in `MODEL_CARD.md` and in the docs carries the evidence label
`demo` with the sentence "computed by `make demo` on synthetic fixture v1, seed 42, not
a benchmark result".

## Model artifacts

No estimator is pickled. `models.json` holds the logistic regression coefficients and
intercept, the preprocessor's fitted means, standard deviations and category lists, the
boosting settings, and the calibrator's thresholds (isotonic) or `a, b` (Platt).
`evaluate` and `rank` consume the saved prediction files, so re-scoring the test
partition does not require the fitted boosting model.

## What can drift between machines

- Logistic coefficients can differ in the last digits across BLAS builds; tests check
  behaviour, not exact coefficient values.
- HistGradientBoosting is deterministic for a given seed and sklearn version. A different
  sklearn minor version may change results slightly.
- Timestamps and `run_id` in `run_manifest.yaml` change on every run by design.
