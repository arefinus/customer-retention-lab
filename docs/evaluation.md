# Evaluation

## Protocol
Customer-aware temporal split (60/20/20 by hashed account group paired with contiguous date
ranges). Models fitted on train, calibration fitted on validation, test touched once.
Metrics: average precision, AUROC (when defined), Brier, reliability bins with counts,
recall and precision at review capacity, per-segment stability.

## Where the numbers are
`examples/output/evaluation.json` (written by `demo`) and `examples/output/smoke/`
(written by `smoke`). Every value there was computed on synthetic fixture v1, seed 42, and
is labelled `demo`. It is not a benchmark result and is not reproduced from any paper.

## Local test run (2026-09-27)
```
python -m pytest -q
42 passed, 2 warnings in 24.36s
```

## Leakage tests that must stay green
- perturbing rows inside the outcome window leaves every feature unchanged;
- no account id appears in two partitions;
- every train observation date precedes every validation date, which precedes every test date;
- censored observation dates (outcome window beyond the panel end) are refused.
