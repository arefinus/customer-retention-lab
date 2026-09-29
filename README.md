# customer-retention-lab

A customer analytics team wants to know which accounts show patterns worth a retention
conversation, and how many of them the team can actually handle. This repository builds
leakage-controlled churn-risk features from a synthetic B2B account panel, trains and
calibrates baseline models, and ranks accounts under an explicit review capacity.

**Status:** `demo_ready` (runs end to end on authored synthetic fixtures; no real-data
evaluation performed in this repository). Label: `original_project`.

## Problem and user

The user is a customer analytics team at a fictional US subscription business. The
decision supported is which accounts to route to a limited retention team in a given
month. Inputs are monthly account activity (usage, tickets, invoices) and a subscription
table. The output is a calibrated probability of churn inside a fixed future window, a
cohort retention table, and a ranked review list cut at capacity. Nothing here estimates
the causal effect of a retention offer; see Limitations.

## Demo

```
python -m customer_retention_lab smoke     # fast end-to-end run on the fixture
python -m customer_retention_lab demo      # full run, writes examples/output/
```

`demo` writes the feature panel, split manifest, validation and test predictions,
evaluation metrics with reliability bins and their counts, cohort retention tables, an
RFM summary, the ranked review list at capacity, a capacity scenario file, figures under
`docs/figures/`, `portfolio.json` for the owner's portfolio site, and a run manifest. Every
number in those files was computed by that run on the synthetic fixture and is labelled
`demo`.

## Implementation status

Implemented and tested: point-in-time feature builder with disjoint observation and
outcome windows and censoring refusal; customer-aware temporal split; cohort retention;
RFM-style summary; logistic and gradient boosting churn baselines; isotonic or Platt
calibration fitted on validation only; average precision, AUROC (when both classes are
present), Brier score, reliability bins with counts, recall and precision at capacity,
per-segment stability; capacity-limited ranking with a documented assumed-benefit model.

Not implemented (`proposed`): a real-data adapter beyond the CSV schema in `DATA_CARD.md`,
survival models, uplift or treatment-effect estimation.

## Quickstart (offline, CPU, under a minute)

```
pip install -e . && pip install pytest
python -m customer_retention_lab smoke
python -m pytest -q
```

Without an editable install, prefix commands with `PYTHONPATH=src`.

## Data

Only the authored synthetic fixture in `examples/fixtures/v1/` ships with the repository
(see `FIXTURE.md` there and `DATA_CARD.md`). Real account data must be supplied by the user
in the documented two-table CSV schema; the loader validates it and refuses anything else.

## Architecture / method

`docs/method.md` defines every feature, the split, the models, calibration, metrics, the
ranking rule and the cohort computation. The one design rule that matters most: a feature
at observation date T may only use activity in the observation window closing at T, and
the label is defined strictly after T. A test proves that perturbing the outcome window
does not change any feature.

## Train, validate, test

`build-features` creates the panel; `train` fits the models on the train partition and
fits calibration on validation; `evaluate` touches the test partition once and writes
metrics; `rank` applies the capacity rule to the test predictions. Accounts never appear in
two partitions and every train date precedes every validation date, which precedes every
test date.

## Results and provenance

Metrics live in `examples/output/evaluation.json` after `demo`, with `evidence: "demo"` and
the fixture hash in `run_manifest.yaml`. There is no `reproduced` or `paper-reported`
number in this repository. `docs/evaluation.md` carries the local test summary line.

## Limitations

Synthetic data; a churn probability is a propensity, not a treatment effect; the
assumed-benefit model in the capacity scenario is a stated assumption, not an estimate;
calibration is only as good as the validation partition it was fitted on; segments with
few accounts produce unstable per-segment metrics and are flagged as such.

## License and sources

MIT, copyright 2026 Md Sultanul Arefin Sourav. See `NOTICE.md` for provenance. No
third-party data or code is redistributed.
