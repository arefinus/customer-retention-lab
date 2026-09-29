# Data

See `DATA_CARD.md` for the full card. This page covers how the code treats the panel.

## Two tables

| | accounts.csv | activity.csv |
|---|---|---|
| Grain | one fictional account | one account-month while active |
| Loader | `io.load_accounts` | `io.load_activity` |
| Derived columns | `signup_idx`, `churn_idx` (integer month counts, `churn_idx` nullable) | `month_idx` |
| Used for | eligibility (at risk at T), label, tenure, `plan_tier`, `region`, `seats` | every activity-based feature |

Months are integers counted from 2000-01 (`io.month_index`), which keeps window
arithmetic to plain comparisons.

## Point-in-time contract

For observation date T:

```
observation window   [T - 5, T]      features read only rows with month_idx in this range
outcome window       (T, T + 3]      label = churn_idx in this range
at risk at T         signup_idx <= T and (churn_idx is null or churn_idx > T)
eligible T           T + 3 <= panel_end (otherwise CensoringError)
```

The activity frame is filtered to the observation window before any aggregation, so no
code path can read the outcome window when building a feature.
`tests/test_features.py::test_features_ignore_outcome_window` corrupts every row after T
and asserts identical features; a second test corrupts rows before the window.

## Observation dates in the demo

`configs/features.yaml` requests monthly dates 2024-06 to 2025-09 (16 dates, all
eligible; 2025-10 to 2025-12 would be censored and are refused). The split assigns the
first ten dates to train, the next three to validation and the last three to test.

## Schema validation

Loaders fail fast, naming the missing column and the command that regenerates the
fixture. Duplicate account ids, negative usage or tickets, and malformed months raise
`SchemaError`. A predictions file passed to `evaluate` or `rank` must carry every
column written by `train`.

## Regeneration

```
python tools/make_fixture.py --seed 42
```

`tests/test_fixture.py` checks determinism, the `SYN-ACCT-` prefix, that no activity row
exists at or after an account's churn month, and that the committed accounts file equals
a fresh build.
