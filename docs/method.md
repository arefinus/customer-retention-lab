# Method

## 1. Features (`features/builder.py`)

At observation date T, for each account at risk, from activity in `[T-5, T]` only:

| feature | definition |
|---|---|
| `tenure_months` | T - signup + 1 |
| `seats` | from the account table |
| `usage_mean`, `usage_last` | mean and last observed usage in the window |
| `usage_trend` | mean usage over the last 3 months minus mean over the first 3 months; a month with no row counts as zero |
| `tickets_sum`, `tickets_last` | support tickets in the window and in the last observed month |
| `active_months` | number of months with a row in the window (RFM frequency) |
| `recency_months` | T minus the last month with a row (RFM recency; window length when none) |
| `invoice_total` | sum of invoice amounts in the window (RFM monetary) |
| `paid_ratio` | share of invoices paid in the window |
| `plan_tier`, `region` | categorical, from the account table |

Label: 1 if `churn_month` falls in `(T, T+3]`, else 0. `segment` = `plan_tier`.

## 2. Split (`features/split.py`)

Accounts are hashed into train/validation/test groups (60/20/20). Observation dates are
cut into contiguous ranges (60/20/20, earliest first). A row is kept only when its
account's group equals its date's range. Consequences, all tested:

- no account appears in two partitions (customer-aware);
- every train date precedes every validation date, which precedes every test date
  (temporal);
- about 44% of rows survive (0.36 + 0.04 + 0.04); the rest are dropped and counted in
  `split_manifest.json`, never reassigned.

## 3. Baselines (`models/`)

Preprocessing (`Preprocessor`) standardises numerics with training means and standard
deviations and one-hot encodes categoricals with training category lists; an unseen
category becomes an all-zero block. Fitted statistics are plain numbers exported to
`models.json`. A test perturbs the test partition and checks the fitted statistics and
the transformed training matrix are unchanged.

- `logistic`: `LogisticRegression(C=1.0, max_iter=500)`.
- `gbm`: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
  max_leaf_nodes=15, min_samples_leaf=40, random_state=42)`.

Both are fitted on the training partition only.

## 4. Calibration (`models/calibration.py`)

A `Calibrator` (isotonic by default, Platt available) is fitted on validation-partition
predictions and labels only, then applied to validation and test predictions. Its
parameters are exported as thresholds or `(a, b)`. `models.json` records
`fitted_on: validation` and `n_fit`, and a test checks `n_fit` equals the validation row
count and not the test row count.

## 5. Metrics (`models/metrics.py`)

Per model, raw and calibrated, on the test partition:

- Brier score;
- AUROC, reported only when both classes are present, otherwise `null`;
- average precision, `null` when there are no positives;
- reliability bins: ten equal-width bins with `count`, `mean_predicted`, `observed_rate`
  and `positives`; empty bins are kept with `count: 0`;
- precision and recall at review capacity k (top-k by risk, ties by index);
- stability by segment: the same quantities per plan tier, with `null` where undefined.

`evaluate` recomputes all of this from a saved predictions file, so the protocol can be
re-run without refitting.

## 6. Capacity-limited ranking (`models/ranking.py`)

`rank_at_capacity` returns the top-k accounts per observation date by calibrated risk,
ties broken by `account_id`. The primary model for ranking is the calibrated `gbm`.

**Assumed-benefit scenario.** For each capacity on a grid, the expected number of
churners among reviewed accounts is the sum of their calibrated probabilities, and
`assumed_saves` multiplies that by `assumed_save_rate` from `configs/capacity.yaml`. The
output carries `causal_claim: "none"` and a note explaining that the save rate is an
input assumption. Nothing in this repository observes the outcome of a review, so the
scenario cannot and does not estimate the effect of reviewing an account. Alongside it,
`realised_at_capacity` reports label-based precision and recall at each capacity,
averaged over test observation dates.

## 7. Cohort retention (`models/cohorts.py`)

For each signup month, retention at offset k is the share of the cohort with
`churn_month` null or later than `signup + k`. Offsets beyond the panel end are not
reported, so recent cohorts have short curves rather than optimistic ones. The RFM
summary reports recency, frequency and monetary quartiles by segment.

## 8. Portfolio output

`examples/output/portfolio.json`:

```
{ "schemaVersion": 1, "mode": "synthetic-demo",
  "cohorts": [ { "cohort": "2024-01", "size": ..., "retention": [1.0, ...] }, ... ],
  "trajectories": [ { "account_id": "SYN-ACCT-...", "observation_date": "...", "rank": 1,
                      "risk": 0.xx, "segment": "tier-1", "months": [...], "usage": [...], "tickets": [...] }, ... ] }
```

Trajectories are the observation-window usage and ticket paths of the top-ranked
accounts at the latest test observation date.
