# Model card: churn-risk baselines (customer-retention-lab)

## Intended use
Demonstration of a leakage-controlled churn-risk workflow on a synthetic B2B account
panel: point-in-time features, customer-aware temporal split, calibrated probability of
churn within the outcome window, and capacity-limited ranking. Evidence status of every
number produced here is `demo`.

## Not intended for
Any real customer decision without retraining on the user's own data; estimating the effect
of a retention offer (propensity is not uplift); populations whose churn definition differs
from "subscription ends inside the outcome window after the observation date".

## Training data
`examples/fixtures/v1/` (authored synthetic, seed 42, see FIXTURE.md). No real accounts.

## Models
Logistic regression and gradient boosting (scikit-learn HistGradientBoosting) on the
features defined in `docs/method.md`. Preprocessing is fitted on the train partition only.

## Calibration and thresholds
Isotonic (default) or Platt calibration fitted on the validation partition only
(`configs/model.yaml`). The review threshold is a capacity, not a probability cut:
the top-k accounts by calibrated risk, with k from `configs/capacity.yaml`.

## Evaluation
Test partition touched once by `evaluate`. Metrics: average precision, AUROC when both
classes are present, Brier score, reliability bins with counts, recall and precision at
capacity, and per-segment metrics with a minimum-count flag. Values are in
`examples/output/evaluation.json` after `demo`.

## Known limitations
Synthetic signal strength is an authoring choice; small segments are unstable; calibration
can drift if the deployment population differs from the validation partition.

## Provenance
Independent implementation, AI-assisted, reviewed before release (see NOTICE.md).
