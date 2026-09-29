# Data card

## Synthetic B2B account panel (the only data in this repository)

Location: `examples/fixtures/v1/`, described in `FIXTURE.md`. Authored synthetic
demonstration data, fictional identifiers, not derived from any person, company, account
or dataset. Evidence label for anything computed on it: `demo`. No real customer data
is shipped, read or required.

| Item | Value |
|---|---|
| Provider | this repository, `tools/make_fixture.py`, seed 42, version v1 |
| Terms | MIT (same as the code) |
| Record unit | `accounts.csv`: one fictional account; `activity.csv`: one account-month while active |
| Rows | 3,000 accounts; 36,026 account-months (2024-01 to 2025-12) |
| Identifiers | `SYN-ACCT-00001` to `SYN-ACCT-03000`; plan tiers `tier-1/2/3`; regions `region-A..D` |
| Target | churn within the 3-month outcome window after an observation date T, derived from `churn_month` (1,154 accounts churn inside the panel) |
| Account columns | `account_id, signup_month, plan_tier, region, seats, churn_month` (empty when not churned inside the panel) |
| Activity columns | `account_id, month, usage, tickets, invoice_amount, invoice_paid` |
| Missingness | none in the files; an at-risk account with no rows in an observation window receives explicit defaults (zero activity, maximal recency) |

### Generating mechanism

Usage follows a per-account random walk in log space with a per-account drift. A latent
monthly hazard depends on usage decline over the previous six months, recent tickets,
plan tier, a missed payment in the previous month, early tenure, and usage falling below
half the starting level. The hazard reads only months before the churn month. An account
with `churn_month = M` has activity rows for `signup .. M-1` and none afterwards.

Hazard coefficients were chosen so that the observation-window features carry a moderate
signal. The churned share and every metric computed here are properties of the
generator, not of any market.

### Splits

No split is stored in the fixture. `features/split.py` builds a customer-aware temporal
split at run time and writes `examples/output/split_manifest.json`:

- accounts partitioned 60/20/20 into train/validation/test by a seeded hash of the id;
- monthly observation dates 2024-06 to 2025-09 cut 60/20/20 into contiguous ranges,
  earliest for train and latest for test;
- a row is kept only when its account group matches its date range; the rest are
  dropped and counted, never moved.

### Censoring

An observation date T is eligible only when `T + 3 months <= 2025-12`, so every kept
row has a fully observed outcome window. Accounts that churned at or before T, or signed
up after T, are not at risk and are excluded. Cohort retention curves stop at the last
offset every cohort has reached.

## Adapting to real data

A user with their own account panel can point `build-features`, `train` and the other
commands at `--accounts` and `--activity` CSV files with the columns above. The loaders
validate the schema and refuse negative usage or ticket counts, duplicate account ids and
malformed `YYYY-MM` months. Nothing in this repository has been run on real data, and
no claim in the docs applies beyond the fixture.
