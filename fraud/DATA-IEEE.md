# IEEE-CIS Fraud Detection (Vesta): what the data looks like

Files: `fraud/data/ieee-cis/*.csv` (git-ignored, downloaded by the owner). Cache: `python -m fraud.ieee_data` → `fraud/data/ieee-cis/ieee.npz`.
Labels below come from the fit rows only. Test-split labels are read only by `fraud/eval_ieee.py`.

## Shape (Observed)

| | |
|---|---|
| `train_transaction.csv` | 590,540 rows, 394 columns, in `TransactionDT` order |
| `train_identity.csv` | 144,233 rows, 41 columns, joined on `TransactionID`; 27.5% of fit rows have one |
| Test files | no `isFraud` column, so never read |
| `TransactionDT` | seconds, 86,400 to about 182 days later; a relative clock, not a timestamp |

## Split (Implementation choice, by row count, before any label was read)

| split | rows | days | fraud | fraud rate | fraud USD per 1,000 rows |
|---|---:|---|---:|---:|---:|
| fit | 354,324 | 0–100 | 11,988 | 3.38% | 4,735 |
| dev | 118,108 | 100–140 | 4,611 | 3.90% | 6,741 |
| test | 118,108 | 140–182 | not read here | | |

## Amounts (Observed, fit)

Median 70, mean 134, 99th percentile 1,092. Fraud median 77, mean 140. Amounts are taken as USD: from memory of the competition's data page, **not verified**.

## Where fraud is more common (Observed, fit; the prompt's context states these)

| field | rates |
|---|---|
| ProductCD | C 10.7%, S 6.0%, H 4.3%, R 3.2%, W 2.0% |
| card6 | credit 6.3%, debit 2.3% |
| R_emaildomain | missing 2.1%; gmail.com 10.3%, outlook.com 13.7% |
| DeviceType | none 2.1%, desktop 5.4%, mobile 9.1% |
| M4 | M2 10.4%, M0 3.7%, M1 2.6%, missing 1.9% |
| repeating hour | 4:00–10:00 about 5–10% on low volume; otherwise 2–4% |
| single numeric columns (rank AUC) | C4 0.677, C8 0.668, C1 0.631, C13 0.400, D1 0.430 |

## Differences from PaySim that matter

- **Natural fraud is about 3.5%, so no enrichment.** A 1,000-row window holds about 35–40 frauds. All numbers here are natural-rate.
- **Every product has fraud**, so nothing can be filtered by type. There are no account pairs and no same-amount cash-out.
- **Amounts are small against a review.** PaySim's review fee of 50 is above the median payment here. `fraud/eval_ieee.py` records the cost choice.
- **Most columns are anonymised.** From memory of the data page, not verified: C = counts, D = time deltas, M = match flags, V = Vesta-engineered.
  Whether all of them are known at decision time can't be checked here (limit stated in the pre-registration).
- **No look-ahead:** prior `isFraud` labels are never an input; labels arrive later.
