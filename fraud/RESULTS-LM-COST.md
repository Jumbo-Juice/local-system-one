# Results: training the language model on cost (Phase 25)

Eval `20261104-140349` (`python -m fraud.eval_lm_cost`). Pre-registration: the `fraud/eval_lm_cost.py` docstring, committed in `4fc8d48` before the adapter was trained.
Numbers: `fraud/results/eval_lm_cost/20261104-140349.json|.md`. Dev only: 88 held-out chains, 88,000 rows, 3,409 frauds, 275 of them 500 USD or more. No test row was used, because none is left. Model scores are uncalibrated. A review costs 10 USD. Lambda comes from all 118 dev windows except logreg's, so every read here is flattered.

The owner's constraint: stay on a decision model, with no tabular score in the prompt.

## Short answer

- **Training on cost didn't lower cost.** `lm-weighted` costs 2,165 USD per window against `lm-rich`'s 2,112: +53, CI -39 to 142, not distinguishable. Each was the cheaper in about half the windows, 46 to 42 in `lm-weighted`'s favour. I predicted about 100 lower, so the direction was wrong. (Observed)
- **It moved cost from good customers to fraud.** Per window, fraud let through rises 124 USD, wrong declines fall 61 and review fees fall 10. Good customers declined fall 12%, from 3,459 to 3,038. Frauds stopped fall 6%, from 2,324 to 2,194. (Observed)
- **The big frauds it was meant to catch were already caught.** The rule alone stopped 258 of the 275 frauds of 500 USD or more. Weighting makes that 259. (Observed)
- **The recipe stays `lm-rich`,** as the registration said it would if the primary wasn't distinguishable. (Implementation choice, stated before the run)
- **It doesn't beat logreg or the trees.** Against logreg it's +40, CI -76 to 157, not distinguishable. Against the trees it's +335, CI 219 to 450, distinguishable. (Observed)
- **This doesn't change what's known on test.** Nothing here touched a test row, and it doesn't choose what's deployed.

## Results (Observed)

| setup | mean cost | 95% CI | AUC | frauds stopped, of 3,409 | of 500 USD or more stopped, of 275 | fraud USD stopped | good customers declined | reviews |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| trees, reference | 1,830 | 1,693–1,982 | 0.9139 | 2,445 | 262 | 484,173 | 3,003 | 3,665 |
| **`lm-rich`** | **2,112** | 1,945–2,294 | 0.8932 | 2,324 | 258 | 465,486 | 3,459 | 3,855 |
| logreg | 2,125 | 1,980–2,277 | 0.8882 | 2,347 | 253 | 461,773 | 4,003 | 2,997 |
| **`lm-weighted`** | **2,165** | 2,013–2,323 | 0.8801 | 2,194 | 259 | 454,533 | 3,038 | 3,766 |
| `lm-rich`, customer line | 2,226 | 2,044–2,428 | 0.8932 | 1,996 | 253 | 442,092 | 945 | 4,196 |
| `lm-weighted`, customer line | 2,249 | 2,081–2,422 | 0.8801 | 1,895 | 257 | 435,964 | 783 | 4,122 |

All with the amount-aware rule. Two lines instead of the rule cost 2,972–3,093 for the same four scorers, in `fraud/results/eval_lm_cost/20261104-140349.md`.

Paired. "Distinguishable" means the 95% window-bootstrap CI excludes 0.

| `lm-weighted` against | windows `lm-weighted` cheaper / dearer | mean difference | 95% CI | |
|---|---:|---:|---|---|
| **`lm-rich` (primary)** | 46 / 42 | +53 | -39 to 142 | not distinguishable |
| **logreg (second)** | 42 / 46 | +40 | -76 to 157 | not distinguishable |
| **trees (third)** | 15 / 73 | +335 | 219 to 450 | distinguishable |
| `lm-rich`, both with the customer line, reported | 42 / 46 | +23 | -83 to 125 | not distinguishable |
| logreg, `lm-weighted` with the customer line, reported | 34 / 54 | +124 | -4 to 259 | not distinguishable |

## Before and after, in customers and dollars (Observed)

`lm-rich` against `lm-weighted`, 88 windows, both with the rule:

| | `lm-rich` | `lm-weighted` | change |
|---|---:|---:|---:|
| good customers declined | 3,459 | 3,038 | -421, -12% |
| good customers reviewed | 3,362 | 3,268 | -94 |
| frauds stopped, of 3,409 | 2,324 | 2,194 | -130, -6% |
| fraud USD stopped | 465,486 | 454,533 | -10,953, -2% |
| per window: fraud let through | 1,351 | 1,475 | +124 |
| per window: wrong declines | 323 | 262 | -61 |
| per window: review fees | 438 | 428 | -10 |
| **per window: total** | **2,112** | **2,165** | **+53** |

Frauds stopped by size, from the traces:

| fraud amount, USD | frauds | `lm-rich` | `lm-weighted` | fraud USD stopped, change |
|---|---:|---:|---:|---:|
| under 50 | 1,181 | 823 | 780 | -1,202 |
| 50 to 100 | 817 | 507 | 472 | -2,941 |
| 100 to 250 | 794 | 466 | 419 | -7,010 |
| 250 to 500 | 342 | 270 | 264 | -1,098 |
| 500 or more | 275 | 258 | 259 | +1,298 |

## Predictions against what happened

| stated before running (Inferred) | what happened (Observed) | |
|---|---|---|
| lower than `lm-rich` by about 100, not distinguishable | +53, not distinguishable | direction wrong |
| AUC falls by up to 0.01 | falls 0.0131 | right direction, more than stated |
| fraud dollars stopped rise | fall 10,953 | wrong |
| frauds of 500 USD or more stopped rise | 258 to 259 | right by one fraud, which is no real change |
| frauds stopped overall fall a little | fall 130, 6% | right |
| beats logreg by about 100, not distinguishable | +40, not distinguishable | direction wrong |
| stays dearer than the trees | +335, distinguishable | right |

## Why (Inferred)

- **The rule already does what the weighting was for.** It weighs each decision by its amount, so large frauds were already stopped 94% of the time. That left almost nothing for training to gain there, and the attention moved away from frauds under 250 USD, where it cost 130 frauds and about 11,000 USD. With two lines instead of the rule, which ignore the amount, weighting does stop more large frauds: 207 against 190. Those two read-outs fit this reading, but neither was a registered question and no CI was computed.
- **The adapter changed, and cost didn't.** Rank correlation with `lm-rich` is 0.836, so many rows were ordered differently. The +53 is about half of what a second seed moved on dev (about 100, Phase 21), so one adapter can't say more than that.

## Limits

- Dev isn't test, and these reads are flattered. `lm-rich`'s dev tie with logreg didn't hold on FINAL.
- One adapter and one weighting. The registration allowed no other, and none was tried.
- Every dollar depends on the assumed friction and review costs.

## Run (Observed)

Training 124 min, 5,994 steps, peak 4.06 GiB. Scoring 81 min for 118,000 dev rows. Latency on lone prompts: p50 53 ms, p90 65 ms, p99 80 ms, against the 250 ms bar, on this RTX 3060 Ti in bf16. No sleep or restart in the system log.

Checked apart from the report code: `python -m fraud.dev.verify_tabular fraud/results/eval_lm_cost/20261104-140349.json` replayed 10 setups over 88 windows from the raw traces with 0 rows decided differently, a largest gap of 0.03 USD per window and 0 customer counts different. Its own bootstrap gives the same verdicts, with bounds within 4 USD. Big-fraud counts and the cost split were recounted from the traces by a separate script, and they match.
