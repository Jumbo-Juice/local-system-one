# Results: decisions that weigh the amount, IEEE-CIS (Phase 14)

Eval `20261103-164830` (`python -m fraud.eval_ieee_amount --adapter ...`). Pre-registration: the `fraud/eval_ieee_amount.py` docstring, committed in `8a0facf` before any model or policy touched a new window.
Numbers: `fraud/results/eval_ieee_amount/20261103-164830.json|.md`. Model scores are uncalibrated. A review costs 10 USD. Nothing was retrained.

## Short answer

- **Weighing the amount lowers logistic regression's cost by 30%.** On the same scores, 2,752 → 1,926 USD per 1,000-row window: cheaper in 18 of 20 windows, mean paired difference -825 USD, 95% CI -1,307 to -464. Distinguishable. (Observed)
- **The same rule on the adapter is not a shown saving.** 2,734 → 2,337: cheaper in 12 of 20, -397 USD, CI -937 to 14. Not distinguishable. (Observed)
- **Logreg with the rule is now the cheapest shown option, and it beats the adapter with the rule.** 1,926 vs 2,337: cheaper in 17 of 20, -411 USD, CI -709 to -158. Distinguishable. (Observed)
- **Blending the adapter in adds nothing.** 1,925 vs 1,926: -2 USD, CI -69 to 67, 10 windows each way. (Observed)
- **It declines many more good customers.** 982 against 655, 50% more, at a median of 75 USD against 100. The cost falls because the model prices a wrong decline at 10% of the amount. A business that counts declined customers rather than dollars may not call this better. (Observed; Inferred on what it means)

## The rule (Implementation choice, pre-registered)

For each row, on that row alone: p from a logistic fit of the score's logit on dev, used only to weigh costs and not claimed as a probability. Expected cost of approve p × amount, of decline (1 − p) × friction, of review the fee + lambda. Take the lowest. Past the budget, a review becomes the cheaper of approve and decline.
Lambda, the price of using up a review, is picked on dev. Costs as Phase 11: friction 10%, at least 1 USD; review 10 USD; at most 5% of a window reviewed.

Two things follow from the arithmetic, which explain most of the results below (Inferred):

- From 10 USD up, friction is 10% of the amount, so the amount cancels and approve vs decline is one line: p above 1/11. Only under 10 USD does the amount move that line.
- A review needs both expected costs above fee + lambda. With lambda 10, that means p × amount and (1 − p) × 10% of the amount are both above 20 USD, so nothing under 200 USD can be reviewed. The smallest reviewed payment was 221 USD. (Observed)

## Setup (Implementation choice unless cited)

- New test windows: seed 20261104, 20 windows of 1,000 contiguous rows, 20,000 rows, 646 fraud, natural rate. They share no row with Phase 11's, Phase 12's or dev. (Observed from the traces)
- Dev: Phase 11's 10 windows and 20 new ones, 30,000 rows, 1,197 fraud.
- Scorers, frozen: `ieee-logreg-all`, refitted, reproduced Phase 11's dev scores as required. Phase 12's adapter (sha256 `10970d702ee9990f…`, checked), order `rev`.
- Fits on dev: logreg lambda 10, adapter lambda 5, blend lambda 10. Score-only setups kept Phase 12's thresholds. (Observed)

## Results (Observed)

| setup | mean cost | 95% CI | AUC | recall | precision | alert rate | rule |
|---|---:|---|---:|---:|---:|---:|---|
| blend, amount-aware | 1,925 | 1,697–2,172 | 0.868 | 0.636 | 0.201 | 0.102 | lambda 10 |
| **logreg all columns, amount-aware** | **1,926** | 1,688–2,185 | 0.869 | 0.627 | 0.200 | 0.101 | lambda 10 |
| adapter, amount-aware | 2,337 | 2,050–2,666 | 0.840 | 0.588 | 0.154 | 0.123 | lambda 5 |
| adapter, thresholds | 2,734 | 2,269–3,254 | 0.840 | 0.551 | 0.203 | 0.087 | 0.472, 0.764 |
| logreg, thresholds from 30 dev windows | 2,744 | 2,361–3,188 | 0.869 | 0.630 | 0.213 | 0.096 | 0.060, 0.113 |
| logreg all columns, thresholds | 2,752 | 2,364–3,202 | 0.869 | 0.627 | 0.209 | 0.097 | 0.059, 0.114 |
| adapter, thresholds from 30 dev windows | 2,757 | 2,291–3,281 | 0.840 | 0.560 | 0.202 | 0.090 | 0.468, 0.731 |
| approve all | 4,660 | 3,826–5,493 | n/a | 0.000 | 0.000 | 0.000 | n/a |

Paired comparisons. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer / tied | mean difference | 95% CI | sign test p | |
|---|---|---:|---:|---|---:|---|
| **logreg, amount-aware** | **logreg, thresholds (primary)** | 18 / 2 / 0 | -825 | -1,307 to -464 | 0.0004 | distinguishable |
| **adapter, amount-aware** | **adapter, thresholds (second)** | 12 / 8 / 0 | -397 | -937 to 14 | 0.50 | not distinguishable |
| **blend** | **logreg, amount-aware (third)** | 10 / 10 / 0 | -2 | -69 to 67 | 1 | not distinguishable |
| **logreg, amount-aware** | **adapter, amount-aware (fourth)** | 17 / 3 / 0 | -411 | -709 to -158 | 0.003 | distinguishable |
| logreg, 30 dev windows | logreg, thresholds | 11 / 5 / 4 | -7 | -80 to 78 | 0.21 | not distinguishable |
| adapter, thresholds | adapter, 30 dev windows | 14 / 5 / 1 | -23 | -61 to 17 | 0.06 | not distinguishable |
| logreg, amount-aware | approve all | 20 / 0 / 0 | -2,733 | -3,540 to -1,979 | 0.000002 | distinguishable |
| adapter, amount-aware | approve all | 19 / 1 / 0 | -2,323 | -3,216 to -1,421 | 0.00004 | distinguishable |

Only the first four rows are pre-registered questions. The others are not corrected for multiple comparisons. Not pre-registered and worked out afterwards: adapter vs logreg on thresholds, 2,734 vs 2,752, -18 USD, CI -345 to 354, 12 windows to 8. Not distinguishable, though it is the first lower mean for a language model against logreg on all columns.

Where the money goes, total USD over the 20 windows:

| setup | fraud let through | good customers declined | review fees | total |
|---|---:|---:|---:|---:|
| logreg, thresholds | 32,452 | 13,140 | 9,440 | 55,032 |
| logreg, amount-aware | 22,541 | 8,968 | 7,020 | 38,529 |
| adapter, thresholds | 39,059 | 6,203 | 9,410 | 48,672 |
| adapter, amount-aware | 25,344 | 12,177 | 9,220 | 46,741 |
| blend, amount-aware | 22,097 | 9,356 | 7,040 | 38,493 |
| approve all | 93,196 | 0 | 0 | 93,196 |

What the rule changed, counted from the traces:

| | logreg, thresholds | logreg, amount-aware | adapter, thresholds | adapter, amount-aware |
|---|---:|---:|---:|---:|
| Frauds stopped, of 646 | 405 | 405 | 356 | 380 |
| Fraud amount stopped, of 93,196 | 60,744 | 70,655 | 54,137 | 67,852 |
| Largest fraud let through | 2,161 | 680 | 2,161 | 600 |
| Reviews done, of 1,000 possible | 944 | 702 | 941 | 922 |
| Median amount reviewed | 82 | 664 | 100 | 527 |
| Reviews that found fraud | 69 | 61 | 102 | 66 |
| Fraud amount those reviews stopped | 12,975 | 37,834 | 22,152 | 37,048 |
| Windows that ran out of reviews | 13 | 2 | 10 | 11 |
| Reviews wanted past the budget | 189 | 12 | 98 | 188 |
| Declines | 991 | 1,326 | 809 | 1,548 |
| Good customers declined | 655 | 982 | 555 | 1,234 |
| Share of declines that were good customers | 66% | 74% | 69% | 80% |

Logreg by amount, the same 20 windows:

| amount, USD | rows | fraud | cost, thresholds | cost, amount-aware | reviews, before → after | declines, before → after |
|---:|---:|---:|---:|---:|---:|---:|
| under 10 | 285 | 30 | 226 | 57 | 18 → 0 | 47 → 44 |
| 10–50 | 6,100 | 176 | 5,683 | 3,478 | 278 → 0 | 235 → 401 |
| 50–100 | 5,376 | 174 | 7,303 | 6,402 | 224 → 0 | 234 → 382 |
| 100–200 | 4,837 | 134 | 11,440 | 11,072 | 214 → 0 | 218 → 367 |
| 200–500 | 2,569 | 92 | 13,016 | 10,858 | 132 → 231 | 172 → 126 |
| 500 and up | 833 | 40 | 17,363 | 6,662 | 78 → 471 | 85 → 6 |

Cost sensitivity, no bar and no CI; lines or lambda re-picked on dev from the same scores (the 10 USD row uses the eval's choices):

| cost set | logreg, thresholds | adapter, thresholds | logreg, amount-aware | adapter, amount-aware | blend |
|---|---:|---:|---:|---:|---:|
| review 1 USD | 2,327 | 2,591 | 1,636 | 2,032 | 1,543 |
| review 5 USD | 2,516 | 2,791 | 1,671 | 2,107 | 1,708 |
| **review 10 USD** | 2,752 | 2,734 | 1,926 | 2,337 | 1,925 |
| review 20 USD | 3,038 | 3,082 | 2,247 | 2,612 | 2,252 |

| | value |
|---|---|
| Latency bar, p90 ≤ 250 ms, adapter | PASS: p50 113, p90 128, p99 149 ms. Slowest test window 144 ms, slowest extra dev window 187 ms |
| Dev cost at the chosen lambda, 30 windows, fitted on the same windows | blend 73,843; adapter 78,681; logreg 79,236 |
| Logreg's lambda on dev | 10 by a hair: 79,236 against 79,270 at lambda 5 |
| Sum of p on test against the 646 frauds | logreg 790; adapter 902. Dev's fraud rate was 3.99%, test's 3.23% |
| Run time | 81 min, 16:48 to 18:09; about 119 s per adapter window |

Checked apart from the report code: a separate script replayed every decision and the review budget from the traces, and recomputed the amount-aware choices from the recorded fit and scores. 0 of 60,000 rows differ, the largest cost gap per window is 0.08 USD (rounding), and a refit on the dev traces alone gives the recorded fit and lambdas. A separate bootstrap with another seed gives -1,297 to -468 on the primary. (Observed)
The score-only setups' scores are identical to the amount-aware setups', and the as-they-are thresholds are Phase 12's. The new traces list and replay in the console with no errors. (Observed) The one count that differs is a near-tie: the replay reads the 30-dev-window logreg row as 6 dearer and 3 tied, because trace amounts are rounded.

## What it means (Inferred)

- **The saving is mostly large payments moving to review.** 65% of logreg's saving, 10,701 of 16,504 USD, comes from the 833 rows of 500 USD and up, 4% of the rows. Their reviews rose from 78 to 471 and their declines fell from 85 to 6.
- **It stops the same number of frauds but more of the money.** 405 of 646 both ways, with 9,911 USD more stopped. Reviews found fewer frauds (69 → 61) worth almost three times as much. By source, the saving is 60% less fraud let through, 25% less friction, 15% fewer fees.
- **Using fewer reviews helped.** Reviews fell from 944 to 702 and the budget stopped running out in most windows. The old rule spent reviews on whatever arrived first, including 18 payments under 10 USD where a 10 USD review costs more than the payment.
- **Small payments now get a yes or a no, with no human look.** No payment under 221 USD was reviewed. Payments of 10 to 200 USD lost all 716 of their reviews and gained 463 declines. This is cheaper in this model, and it is also where the extra good customers declined come from.
- **The adapter gains less because it ranks worse.** AUC 0.840 against 0.869. Its p also overshoots further, 902 against 646 frauds, and it declines 1,234 good customers, 80% of its declines. That may be why the rule pays less there, but the two weren't separated.
- **More dev windows alone did little.** Thresholds barely moved, and differences of -7 and +23 USD are inside the noise. The gain comes from the rule, not from more data to pick lines.
- **The dev ranking did not carry over.** On dev the blend led the other two by about 4,800 and 5,400 USD, and the adapter was just ahead of logreg. On test the blend tied logreg and the adapter was distinguishably dearer. So dev is not a good guide to which scorer wins here, though it did pick the right rule.
- **The prediction made before running held.** Both amount-aware setups have lower means, by less than the held-out dev gap (logreg -825 against -1,107 per window; adapter -397 against -1,498). At least one is distinguishable. The blend doesn't beat logreg. More dev windows change little.

## Negative results

- **The adapter's saving is not distinguishable** on the second question, and 8 of 20 windows got dearer. (Observed)
- **The language model loses the fourth question.** With the same rule, logreg is distinguishably cheaper, and it needs no GPU. No language model has a shown cost win against logreg on all columns. (Observed)
- **The blend gives no shown gain** and would need the adapter's latency for it. It is lower at a 1 USD review (1,543 against 1,636), with no CI. (Observed)
- **Good customers declined rise 50% with logreg, and 122% with the adapter** (555 → 1,234). Precision falls for both, from 0.209 to 0.200 and from 0.203 to 0.154. (Observed)
- **The saving leans on a few rows.** 1,388 rows changed cost, and the 10 biggest savings make up 61% of the total; without them it is 6,417 USD, about 321 per window. One window, `amt4`, saved 4,374. Without it the mean is -638, and without the top three -508; the median is -490. So the direction holds, but -825 may be high. (Observed; not pre-registered)
- **p overshoots on test.** The fitted p adds up to 790 and 902 against 646 frauds. Dev had more fraud than these windows, so this fit may be off on another period. (Observed)
- **More dev windows gave no shown gain.** (Observed)

## Limits

- **The result follows the assumed costs.** The rule optimises the same cost model it is scored on, so it exploits the 10% friction and the flat fee directly. If a wrongly declined 30 USD payment really costs more than 3 USD, the extra declines would count against it, and how much isn't measured here. (Inferred) The sensitivity table varies the fee only, not friction.
- **Reviews are assumed to always find the truth at a flat fee.** The rule now sends payments of up to 4,607 USD to review. If those take longer or cost more, part of this saving goes. At a 20 USD fee logreg with the rule still costs 2,247 against 3,038. (Implementation choice; Observed, no CI)
- **New rows, same period.** The windows share no row with earlier ones but come from the same stretch of time, so this is not a later period. These windows hold fewer frauds than Phase 12's (646 against 778; approve-all 4,660 against 5,283), so don't compare across phases.
- **USD amounts and the meaning of the C, D, M and V columns are from memory of the competition's data page, not verified.** Whether they are known at decision time can't be checked. If some aren't, logreg's numbers here are optimistic. (Inferred)
- **One test, 20 windows, 646 frauds, and a heavy tail.** Approve-all ranges from 1,129 to 7,926 per window.
- **The past-budget fallback differs between the two rules, on purpose**, as it is part of the new rule. It touched 12 rows for logreg and 188 for the adapter, 73 of those declines. (Observed)
- **The rule isn't visible in the console.** Traces hold it in the header as `policy`, with `thresholds` empty. The run page shows each decision but neither the rule nor lambda, and the Eval page doesn't show this phase. (Observed)
- **Latency depends on the state of the laptop.** p90 128 ms here against 113 in Phase 12, both awake, on different windows. Phase 12's scoring ran at Above Normal priority and this one did not. Four dev windows in a row rose to 154–187 ms and came back; the cause wasn't measured. (Observed)
- **The screen was held on for the whole run.** No sleep, standby or restart event was logged from 16:48 to the end. Scores are unaffected either way. (Observed)

## Not run

- Friction varied in the sensitivity table, or a cost for each good customer declined.
- The fitted rule on Phase 11's and Phase 12's windows, which were not part of the pre-registration.
- Retraining, more prompt columns, per-card history, a later period, or frozen d1-Omni-2B on these windows.
- Fine-tuned d1-Omni on PaySim's 10M burst hours (owner: not now).
