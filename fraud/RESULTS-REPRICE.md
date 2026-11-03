# Results: PaySim's older results at a 10 USD review (Phase 13)

Run `20261103-162503` (`python -m fraud.reprice_paysim`). Pre-registration: the `fraud/reprice_paysim.py` docstring, committed before anything was priced at 10 USD. Numbers: `fraud/results/reprice_paysim/20261103-162503.json|.md`.
No model ran and no trace changed. The older results and write-ups stay at 50 USD, as pre-registered. Model scores are uncalibrated.

## Short answer

- **Nothing changes rank or verdict.** Every recorded bar would read the same on the same side of its line, and the order of setups is the same. (Observed)
- **The chosen thresholds did not move at all.** Every setup re-picked the same review and decline lines on dev, so the two pre-registered views are identical: same decisions and re-picked thresholds give the same numbers. (Observed)
- **The setups that pay almost only review fees fall to a fifth.** On the 20 windows of Phases 5–9, rules goes from 808 to 162 per window, d1-Omni fine-tuned from 810 to 162, Qwen fine-tuned from 830 to 166, and d3-nano fine-tuned from 835 to 167. (Observed)
- **Everything else barely moves**, because a fee this small is lost beside PaySim's amounts: logreg 190,910 → 190,432, frozen Qwen 5,146,777 unchanged, approve-all 42,190,884 unchanged. (Observed)

## Setup (Implementation choice unless cited)

- Costs: `costs.DEFAULT`. Only the review fee changes, 50 → 10. Friction stays 10% of the amount, at least 10, and the budget stays 5% of a window. IEEE-CIS's friction minimum of 1 USD is not brought over, so PaySim and IEEE-CIS costs still aren't on one scale.
- Re-priced: test windows of Phases 4, 5, 7 and 9, and dev totals of Phases 6 and 8. Not re-priced: Phase 10 (scored by F1, no USD) and `reviewer_check` (exploratory, dev).
- Gate before pricing (Observed): at 50 USD the code reproduced all 260 committed test windows to 0.01 USD, every recorded action, and the committed thresholds and best dev costs.
- Checked apart from the script's scoring (Observed): committed cost minus 40 USD per committed review matches every setup's mean and total exactly. No trace file changed.

## Results (Observed)

The 20 windows shared by Phases 5, 7 and 9 (`test200`–`test233`, 500 rows each, enriched to about 5% fraud):

| setup | mean at 50 | mean at 10 | median at 10 | natural-rate at 10 | recall | precision | alert rate | reviews |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 808 | **162** | 160 | 1,414 | 0.996 | 0.868 | 0.060 | 323 |
| d1-Omni-2B fine-tuned | 810 | **162** | 160 | 1,415 | 1.000 | 0.868 | 0.060 | 324 |
| Qwen2.5-1.5B fine-tuned | 830 | **166** | 165 | 1,427 | 0.996 | 0.868 | 0.060 | 332 |
| d3-nano fine-tuned | 835 | **167** | 165 | 1,428 | 1.000 | 0.868 | 0.060 | 334 |
| logreg | 190,910 | 190,432 | 166,689 | 445,907 | 0.886 | 0.852 | 0.056 | 239 |
| frozen Qwen2.5-1.5B | 5,146,777 | 5,146,777 | 4,988,782 | 207,111,353 | 1.000 | 0.143 | 0.386 | 0 |
| random | 19,605,342 | 19,604,342 | 14,210,557 | 157,998,772 | 0.583 | 0.142 | 0.222 | 500 |
| approve all | 42,190,884 | 42,190,884 | 34,751,285 | 102,490,105 | 0.000 | 0.000 | 0.000 | 0 |

The 20 windows of Phase 4 (`test100`–`test119`):

| setup | mean at 50 | mean at 10 | median at 50 | median at 10 | natural-rate at 10 | recall | precision | reviews |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 400,845 | 400,169 | 825 | **165** | 43,910,046 | 1.000 | 0.857 | 338 |
| logreg | 549,247 | 548,707 | 134,044 | 133,584 | 44,254,658 | 0.913 | 0.797 | 270 |
| frozen Qwen2.5-1.5B | 6,373,694 | 6,373,694 | 5,404,353 | 5,404,353 | 343,756,836 | 1.000 | 0.147 | 0 |
| random | 17,877,963 | 17,876,963 | 14,374,362 | 14,373,362 | 163,625,121 | 0.537 | 0.141 | 500 |
| approve all | 31,735,048 | 31,735,048 | 27,654,563 | 27,654,563 | 46,989,968 | 0.000 | 0.000 | 0 |

Where the money goes on `test200`–`test233`, total USD over the 20 windows at 10 USD:

| setup | fraud let through | good customers declined | review fees | total |
|---|---:|---:|---:|---:|
| rules | 0 | 0 | 3,230 | 3,230 |
| d1-Omni-2B fine-tuned | 0 | 0 | 3,240 | 3,240 |
| Qwen2.5-1.5B fine-tuned | 0 | 0 | 3,320 | 3,320 |
| d3-nano fine-tuned | 0 | 0 | 3,340 | 3,340 |
| logreg | 3,299,153 | 507,099 | 2,390 | 3,808,642 |
| frozen Qwen2.5-1.5B | 0 | 102,935,543 | 0 | 102,935,543 |
| random | 348,235,510 | 43,846,326 | 5,000 | 392,086,836 |
| approve all | 843,817,672 | 0 | 0 | 843,817,672 |

The recorded bars at 10 USD. Their verdicts stand as recorded in those evals; this is only what they would read.

| bar | rule | need | at 50 | at 10 | |
|---|---|---:|---:|---:|---|
| Phase 4 bar 1 | frozen Qwen cheaper than rules | 16 of 20 | 0 | 0 | still fails |
| Phase 5 bar A | Qwen fine-tuned ≤ logreg | 10 of 20 | 19 | 19 | still passes |
| Phase 7 bar A | d3-nano fine-tuned ≤ logreg | 10 of 20 | 19 | 19 | still passes |
| Phase 7 bar C | d3-nano fine-tuned ≤ Qwen fine-tuned | 10 of 20 | 18 | 18 | still passes |
| Phase 9 bar A | d1-Omni fine-tuned ≤ logreg | 10 of 20 | 19 | 19 | still passes |
| Phase 9 bar C | d1-Omni fine-tuned ≤ Qwen fine-tuned | 10 of 20 | 20 | 20 | still passes |

Paired comparisons at 10 USD on `test200`–`test233`. Not corrected for multiple comparisons.

| lower cost | against | cheaper / dearer / ties | sign test p | mean difference | 95% CI | |
|---|---|---:|---:|---:|---|---|
| d1-Omni fine-tuned | logreg | 19 / 1 / 0 | 4e-05 | -190,270 | -249,959 to -136,727 | distinguishable |
| rules | d1-Omni fine-tuned | 1 / 0 / 19 | 1 | -0.5 | -2 to 0 | not distinguishable |
| d1-Omni fine-tuned | Qwen fine-tuned | 8 / 0 / 12 | 0.008 | -4 | -6 to -2 | distinguishable |
| Qwen fine-tuned | d3-nano fine-tuned | 2 / 0 / 18 | 0.5 | -1 | -2 to 0 | not distinguishable |
| rules | Qwen fine-tuned | 9 / 0 / 11 | 0.004 | -4 | -6 to -2 | distinguishable |
| rules | logreg | 19 / 1 / 0 | 4e-05 | -190,271 | -249,959 to -136,727 | distinguishable |

Thresholds (review, decline) at both prices: logreg 0.01, 0.2; Qwen fine-tuned 0.05, 0.95; d3-nano fine-tuned 0.05, 0.9; d1-Omni fine-tuned 0.05, 0.95.

Dev totals over dev0–dev9 (in-sample thresholds; these models never reached test), thresholds unchanged by the price:

| model | thresholds | total at 50 | total at 10 |
|---|---|---:|---:|
| logreg | 0.01, 0.2 | 1,631,627 | 1,626,307 |
| frozen d3-mini | 0.2, 0.9 | 2,697,754 | 2,688,034 |
| frozen d1-Omni-2B | 0.5, 0.7 | 2,698,314 | 2,688,554 |
| frozen d3-nano | 0.2, 0.7 | 5,784,720 | 5,774,720 |
| frozen Qwen2.5-1.5B | 0.7, 0.9 | 27,016,342 | 27,006,342 |

## What it means (Inferred)

- **The fee is too small to matter on PaySim's amounts.** A wrong call costs thousands to millions; the fee difference is 40 USD on at most 25 reviews a window, at most 1,000. So no threshold is worth changing for it. On IEEE-CIS, with amounts around 70 USD, the same fee does change what is worth reviewing.
- **The ratios among the near-perfect setups are the same, not closer.** They differ by one or two reviews per window. Absolute gaps shrink by five (d1-Omni vs Qwen fine-tuned, 20 → 4 USD per window), so "distinguishable" here means a few dollars.
- **The prediction made before running held, with one miss.** Fee-only setups fell to a fifth, logreg and frozen Qwen barely moved, and bars and order stayed put. **The miss:** "rules falls to about a fifth" is true on `test200`–`test233` and on the median of `test100`–`test119` (825 → 165), but its mean there stays near 400,000. I had not allowed for one window, `test109`, where rules wrongly declined eight round 10M TRANSFERs at 1,000,000 USD of friction each. That is 8,000,000 of its 8,003,380 total at 10 USD. (Observed)

## Negative results

- **A cheaper review did not make logreg or any model decide differently.** Re-picking was pre-registered as the headline, and it changed nothing. (Observed)
- **The fine-tuned models still don't beat rules on cost.** Rules has the lowest mean on both window sets at both prices, though d1-Omni fine-tuned is not distinguishable from it (a difference of 0.5 USD per window). (Observed)
- **Rules' 10M problem is untouched.** `test109` is friction, not fees, so the price can't help it. Fine-tuned d1-Omni on PaySim's 10M burst hours is still not run (owner: not now). (Observed)

## Limits

- **All these test windows had been read before.** Nothing was tuned on them, because thresholds come from dev only and the price is the owner's. New windows would need the models re-run. (Implementation choice)
- **Re-picked and "same decisions" can only differ through thresholds.** Prompts, training and the model's order choice were made under 50 USD and are not revisited. Whether the hand-written rules would be written differently at 10 USD is not tested. (Implementation choice)
- **Reviews are still assumed to always find the truth**, at a flat assumed fee. (Implementation choice)
- **PaySim is synthetic and these windows are enriched** to about 5% fraud. The natural-rate columns reweight them; trace weights are rounded, so they differ from the older reports by up to 0.5%, within the gate. (Observed)
- **The analyst console's Eval page and the older write-ups still show 50 USD**, by design. These results are not on the Eval page. (Observed)

## Not run

- Any model, or any new window.
- A friction minimum of 1 USD on PaySim, to match IEEE-CIS. The owner's change was to the review fee.
- Fine-tuned d1-Omni on PaySim's 10M burst hours (owner: not now).
