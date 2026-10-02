# Results: the pre-registered fraud eval

Eval `20261003-052158`, run on 2026-10-03 on the Lenovo (Arc 140V iGPU, 7.18 GiB reported).
Pre-registration: the docstring of `fraud/eval.py`, committed in `47c32b1` before any test window
ran. Raw results: [`fraud/results/eval/20261003-052158.json`](../results/eval/20261003-052158.json)
and [`.md`](../results/eval/20261003-052158.md). The traces (`fraud/runs/20261003-052158_eval_*`)
are git-ignored. Every number below is **Observed** in that eval unless labelled otherwise.

## The bars

| bar | rule | value | result |
|---|---|---:|---|
| 1 | hybrid-1.5b cheaper than rules-only in ≥ 16 of 20 test windows | **0 of 20** | **FAIL** (expected and stated before the run) |
| 2 | hybrid-1.5b per-decision latency p90 ≤ 250 ms | **127.2 ms** | **PASS** |

Latency is over 3,800 model decisions, order debias on (two readings per decision), float32 LM head:
p50 121 ms, p90 127 ms, max 203 ms. The prompt is 214 tokens on average (max 223).

## Setups (20 test windows, 500 transactions each, ~5% fraud)

| setup | mean cost | median cost | recall | precision | alert rate | cheaper than rules |
|---|---:|---:|---:|---:|---:|---:|
| rules-only | 400,845 | 825 | 1.00 | 0.86 | 0.061 | – |
| logistic regression (filter + numpy LR) | 549,247 | 134,044 | 0.91 | 0.80 | 0.062 | 2 of 20 |
| **hybrid-1.5b** (filter + Qwen2.5-1.5B) | 6,373,694 | 5,404,353 | 1.00 | 0.15 | 0.380 | 0 of 20 |
| random (filter + random) | 17,877,963 | 14,374,362 | 0.54 | 0.14 | 0.208 | 0 of 20 |
| approve-all | 31,735,048 | 27,654,563 | 0.00 | 0.00 | 0.000 | 0 of 20 |

Paired window counts (sign test, two-sided): rules cheaper than the hybrid in 20 of 20
(p = 1.9e-6); logistic regression cheaper than the hybrid in 20 of 20 (p = 1.9e-6); the hybrid
cheaper than random in 19 of 20 (p = 4.0e-5); rules cheaper than logistic regression in 18 of 20
(p = 4.0e-4).

## What the model did

- **It declined every transaction it saw: 3,800 of 3,800** (520 fraud, 3,280 legit). Its recall of
  1.00 is a by-product of that, not detection. All its cost is friction on the 3,280 legit TRANSFERs
  and CASH_OUTs it declined.
- This was known before the run. On dev windows no prompt variant (v0–v7) moved the top option off
  "decline", although the scores ranked fraud above legit (AUC 0.90–0.93 for v4). A threshold rule
  on v4's scores, tuned on dev windows, also lost to rules-only in 10 of 10 dev windows
  (`fraud/PROGRESS.md`). The owner chose to keep argmax and report the result.
- **Model scores (uncalibrated):** mean top score 0.708, but the top option was right for only
  13.7% of the decisions (legit → approve, fraud → review or decline). ECE 0.571. Mean
  `outside_mass` 0.0115: the model almost always answers with one of the three labels; it just
  picks the same one.
- **The two readings disagree.** In the listed order (no, unsure, yes) the top option was
  "yes: decline" in 3,800 of 3,800 decisions, typically with a score above 0.9. In the reversed
  order (yes, unsure, no), the model's top option was "no: approve", which is then the last
  option, in 2,687 of 3,800 decisions (71%), and "decline" in the other 1,113. The averaged score
  still put "decline" on top every time, because the listed-order readings were far more
  confident (e.g. 0.94 vs 0.63).
- **Inferred:** much of the 1.5B's answer is a preference for the last-listed option, on top of
  some content signal (the scores rank fraud above legit on dev windows). Order debias removes
  the position effect from the ranking but not from the argmax when one reading is much more
  confident than the other. The argmax of this model is not a usable fraud decision here.

## Why rules-only is so hard to beat

- In 19 of 20 test windows rules-only cost only its review fees (650–1,150): every fraud
  caught, no legit declined. The same held in all 10 dev windows.
- **Inferred:** two signals nearly separate PaySim fraud (a TRANSFER to a receiver that never
  received money; a CASH_OUT of an amount already moved this hour; `fraud/docs/data.md`). Rules on
  those two get almost everything right, and the rest costs at most a review fee of 50 per row.
  To be cheaper in a window, a decider has to clear every one of the ~15–20 reviewed transfers
  without a review and without a mistake. One wrongly declined transfer costs ~10% of ~0.5M, and
  one missed fraud costs its full amount. Bar 1 was out of reach for any decider on these signals,
  not only for the model.
- Logistic regression was cheaper than rules in 2 windows (test100: 500 vs 1,100; test117: 350
  vs 900). There it declined most of the fraud instead of reviewing it, and declined no legit row.
  Over all 20 windows its cost was 8.42M of wrong declines (8.0M of it the same eight 10M
  transfers in test109, see below), 2.55M of missed fraud (recall 0.91) and 13,500 of reviews.

## A rules-only failure (test109)

- In test109 (steps 327–328) rules-only declined **8 legit TRANSFERs of exactly 10,000,000** under
  the "round transfer" rule, for a cost of 8.0M. That is 99.8% of rules-only's total over the 20
  windows. Each had the same amount moved 41–322 times earlier in the hour.
- **Inferred:** PaySim caps a transaction at 10,000,000. A large legit transfer appears as a
  burst of capped 10M rows, and 10M is a multiple of 1,000. The training steps had only 39 round
  TRANSFERs (37 fraud), so the rule did not see such a burst.
  Per `fraud/CLAUDE.md` a test result does not change the rules; a fix would need new, untouched
  windows.
- The burst also dominates the natural-rate costs. test109's natural weight is ~110 per legit row,
  which turns rules-only's 8.0M into 878M there and gives a natural-rate mean of 43.9M, close to
  approve-all's 47.0M. Medians are more representative: rules-only 3,651; logistic regression
  225,693; hybrid 59.7M; random 70.6M; approve-all 40.5M.

## Summary

- **Observed:** System One answers fast enough for a payment stream on an iGPU (p90 127 ms with
  order debias). Its decisions on this task are worthless: the hybrid declined everything and
  lost to rules-only and to logistic regression in every test window.
- **Observed:** two hand-written rules from the training steps beat every other setup, with one
  failure mode on capped 10M transfer bursts.
- **Inferred:** with leak-free PaySim signals the problem is close to rule-separable, which leaves
  a language model little room to add value. A fairer test of "does the model add anything" would
  need data where the rules are not near-perfect (a non-goal for v1).
- **Not tested:** the 3B (does not fit the free iGPU memory) and a score-threshold decision rule
  on the test windows (dev only).
