# Results: the fine-tuned brain (Phase 5)

Eval `20261005-022333`, run on 2026-10-05 on the Lenovo (Arc 140V iGPU). Pre-registration: the
docstring of `fraud/eval_ft.py`, committed in `7d193db` before the adapter was trained. Raw results:
[`fraud/results/eval/20261005-022333.json`](../results/eval/20261005-022333.json) and
[`.md`](../results/eval/20261005-022333.md). The traces (`fraud/runs/20261005-022333_*`) and the
adapter (`fraud/models/20261005-013044_lora`, sha256 `02a0c5d9…`) are git-ignored. Every number
below is **Observed** in that eval unless labelled otherwise.

## The question and the answer

Can a LoRA fine-tuned Qwen2.5-1.5B, behind the same type filter and reading the same prompt as the
Phase 3 hybrid, decide as cheaply as logistic regression? **Yes, and more cheaply:** it was cheaper
than logistic regression in 19 of 20 new test windows. It **did not beat rules-only** (0 of 20),
which was expected and stated before the run.

| bar | rule | value | result |
|---|---|---:|---|
| gate | dev AUC(hybrid-ft) > dev AUC(hybrid-1.5b) | 0.995 vs 0.903 | **PASS** |
| A | hybrid-ft cost ≤ logreg cost in ≥ 10 of 20 new windows | **19 of 20** | **PASS** |
| B | hybrid-ft latency p90 ≤ 250 ms | **127.9 ms** | **PASS** |
| (old bar 1, reported) | hybrid-ft cheaper than rules in ≥ 16 of 20 | 0 of 20 | FAIL (expected) |

## Setups (20 new test windows, 500 transactions each, ~5% fraud)

| setup | mean cost | median cost | recall | precision | alert rate | p50 / p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| **hybrid-ft** (filter + fine-tuned 1.5B, dev thresholds) | **830** | 825 | 1.00 | 0.87 | 0.060 | 125 / 128 |
| rules-only | 808 | 800 | 1.00 | 0.87 | 0.060 | – |
| logistic regression (filter + numpy LR) | 190,910 | 167,148 | 0.89 | 0.85 | 0.056 | – |
| hybrid-1.5b (frozen Phase 3 model, argmax) | 5,146,777 | 4,988,782 | 1.00 | 0.14 | 0.386 | 122 / 125 |
| random | 19,605,342 | 14,211,557 | 0.58 | 0.14 | 0.222 | – |
| approve-all | 42,190,884 | 34,751,285 | 0.00 | 0.00 | 0.000 | – |

Paired window counts (two-sided sign test, ties dropped): hybrid-ft cheaper than logreg in 19,
logreg cheaper in 1 (test217: 650 vs 900), p = 4.0e-5. hybrid-ft vs rules: 0 cheaper, 9 dearer,
11 ties, p = 0.0039. hybrid-ft cheaper than the frozen hybrid in 20 of 20 (p = 1.9e-6).
The new windows reproduce the Phase 3 ordering of the baselines: rules cheaper than logreg in 19 of 20.

## What the fine-tune changed

- **Ranking.** On dev0–dev9 (2,194 model rows, 260 fraud) the AUC of the decline score went from
  0.903 to **0.995**. Logistic regression on the same rows: 0.992 (computed separately, same rows;
  not part of the gate).
- **Argmax.** The frozen model's top option was right for 13.4% of its 3,862 test decisions
  (it declined nearly everything). The fine-tune's top option was right for 97.8%. With argmax alone
  (the dev traces, before thresholds) its mean dev cost per window was 0.31M against 5.35M.
- **Scores (uncalibrated).** ECE of the top score 0.575 → 0.011; mean `outside_mass` 0.0117 →
  0.0002. These are softmax values of a model trained at a 1:3 fraud share on enriched windows;
  they are not fraud probabilities.
- **Speed.** Unchanged: the adapter is merged into the weights (p90 128 ms vs 125 ms).
- **Cost of training** (Observed, the manifest): 12,084 examples, 1,510 steps, 52 min, 3.38 GiB peak
  on the iGPU.

## Insight: it learned the rule table

- On the 3,862 transactions the model judged in the test windows, hybrid-ft wanted **the same
  action as rules-only for 3,853 (99.8%)**: 3,264 approve, 323 review, 266 decline on both sides.
- All 9 differences are one case: a **fraud TRANSFER of exactly 10,000,000 to a new account**.
  Rules-only declines it ("round transfer" rule); the fine-tune scores it 0.92–0.93, just under its
  0.95 decline threshold, and sends it to review. A review costs 50 and finds the fraud, so each of
  these costs 50 more than the rule. That is the whole gap to rules-only: 9 windows dearer by
  exactly 50, the other 11 tied.
- **Inferred:** trained on labels only, the model rediscovered the two signals that nearly separate
  PaySim fraud (`fraud/docs/data.md`) and the same review band as the six-row rule table in
  `fraud/rules.py`. It never saw the rule table. Its
  cost ceiling is the rules' cost, because rules-only already pays only review fees in most windows
  (`fraud/docs/results.md`, "Why rules-only is so hard to beat").
- **Inferred, untested:** the model is less sure than the rule about capped 10M transfers. In
  Phase 3, rules-only lost 8.0M in test109 by declining eight legit 10M transfers (a burst at
  PaySim's cap). Whether the fine-tune would review them instead is not measured here: test109
  belongs to the Phase 3 eval, and the new windows have no such burst.

## Why logistic regression lost

- Its cost is missed fraud, not friction. On the model rows it approved 59 fraud transactions,
  57 of which the fine-tune sent to review; the fine-tune approved 2 fraud transactions in total.
  Recall 0.89 vs 1.00.
- 57 of its 59 approved frauds are **TRANSFERs to a new receiver**, the cell the rule table sends to
  review (4.2% fraud in training). Logistic regression scores them at about 0.009. Its review
  threshold, picked on dev windows, is 0.01, the lowest value in its pre-registered grid, so part
  of that cell falls just below it and is approved.
- **Inferred:** its ranking is about as good (dev AUC 0.992 vs 0.995). The difference is where the
  scores sit. Fitted at the natural fraud rate, logistic regression puts the whole review cell
  near 0.01, at the edge of its grid; the fine-tune, trained at a 1:3 fraud share, puts it well
  inside its review band (≥ 0.05). A lower logreg grid might close much of the gap; that was not
  pre-registered and is not tested here.
- Known asymmetry (stated in the pre-registration): logistic regression also reads four account
  counts the prompt does not show. It lost with more information, not less.

## Limits

- One adapter, one training seed. XPU bf16 training is not bit-reproducible run to run
  (PROGRESS 2026-10-05), so a retrain will differ slightly.
- 20 windows from one synthetic dataset. PaySim's fraud follows a fixed simulated pattern; a
  rule table and a model that both find it say little about real fraud.
- The thresholds came from 10 dev windows with 260 model-row fraud cases; the 0.95 decline
  threshold sits just above the 10M-transfer scores (0.92–0.93), so it decides the 9 differences.

## Summary

The fine-tune fixed what Phase 3 found broken. The 1.5B's ranking went from 0.90 to 0.995 AUC,
its decisions went from "decline everything" to the rules' own decisions on 99.8% of transactions,
and it beat logistic regression in 19 of 20 untouched windows at the same latency. It ties
rules-only and cannot beat it on these signals. A local model can learn a usable fraud policy from
labels in under an hour on a laptop iGPU, but here the policy it learned is the one a person can
write down in six rules. Its win over logistic regression is real on the pre-registered setup, but
part of it comes from where logistic regression's scores sit relative to its threshold grid, not
only from better ranking.
