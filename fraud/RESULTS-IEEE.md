# Results: first real data, IEEE-CIS (Phase 11)

Eval `20261102-183431` (`python -m fraud.eval_ieee --ieee-adapter ...`). Pre-registration and the owner's amendment: the `fraud/eval_ieee.py` docstring,
committed before any eval output was read. Numbers: `fraud/results/eval_ieee/20261102-183431.json|.md`. Model scores are uncalibrated. A review costs 10 USD.

## Short answer

- **Training on the real data was worth it.** The IEEE fine-tune costs 3,794 per window against the PaySim fine-tune's 4,836: 22% less, cheaper in 17 of 20 windows,
  mean paired difference -1,042 USD, 95% CI -1,664 to -445. Same base model, prompt, windows and training size, so this changes one thing. (Observed)
- **Frozen vs PaySim fine-tune: no winner on cost.** 4,858 vs 4,836; 9 windows to 11; CI -591 to 544. The fine-tune ranks far better (AUC 0.739 vs 0.571) but declines too many
  good customers. They also differ in base model, so this can't say what fine-tuning alone did. (Observed)
- **Still not the cheapest.** Logistic regression on all the columns costs 3,426, 368 less than the IEEE fine-tune (CI -728 to -7, only just). So the lowest-cost
  option on this data is not a language model. The IEEE fine-tune does beat logistic regression on the same fields as its prompt: 4,373, CI -951 to -177. (Observed)

## Setup (Implementation choice unless cited)

- IEEE-CIS train files, split by row order 60/20/20. Natural fraud rate, no enrichment, no rules, no type filter. 10 dev and 20 test windows of 1,000 contiguous rows;
  20,000 test rows, 685 fraud.
- Costs: approved fraud loses its amount; a wrong decline costs 10% of the amount, at least 1 USD; a review costs 10 USD (owner); at most 5% of a window reviewed; a review
  past the budget becomes an approve. Thresholds picked on dev per setup, and "never" is on the grid.
- Drawn with seed 20261102: frozen Qwen2.5-1.5B, fine-tuned d1-Omni-2B (`20261102-152745_lora-d1`, trained on PaySim). New: `20261102-210541_lora-d1-ieee`, same base,
  LoRA, schedule and 1,510 batches, on 12,084 fit rows (3,021 fraud, 9,063 legit).

## Results (Observed)

| setup | mean cost | 95% CI | AUC | 95% CI | recall | precision | alert rate | p90 ms |
|---|---:|---|---:|---|---:|---:|---:|---:|
| logreg, all columns | 3,426 | 2,749–4,175 | 0.846 | 0.821–0.870 | 0.590 | 0.202 | 0.100 | n/a |
| **d1-Omni fine-tuned on IEEE-CIS** | 3,794 | 3,078–4,593 | 0.822 | 0.796–0.847 | 0.546 | 0.179 | 0.104 | 109 |
| logreg, the prompt's fields | 4,373 | 3,716–5,049 | 0.788 | 0.755–0.818 | 0.601 | 0.137 | 0.150 | n/a |
| d1-Omni fine-tuned on PaySim | 4,836 | 3,923–5,775 | 0.739 | 0.717–0.763 | 0.575 | 0.096 | 0.204 | 98 |
| frozen Qwen2.5-1.5B | 4,858 | 3,987–5,825 | 0.571 | 0.543–0.605 | 0.114 | 0.082 | 0.047 | 173 |
| approve all | 5,295 | 4,120–6,557 | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a |

Where the money goes, total USD over the 20 windows. Recomputed from the traces apart from the report code; the totals match.

| setup | fraud let through | good customers declined | review fees | total |
|---|---:|---:|---:|---:|
| logreg, all columns | 42,533 | 16,540 | 9,440 | 68,513 |
| d1-Omni fine-tuned on IEEE-CIS | 52,152 | 14,070 | 9,660 | 75,882 |
| logreg, the prompt's fields | 40,185 | 37,996 | 9,280 | 87,461 |
| d1-Omni fine-tuned on PaySim | 61,211 | 25,508 | 10,000 | 96,719 |
| frozen Qwen2.5-1.5B | 87,667 | 0 | 9,490 | 97,157 |
| approve all | 105,907 | 0 | 0 | 105,907 |

Paired comparisons. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | sign test p | |
|---|---|---:|---:|---|---:|---|
| **IEEE fine-tune** | **PaySim fine-tune (the primary question)** | 17 / 3 | -1,042 | -1,664 to -445 | 0.003 | distinguishable |
| PaySim fine-tune | frozen 1.5B (different bases) | 9 / 11 | -22 | -591 to 544 | 0.82 | not distinguishable |
| IEEE fine-tune | logreg, the prompt's fields | 14 / 6 | -579 | -951 to -177 | 0.12 | distinguishable |
| logreg, all columns | IEEE fine-tune | 15 / 5 | -368 | -728 to -7 | 0.04 | distinguishable |
| IEEE fine-tune | approve all | 15 / 5 | -1,501 | -2,355 to -752 | 0.04 | distinguishable |
| logreg, the prompt's fields | PaySim fine-tune | 11 / 9 | -463 | -1,162 to 219 | 0.82 | not distinguishable |
| logreg, all columns | PaySim fine-tune | 18 / 2 | -1,410 | -2,173 to -764 | 0.0004 | distinguishable |
| PaySim fine-tune | approve all | 12 / 8 | -459 | -1,030 to 142 | 0.50 | not distinguishable |

Only the first row is the primary question. The others are not corrected for multiple comparisons, so the logreg-all row, whose CI nearly touches 0, is weak evidence. (Inferred)

| | value |
|---|---|
| Latency bar, p90 ≤ 250 ms | all three models PASS: 173, 98 and 109 ms. Slowest single test window: 225, 123 and 133 ms |
| Training | 1,510 of 1,510 batches, not stopped early; 65.0 min process; 2.55 s/batch; peak 3.83 GiB |
| Loss | 0.549 over the first 100 batches → 0.394 over the last 100. By fifths: 0.504, 0.447, 0.391, 0.390, 0.401 |
| Thresholds (review, decline) | IEEE fine-tune 0.388, 0.531; PaySim fine-tune 0.136, 0.242; frozen 0.495, never |
| Dev → test AUC | IEEE fine-tune 0.869 → 0.822; PaySim fine-tune 0.796 → 0.739; frozen 0.582 → 0.571; logreg same 0.862 → 0.788; logreg all 0.895 → 0.846 |
| Argmax on dev | the frozen 1.5B says decline on 98.6% of rows; both fine-tunes approve about 95% and never pick review |
| Adapter sha256 | `769bfe7b021598e1…`; rows sha256 `bf11133ad20d…` |

Cost sensitivity, no bar and no CI; thresholds re-picked on dev from the same scores:

| cost set | frozen | PaySim fine-tune | IEEE fine-tune | logreg same | logreg all | approve all |
|---|---:|---:|---:|---:|---:|---:|
| review 1 USD | 4,422 | 4,386 | 3,359 | 3,955 | 3,001 | 5,295 |
| review 5 USD | 4,621 | 4,586 | 3,553 | 4,141 | 3,190 | 5,295 |
| **review 10 USD** | 4,858 | 4,836 | 3,794 | 4,373 | 3,426 | 5,295 |
| review 20 USD | 5,295 | 4,763 | 4,118 | 4,778 | 3,804 | 5,295 |
| Phases 1–10 (review 50, friction min 10) | 5,295 | 5,435 | 4,373 | 4,270 | 4,035 | 5,295 |

## What it means (Inferred)

- **The gain came from both sides.** Against the PaySim adapter, the IEEE adapter lets through 9,059 USD less fraud and costs 11,438 USD less in wrongly declined good customers.
  The PaySim adapter's decline line sits lower because its scores separate less, so 89% of its 3,083 declines hit good customers. For the IEEE adapter it is 72% of 1,123.
- **The PaySim adapter carried over less than its AUC suggests.** 0.739 is well above frozen Qwen's 0.571, but it can't be credited to PaySim training: the frozen model is a different base, and
  the prompt's context states this data's fraud patterns. Frozen d1-Omni-2B on these windows would separate the two. It wasn't in the draw and hasn't been run.
- **The prompt leaves money on the table, but less than feared.** Logreg gains 0.057 AUC from the columns the prompt doesn't show. The IEEE adapter gets 0.034 more AUC than logreg on the same fields.
- **Every setup drops from dev to test**, so fraud on this data shifts over time. The test rows are the latest ones.
- **Predictions stated before running.** "Both likely lose to logistic regression": frozen and PaySim fine-tune do cost more than both on the mean, though PaySim vs logreg-same is not
  distinguishable. "The IEEE fine-tune beats the PaySim one on AUC and probably cost, but may still not beat logistic regression": held, with a split result on logistic regression.

## Negative results

- **No language model beat logistic regression on all the columns**, at any review fee in the sensitivity table. (Observed)
- **The PaySim fine-tune, as it stands, is not distinguishable from approving everything** (CI -1,030 to 142). PaySim's near-perfect numbers do not carry over. (Observed)
- At PaySim's old costs the PaySim fine-tune is dearer than approve-all (5,435 vs 5,295): the thresholds picked on dev did not hold on test. (Observed)
- At PaySim's old costs logreg on the same fields edges out the IEEE fine-tune (4,270 vs 4,373), no CI. (Observed)
- **The withdrawn bar would have failed.** The amendment withdrew "retraining is not necessary if the better model is no dearer than logreg-same in ≥ 10 of 20 windows". The better of the two
  drawn models, the PaySim fine-tune by 22 USD, reaches 9. Nothing was decided by this; it agrees with the owner treating retraining as guaranteed. (Observed)

## Limits

- **Only 3,021 of the 11,988 fit frauds were used**, to match the PaySim adapter's size, which is what keeps the comparison to one change. Loss flattened after about 60% of the batches, so more
  passes over these rows look unlikely to help; more rows are untested. (Inferred)
- **One split and one test period.** 20 windows, 685 frauds, and a heavy tail: approve-all ranges from 1,462 to 11,302 per window. The CIs on single setups are wide and overlap; the paired ones are what separate them. (Observed)
- **USD amounts and the meaning of the C, D, M and V columns are from memory of the competition's data page, not verified.** Whether all of them are known at decision time can't be checked. If some aren't,
  every number here is optimistic, logreg-all's most of all because it uses the most columns. (Inferred)
- **Five setups were scored while the fee was still 5 USD, then re-priced.** The fee only prices decisions, so `--reprice` re-picked dev thresholds and re-decided every trace at 10 USD from the recorded scores.
  The IEEE fine-tune was scored live at 10 USD. All six trace headers show 10.0. (Observed)
- **Reviews are assumed to always find the truth**, at a flat assumed fee. (Implementation choice)
- **One prompt, no iteration.** Its context was written from fit rows and not tuned, so a better prompt may do better; none was tried on this data. (Implementation choice)
- **Latency on a laptop in use.** The owner was watching light video during some of the run, so latencies may read high. Three rows stalled for 48.2, 19.8 and 5.6 minutes. The longest
  ends within seconds of a Windows Modern Standby exit in the system log (23:03:39), so the machine pausing is the likely cause rather than the video. (Inferred) Scores and p90 are not affected.
  (Observed: p90 ignores three rows.) Replays of `dev2` for the IEEE fine-tune and `test3` for the frozen model keep those pauses.
  Correction from Phase 12: the IEEE fine-tune's first 9 test windows ran in standby at p90 95 ms against 115 ms for its 11 awake windows, so its 109 ms reads low for a laptop in use rather than high (`fraud/RESULTS-IEEE-FULL.md`, Limits). (Observed)

## Not run

- Re-pricing older PaySim results at 10 USD. They stay at 50, as committed.
- Frozen d1-Omni-2B on these windows.
- A larger IEEE training run on all fit frauds. Since run as Phase 12: `fraud/RESULTS-IEEE-FULL.md`.
- Fine-tuned d1-Omni on PaySim's 10M burst hours (owner: not now).
