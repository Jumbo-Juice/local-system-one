# Results: IEEE-CIS fine-tune on every fit fraud (Phase 12)

Eval `20261103-142622` (`python -m fraud.eval_ieee_full --adapter ...`). Pre-registration: the `fraud/eval_ieee_full.py` docstring, committed before training and before any model scored a new window.
Numbers: `fraud/results/eval_ieee_full/20261103-142622.json|.md`. Model scores are uncalibrated. A review costs 10 USD.

## Short answer

- **Four times the fraud rows did not give a distinguishable saving.** The new adapter costs 3,483 per window against Phase 11's adapter at 3,710: cheaper in 13 of 20 windows, mean paired difference
  -227 USD, 95% CI -555 to 90. It is the lower mean, but by the pre-registered rule the two are not distinguishable. (Observed)
- **It did not beat logistic regression on all the columns.** 3,483 vs 3,274: logreg is 209 lower on the mean, the new adapter is cheaper in 11 of 20 windows, CI -607 to 182. Not distinguishable either way. (Observed)
- **So the lowest-cost option is still not shown to be a language model.** Logreg on all the columns keeps the lowest mean and needs no GPU. Nothing here justifies replacing it on cost. (Inferred)
- **Ranking did improve:** AUC 0.851 vs 0.838 on test and 0.884 vs 0.869 on dev. The intervals on test overlap, and AUC has no pre-registered paired test. (Observed)

## Setup (Implementation choice unless cited)

- The adapter: `20261103-091222_lora-d1-ieee-full`, d1-Omni-2B from frozen weights, the same prompt, LoRA and optimiser as Phase 11. 47,952 fit rows (11,988 fraud, 35,964 legit), seed 0, one pass, 5,994 batches.
- New test windows: seed 20261103, 20 windows of 1,000 contiguous rows that share no row with Phase 11's. 20,000 rows, 778 fraud, natural rate.
- The new adapter read Phase 11's 10 dev windows in both orders. Phase 9's debias rule picked `rev`; thresholds from dev. Phase 11's adapter, logreg-all and logreg-same kept the order and thresholds of eval
  `20261102-183431`, nothing re-picked. The trace headers confirm Phase 11's adapter ran at its old thresholds. (Observed)
- Costs as Phase 11: friction 10%, at least 1 USD; review 10 USD; at most 5% of a window reviewed; past the budget a review becomes an approve.

## Results (Observed)

| setup | mean cost | 95% CI | AUC | 95% CI | recall | precision | alert rate | p90 ms |
|---|---:|---|---:|---|---:|---:|---:|---:|
| logreg, all columns | 3,274 | 2,707–3,923 | 0.864 | 0.844–0.883 | 0.652 | 0.248 | 0.102 | n/a |
| **d1-Omni fine-tuned on every fit fraud** | 3,483 | 2,874–4,180 | 0.851 | 0.827–0.871 | 0.562 | 0.256 | 0.085 | 113 |
| d1-Omni fine-tuned in Phase 11 | 3,710 | 3,097–4,386 | 0.838 | 0.815–0.859 | 0.594 | 0.205 | 0.112 | 117 |
| logreg, the prompt's fields | 4,212 | 3,700–4,728 | 0.797 | 0.769–0.825 | 0.634 | 0.160 | 0.154 | n/a |
| approve all | 5,283 | 4,324–6,274 | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a |

Where the money goes, total USD over the 20 windows. Recomputed from the traces apart from the report code; costs and AUCs match the report (the largest gap is 0.1 USD, rounding).

| setup | fraud let through | good customers declined | review fees | total | declines | of which good customers |
|---|---:|---:|---:|---:|---:|---:|
| logreg, all columns | 43,178 | 12,684 | 9,620 | 65,482 | 1,083 | 62% |
| d1-Omni, every fit fraud | 53,307 | 6,767 | 9,590 | 69,664 | 745 | 57% |
| d1-Omni, Phase 11 | 52,902 | 11,717 | 9,580 | 74,200 | 1,291 | 68% |
| logreg, the prompt's fields | 41,013 | 33,553 | 9,680 | 84,246 | 2,117 | 79% |
| approve all | 105,651 | 0 | 0 | 105,651 | 0 | n/a |

Paired comparisons. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | sign test p | |
|---|---|---:|---:|---|---:|---|
| **new adapter** | **Phase 11's adapter (the primary question)** | 13 / 7 | -227 | -555 to 90 | 0.26 | not distinguishable |
| **logreg, all columns** | **new adapter (the second question)** | 9 / 11 | -209 | -607 to 182 | 0.82 | not distinguishable |
| new adapter | logreg, the prompt's fields | 16 / 4 | -729 | -1,232 to -191 | 0.01 | distinguishable |
| new adapter | approve all | 18 / 2 | -1,799 | -2,482 to -1,153 | 0.0004 | distinguishable |
| Phase 11's adapter | logreg, the prompt's fields | 17 / 3 | -502 | -931 to -44 | 0.003 | distinguishable |
| logreg, all columns | Phase 11's adapter | 13 / 7 | -436 | -865 to -55 | 0.26 | distinguishable |
| Phase 11's adapter | approve all | 16 / 4 | -1,573 | -2,339 to -846 | 0.01 | distinguishable |

Only the first two rows are pre-registered questions. The others are not corrected for multiple comparisons.

| | value |
|---|---|
| Latency bar, p90 ≤ 250 ms | both PASS: 113 and 117 ms. Slowest single test window: 137 and 140 ms |
| Training | 5,994 of 5,994 batches, not stopped early; 314.7 min process; 3.14 s/batch; peak 3.83 GiB |
| Loss | 0.583 over the first 100 batches → 0.344 over the last 100. By fifths: 0.445, 0.390, 0.396, 0.367, 0.353 |
| Thresholds (review, decline) | new adapter 0.472, 0.764; Phase 11's 0.388, 0.531 |
| Dev → test AUC | new adapter 0.884 → 0.851; Phase 11's 0.869 → 0.838; logreg same 0.862 → 0.797; logreg all 0.895 → 0.864 |
| Adapter sha256 | `10970d702ee9990f…`; rows sha256 `e6d057cc1de3…` |

Cost sensitivity, no bar and no CI; thresholds re-picked on dev from the same scores (the 10 USD row uses the eval's thresholds):

| cost set | new adapter | Phase 11's | logreg same | logreg all | approve all |
|---|---:|---:|---:|---:|---:|
| review 1 USD | 3,294 | 3,279 | 3,777 | 2,841 | 5,283 |
| review 5 USD | 3,494 | 3,470 | 3,970 | 3,034 | 5,283 |
| **review 10 USD** | 3,483 | 3,710 | 4,212 | 3,274 | 5,283 |
| review 20 USD | 3,907 | 4,117 | 4,625 | 3,617 | 5,283 |

## What it means (Inferred)

- **The saving, such as it is, is fewer wrongly declined customers.** Against Phase 11's adapter the new one lets through 405 USD more fraud and costs 4,950 USD less in good customers declined. Its decline line sits higher
  and it declines 546 fewer rows, so it stops fewer frauds (437 vs 462 of 778) but is right more often when it does.
- **The two models lose money in opposite places.** Logreg on all the columns lets through 10,129 USD less fraud than the new adapter and costs 5,917 USD more in declined customers.
- **The lead over Phase 11's adapter depends on thresholds.** It is gone at reviews of 1 and 5 USD, where the old adapter is slightly lower, and back at 20 USD. The new adapter's 5 USD cost is above its 10 USD cost, which a threshold
  fitted perfectly could not do, so where dev puts the lines moves these numbers by as much as the difference being tested.
- **More training may still help a little.** Loss was still falling in the last fifth here, while Phase 11's had flattened after about 60%. Falling loss on fit rows does not promise lower cost on test, and no second pass was tried.
- **The prediction made before running held.** "Beats the old on AUC and probably on cost, but the difference may not be distinguishable": better AUC and lower mean cost, not distinguishable. "Probably still does not beat logistic
  regression on all columns": logreg kept the lower mean, though the new adapter is not distinguishably dearer.
- **Phase 11's weak finding repeated.** Logreg on all the columns beat Phase 11's adapter again on windows nobody had read (CI -865 to -55; Phase 11 -728 to -7). That is the same result twice, not a pre-registered test.
  A distinguishable loss for the old adapter and a non-distinguishable one for the new adapter does not itself show the new adapter is better.

## Negative results

- **The primary question has no distinguishable answer.** About five hours of training on four times the rows did not produce a saving this test can separate from zero. (Observed)
- **No language model has beaten logistic regression on all the columns on mean cost**, at any review fee in either phase. (Observed)
- **The new adapter stops fewer frauds than Phase 11's** (recall 0.562 vs 0.594) and than both logregs. A reader who values stopped fraud over customer friction would not call it better. (Observed)
- **At reviews of 1 and 5 USD the new adapter is not the cheaper fine-tune.** (Observed, no CI)
- **The first training run died.** It reached batch 1,551 of 5,994 before standby and a Windows Update restart, so it was restarted from zero as pre-registered. Its checkpoint is not scored. (Observed)

## Limits

- **One training run, one seed.** Nothing here measures how much a second seed would move the costs.
- **20 windows, 778 frauds, and a heavy tail.** Approve-all ranges from 2,127 to 9,109 per window. A real difference of about 200 USD a window may be too small for this test to find. (Inferred)
- **New rows, same period.** The windows share no row with Phase 11's but come from the same stretch of time, so this is not a later period. All the setups score higher here than in Phase 11 (logreg all 0.864 vs 0.846), so these windows look a little easier and the two phases' numbers should not be compared directly. (Observed)
- **Thresholds differ in age.** The new adapter's are new and Phase 11's were already known. Both come from the same dev windows, and dev is not test for either.
- **USD amounts and the meaning of the C, D, M and V columns are from memory of the competition's data page, not verified.** Whether all of them are known at decision time can't be checked. If some aren't, every number here is optimistic,
  logreg-all's most of all. (Inferred)
- **Reviews are assumed to always find the truth**, at a flat assumed fee. (Implementation choice)
- **A change during the run, not pre-registered.** At the owner's request for a faster finish, the training and scoring processes ran at Above Normal priority from 14:12, the last 14 minutes of training and all of scoring. It should only affect timing, not scores. (Inferred)
  It probably helped little: Python used about one of 8 CPU threads, so the CPU was not crowded. (Observed at 14:11; the effect itself was not measured)
- **Latency depends on the state of the laptop, which is also a correction to Phase 11.** The same Phase 11 adapter on test windows, one order: p90 95 ms in standby, 115 ms awake, 117 ms in this run. So most of the gap is the machine being awake rather than the adapter, and Phase 11's reported 109 ms mixes
  the two. (Observed) On the first three dev windows, same rows and both orders, this run's median is 24–28% slower than Phase 11, which ran in standby. Heat, the Unsloth update, wallpaper and video may add to that, but they can't be separated here. The wallpaper was gone by 14:39 and dev windows did not get faster. (Inferred) Both models still pass.
- **The screen was held on for the whole 7 hours.** No standby or restart event was logged from 09:11 to 16:11. Scores are unaffected either way. (Observed)

## Not run

- A second pass over the fit rows, a second seed, or a different prompt.
- Frozen d1-Omni-2B on these windows.
- Re-pricing older PaySim results at 10 USD. They stay at 50, as committed.
- Fine-tuned d1-Omni on PaySim's 10M burst hours (owner: not now).
