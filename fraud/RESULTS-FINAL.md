# Results: the final test (Phase 23)

Eval `20261104-090649` (`python -m fraud.eval_final`), run once. Pre-registration: the `fraud/eval_final.py` docstring, committed in `bfbfae6`, with dev results in `9c670f2`, before any setup scored a FINAL row.
Numbers: `fraud/results/eval_final/20261104-090649.json|.md`; read-outs beyond the registration, `extras.json`. 20 FINAL windows, reserved in the first commit of this work, 20,000 rows, 630 frauds, 104,651 USD of fraud. Model scores are uncalibrated. A review costs 10 USD. Lone prompts on an RTX 3060 Ti, bf16.

## Short answer

- **By the registered rule, the setup to deploy is d1-Omni-2B reading the tabular score, with the amount-aware rule and the customer line: `final-signal-cust`.** (Implementation choice, stated before the run)
- **Against the baseline, logreg with Phase 14's rule, its dollars aren't distinguishable:** 1,992 against 2,106, -115 USD per window, CI -391 to 273, 16 windows to 4. (Observed)
- **It does much better by good customers:** 247 declined against 1,051, 76% fewer, and 962 declined or reviewed against 1,703, 44% fewer. (Observed)
- **What that costs in fraud:** 335 of 630 frauds stopped against 384, 49 fewer, and 3,233 USD more fraud let through over 20 windows (29,338 against 26,105), 3% of all fraud dollars. (Observed)
- **The primary question has no distinguishable saving:** `final-signal-amt` against logreg, -163, CI -462 to 238, 15 windows to 5. (Observed)
- **A language model with no trees is no bargain here.** `final-rich-amt`'s dev tie with logreg didn't carry over: +321, CI -74 to 774, not distinguishable, and it's distinguishably dearer than `final-signal-amt`, by 484, CI 103 to 972. (Observed)
- **So most of what the deployed setup knows comes from the score it reads.** Its rank correlation with that score is 0.984 on FINAL rows. The test shows nothing the language model adds, because the score alone wasn't a registered setup. (Observed; the reading is Inferred)
- **Latency passes easily:** p90 56 ms for both adapters against the 250 ms bar, on this GPU. (Observed)

## Results (Observed)

| setup | mean cost | 95% CI | AUC | frauds stopped, of 630 | fraud USD stopped, of 104,651 | fraud let through | good customers declined | reviews | good declined USD | review fees | p90 ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final-signal-amt | 1,943 | 1,651–2,338 | 0.883 | 404 | 81,090 | 23,562 | 928 | 702 | 8,276 | 7,020 | 56 |
| **final-signal-cust, recommended** | **1,992** | 1,683–2,383 | 0.883 | 335 | 75,314 | 29,338 | **247** | 804 | 2,454 | 8,040 | 56 |
| **final-logreg-amt, the baseline** | **2,106** | 1,946–2,292 | 0.853 | 384 | 78,547 | 26,105 | 1,051 | 729 | 8,728 | 7,290 | n/a |
| final-rich-amt | 2,427 | 2,023–2,896 | 0.859 | 375 | 74,491 | 30,161 | 1,081 | 895 | 9,424 | 8,950 | 56 |
| approve all | 5,233 | 4,312–6,121 | n/a | 0 | 0 | 104,651 | 0 | 0 | 0 | 0 | n/a |

Paired. "Distinguishable" means the 95% window-bootstrap CI excludes 0.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| **final-signal-amt** | **baseline (primary)** | 15 / 5 | -163 | -462 to 238 | not distinguishable |
| **final-signal-amt** | **final-signal-cust (second)** | 11 / 9 | -49 | -160 to 47 | not distinguishable |
| **baseline** | **final-rich-amt (third)** | 12 / 8 | -321 | -774 to 74 | not distinguishable |
| final-signal-cust | baseline, reported | 16 / 4 | -115 | -391 to 273 | not distinguishable |
| final-signal-amt | final-rich-amt, reported | 17 / 3 | -484 | -972 to -103 | distinguishable |
| final-signal-amt | approve all, reported | 19 / 1 | -3,290 | -4,127 to -2,422 | distinguishable |

Before and after, the recommended setup against the baseline over 20 windows:

| | baseline | recommended | change |
|---|---:|---:|---:|
| good customers declined | 1,051 | 247 | -804, -76% |
| good customers declined or reviewed | 1,703 | 962 | -741, -44% |
| frauds stopped, of 630 | 384 | 335 | -49, -13% |
| fraud USD let through | 26,105 | 29,338 | +3,233, +12% |
| reviews, a budget of 1,000 | 729 | 804 | +75 |
| mean cost per window | 2,106 | 1,992 | -115, not distinguishable |

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 5 setups from the traces. 0 rows decided differently, largest cost gap 0.04 USD per window, 0 customer counts differ. A second bootstrap gives -464 to 238 on the primary. (Observed)

## Beyond the registration (Observed; none of it chose anything)

`fraud/dev/final_extras.py`, from the traces. These are read-outs after the result, so they don't count as answers.

- **If a declined good customer truly costs 10 USD more,** Phase 18's told cost, the recommended setup costs 2,115 against the baseline's 2,632: -517, CI -796 to -130, which excludes 0. It's also 292 below `final-signal-amt`, CI -388 to -191. The dollars favour the customer line more the more a wrongly declined customer costs. The 10 USD is an assumption, not a measurement.
- **One window carries much of the uncertainty.** In final12 the recommended setup costs 4,936 against the baseline's 2,066. Two frauds of 1,504.47 USD each were let through. Without that window the difference is -272, CI -435 to -100. Removing a bad window after seeing it isn't evidence, and such windows will happen. It says the risk is in large single frauds.

## After the test: the owner's verdict, and d1-Omni-2B with no tabular score

Owner, 2026-11-04, after seeing these results: `final-signal-cust` is "really not a decision model doing its grunt work". So **the setup that meets the owner's constraint is d1-Omni-2B with no tabular score**, not the one the registered rule recommended. The rule's result stands above as it came out.

The customer line on that setup was asked for after every FINAL result was seen, so this is a read-out and not a test (`fraud/dev/final_rich_cust.py`, `fraud/results/eval_final/rich_cust_readout.json`). No model ran. It uses `final-rich-amt`'s recorded scores, fit and lambda. Replayed with separate arithmetic: 0 rows differ, largest gap 0.02 USD. CIs are for scale. (Observed)

| 20 windows, 630 frauds | d1, rule | d1, rule and customer line | logreg baseline |
|---|---:|---:|---:|
| mean cost per window | 2,427 | 2,417 | 2,106 |
| 95% CI | 2,023–2,896 | 2,030–2,867 | 1,946–2,292 |
| good customers declined | 1,081 | 444 | 1,051 |
| good customers declined or reviewed | 1,899 | 1,327 | 1,703 |
| frauds stopped | 375 | 311 | 384 |
| fraud USD let through, of 104,651 | 30,161 | 33,508 | 26,105 |
| reviews, a budget of 1,000 | 895 | 974 | 729 |

- Against the baseline, the customer line setup is +311 per window, CI -75 to 746, logreg cheaper in 12 windows of 20. Against d1 with the rule alone: -10, CI -91 to 87.
- At a true 10 USD per declined customer, an assumption, it ties the baseline: 2,639 against 2,632, +7, CI -405 to 481.
- The prediction: about 300 good customers declined, actually 444, a miss. About 320 frauds stopped, 311, held. Cost within about 150 of 2,427, probably up: within, but down by 10.

So on these windows the customer line gives up 64 frauds and 3,347 USD of fraud stopped, for 637 fewer good customers declined, at about the same cost. (Observed; the reading is Inferred)

## What it means (Inferred)

- **The language model at heart costs nothing shown against logreg, and gains nothing shown in dollars.** The gain that holds up is to customers, and it comes from the customer line, which could also be put on logreg. Phase 18 found that it cut logreg's good customers declined by 68%.
- **The setup's AUC lead, 0.883 against 0.853, is the score's.** On FINAL rows the score wasn't read out alone, so this rests on the 0.984 correlation and Phase 21's dev result. An honest description of what's deployed: a language model that reads a boosted-tree score and mostly agrees with it.
- **A language model reading the transaction's own columns isn't yet a strong decision-maker here.** Its AUC is 0.859, and it's 321 dearer than logreg. It was the only candidate with no trees in it. That's the fairest number for the language model's own judgement, and it's one adapter on one prompt.
- **Dev's ordering failed again,** for `lm-rich` against logreg (-13 on dev, +321 here). `lm-signal`'s lead held direction and shrank in certainty (-205, distinguishable, on dev; -163, not, here).
- **The predictions partly held.** Everything dearer than on dev: a miss for logreg (2,106 against 2,125), and small for `lm-signal` (1,943 against 1,920), so test wasn't as hard as Phases 18 and 20 suggested. `lm-signal` about 100 below the baseline, not distinguishable: held at 163. The customer line within noise, at about a quarter of the good customers: held, 27%. `lm-rich` within about 150 of the baseline: a miss at 321. All p90 under 100 ms: held.

## Negative results

- **No candidate shows a distinguishable dollar saving against the baseline.** (Observed)
- **The setup with the language model's own judgement is the dearest of the three, and dearer in mean than logreg.** (Observed)
- **The customer line lets more fraud through.** 49 more frauds and 3,233 USD over 20 windows. (Observed)

## Limits

- **20 windows, 630 frauds, heavy tail.** CIs are wide: this test couldn't tell apart a saving of 391 from a loss of 273 per window. (Observed)
- **Dollars rest on assumed costs.** Phase 16: the rule wins at every friction rate tried and is no better than two lines near 50-60 USD flat per decline. (Observed)
- **C columns stay in, through the score and logreg both.** Phase 17 couldn't clear their timing. Without C, D and V costs rose by about 1,000 on logreg, so if these columns aren't available at decision time this result doesn't hold. (Observed; the reading is Inferred)
- **Lambda for the language-model setups came from 118 dev windows, logreg's from 30.** A small difference in procedure that favours neither by design. (Implementation choice)
- **Trees weren't run on FINAL,** by the limit of three and the owner's constraint. Phase 20's windows are the nearest evidence, and they aren't comparable with these. (Implementation choice)
- **Latency is this machine's.** The whole stack also needs the trees and history features at decision time, whose time isn't measured here. (Observed)
- **One adapter per setup.** A seed moved cost by about 100 on dev. (Observed)
- **The labels are IEEE-CIS's,** and costs per window aren't a business's real costs. (Inferred)

## Not run

- The signal alone on FINAL, and trees. Either would answer what the language model adds, and FINAL has now been used, so that needs new data.
- A second run of anything on FINAL.
- Delayed labels as features, which needs the owner's decision.
