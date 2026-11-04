# Fraud on IEEE-CIS: what to deploy (2026-11-04)

**Superseded in part, same day.** After this summary the owner judged the recommended setup "really not a decision model doing its grunt work", because it mostly follows a tree score. The setup that meets the constraint is d1-Omni-2B with no tabular score: `final-rich-amt`, 2,427 per window, or with the customer line 2,417, 444 good customers declined and 311 frauds stopped, against the baseline's 2,106, 1,051 and 384. It is +311 against the baseline, CI -75 to 746. The customer-line numbers are a read-out after the test, not a registered result. See "After the test" in `fraud/RESULTS-FINAL.md`. Training that setup on cost (Phase 25, dev only) didn't lower its cost: +53 per window, CI -39 to 142, so its recipe stays. See `fraud/RESULTS-LM-COST.md`. The text below is the summary as first written.

Final test `20261104-090649`: 20 windows reserved before this work began, run once, 20,000 rows, 630 frauds, 104,651 USD of fraud. Costs: wrong decline 10% of the amount, at least 1 USD; review 10 USD; at most 5% reviewed. Details: `fraud/RESULTS-FINAL.md`.

## What to deploy

**d1-Omni-2B, fine-tuned, reading the columns and a boosted-tree risk score, with the amount-aware rule and the customer line** (`final-signal-cust`). It was chosen by a rule fixed before the run. (Implementation choice)

Be clear about what this is: a language model that mostly agrees with a tree score. Its rank correlation with that score is 0.984 on these rows. On dev it did no better than the score alone (+67 USD per window, CI -33 to 166). It meets your terms, a language decision model that may read a score as a signal, but **nothing here shows that the language model's own judgement lowers cost**. (Observed; the reading is Inferred)

## What it costs against the Phase 14 baseline (logreg with the amount-aware rule)

| per window, FINAL | baseline | recommended | difference |
|---|---:|---:|---|
| mean cost, USD | 2,106 | 1,992 | **-115, CI -391 to 273: not distinguishable** |

No candidate shows a distinguishable saving. The registered primary, the same setup without the customer line, is -163, CI -462 to 238. (Observed)

## What it does to good customers (20 windows)

| | baseline | recommended | change |
|---|---:|---:|---:|
| good customers declined | 1,051 | **247** | -76% |
| good customers declined or reviewed | 1,703 | 962 | -44% |
| frauds stopped, of 630 | 384 | 335 | -49 |
| fraud USD let through | 26,105 | 29,338 | +3,233, 3% of all fraud dollars |

If a wrongly declined customer truly costs 10 USD more, the difference becomes -517, CI -796 to -130. That read-out is outside the registration and the 10 USD is an assumption. (Observed arithmetic)

## What the work found (Observed unless marked)

- **The rule holds up.** Weighing the amount beats two lines at every friction rate tried; it stops winning near 50-60 USD flat per decline (Phase 16).
- **C, D and V carry much of the signal.** Without them logreg costs about 1,000 more per window, and C's timing couldn't be cleared (Phase 17).
- **The customer line is the steadiest gain,** cutting good customers declined by 68–81% on three sets of windows.
- **Boosted trees promised -295 on dev and delivered -97, not distinguishable** (Phase 20). On held-out dev they still beat every language-model setup.
- **More prompt columns helped the language model on dev, and that didn't hold.** `lm-rich` tied logreg on dev, and on FINAL it's +321, CI -74 to 774, with no trees in it.
- **Latency isn't a constraint:** p90 56 ms on this GPU against a 250 ms bar. Trees and history features add time that wasn't measured.

The search stopped by your rule: Phases 18, 19 and 20 gave no distinguishable saving on a registered question. Phases 21-23 ran for your language-model requirement, not for expected savings.

## What remains unknown

- **What the language model adds over its score on new rows.** Neither the score alone nor trees ran on FINAL, by the limit of three. It needs new data, because FINAL is spent.
- **Whether C, D and V are available at decision time.** If they aren't, this result doesn't hold.
- **The true cost of a wrongly declined customer and of a review.** Every dollar here assumes them.
- **How much the result depends on large single frauds.** One window, with two frauds of 1,504 USD let through, costs 2,870 more than the baseline. Without it the difference is -272, but such windows will happen.
- **How another adapter would do.** One seed moved cost by about 100 on dev.
- **Whether LightGBM or XGBoost beats the numpy trees, and what delayed-label features would add.** The first needs a new package and the second your decision; neither was tried.

## Decisions for you

1. Is `lm-signal` close enough to "a decision model at heart", given that it follows its score? If not, `lm-rich` is the alternative, about 484 per window dearer than `lm-signal` without the customer line (CI 103 to 972).
2. Is 3,233 USD more fraud over 20 windows a fair price for 804 fewer good customers declined?

## Corrections on the record

A float bug in my trees (Phase 20) forced a rerun, its traces deleted unread. A Phase 19 commit went in with a failing test, fixed in the next. My run-time estimates were low, and the job finished about 5 hours later than first told. All are in `fraud/PROGRESS.md`.
