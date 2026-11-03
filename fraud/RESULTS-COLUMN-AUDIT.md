# Results: what the C, D and V doubt could cost (Phase 17)

Eval `20261103-210600` (`python -m fraud.eval_column_audit`). Pre-registration: the `fraud/eval_column_audit.py` docstring, committed in `c3deb2a` before any setup here scored a new window.
Numbers: `fraud/results/eval_column_audit/20261103-210600.json|.md` and `leak.json`. Model scores are uncalibrated. A review costs 10 USD. 8 new test windows, 8,000 rows, 244 frauds.

## Short answer

- **If none of the C, D and V columns are known at decision time, logreg with the rule costs 50% more:** 1,999 → 3,001 USD per window, cheaper in 8 of 8 windows with all columns, -1,002 USD, 95% CI -1,870 to -405. Distinguishable. On the 88 held-out dev chains it is +626, CI 475 to 780, or 29%. (Observed) So the doubt is worth **roughly 600 to 1,000 USD per window**.
- **That worst case isn't shown to beat approve-all:** 3,001 against 3,649, -649, CI -1,423 to 298. It stops only 108 of 244 frauds and declines 865 good customers, against 142 and 391. (Observed)
- **The rule still beats two lines with those columns gone:** 3,001 against 3,807, -806, CI -1,339 to -298. So Phase 14's rule isn't what the doubt puts at risk. (Observed)
- **C is the group that matters most, and it is also the one that looks most doubtful.** Losing it costs +255 (CI 86 to 479) here and +159 (80 to 236) on held-out dev. In the leak look, 10 of 14 C columns lean on a card's later rows more than its earlier ones, though weakly. (Observed; the suspicion is Inferred)
- **Losing V or D alone isn't shown to cost anything on the new windows** (V -5, CI -278 to 224; D +92, CI -142 to 401). Held-out dev, with more windows, sees V at +221 (145 to 299) and D at +2 (-81 to 90). So V is worth a little and D about nothing, with C and D able to stand in for each other. (Observed; Inferred)

## Setup (Implementation choice unless cited)

- Logreg on fit rows, as `ieee-logreg-all`, with a group and its missing flags removed. `audit-all` matched Phase 14's dev scores, largest gap 6.5e-9, as the script required before anything else ran. Fits took 11-52 s here.
- Each set gets its own logistic fit, lambda and two lines from Phase 14's 30 dev windows. Nothing else was chosen. Lambda came out 10 in every set except D and D, V, at 5.
- New windows: chains audit0-7 from `fraud/final_test.py`, 1,000 rows each, in arrival order, skipping rows other windows used. Each ran once per setup. Second read: 88 chains of dev rows that fitted nothing here.

## Results (Observed)

| columns left out | mean cost, rule | 95% CI | AUC | frauds stopped, of 244 | fraud USD stopped, of 29,195 | good customers declined | reviews | mean cost, lines |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| V | 1,994 | 1,635–2,409 | 0.854 | 144 | 19,443 | 421 | 262 | 3,050 |
| **none** | **1,999** | 1,629–2,375 | 0.871 | 142 | 19,016 | 391 | 228 | 2,785 |
| M (not asked) | 2,027 | 1,671–2,378 | 0.875 | 143 | 19,156 | 408 | 245 | 2,765 |
| D | 2,091 | 1,624–2,525 | 0.864 | 142 | 18,441 | 348 | 345 | 2,732 |
| D, V | 2,240 | 1,751–2,757 | 0.830 | 137 | 17,612 | 370 | 346 | 3,438 |
| C | 2,254 | 1,978–2,555 | 0.850 | 129 | 17,189 | 411 | 258 | 3,079 |
| C, D | 2,372 | 2,016–2,722 | 0.838 | 127 | 17,079 | 485 | 248 | 3,652 |
| C, V | 2,576 | 2,071–3,141 | 0.803 | 119 | 15,341 | 453 | 319 | 3,507 |
| **C, D, V** | **3,001** | 2,298–3,755 | 0.759 | 108 | 15,400 | 865 | 320 | 3,807 |
| approve all | 3,649 | 3,016–4,321 | n/a | 0 | 0 | 0 | 0 | |

Paired, with the rule, against all the columns. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| left out | new windows: cost added | 95% CI | | held-out dev: cost added | 95% CI |
|---|---:|---|---|---:|---|
| **C, D, V (primary)** | **+1,002** | 405 to 1,870 | distinguishable | +626 | 475 to 780 |
| V (second) | -5 | -278 to 224 | not distinguishable | +221 | 145 to 299 |
| C (second) | +255 | 86 to 479 | distinguishable | +159 | 80 to 236 |
| D (second) | +92 | -142 to 401 | not distinguishable | +2 | -81 to 90 |
| C, D | +373 | 108 to 823 | distinguishable | +235 | 145 to 325 |
| C, V | +577 | 170 to 1,025 | distinguishable | +424 | 306 to 538 |
| D, V | +241 | -124 to 616 | not distinguishable | +343 | 186 to 505 |
| M | +28 | -29 to 94 | not distinguishable | -14 | -76 to 47 |

Only the first four rows are pre-registered questions, along with two more: C, D, V with the rule against its lines (-806, CI -1,339 to -298, distinguishable) and against approve-all (-649, CI -1,423 to 298, not distinguishable). The rest are not corrected for multiple comparisons. Against its lines, all columns: -786, CI -1,103 to -411. On held-out dev: the rule beats its lines in every set by -960 to -1,682, each interval excluding 0.

### The leak look (Observed; feature columns only, no labels; not a test)

For each column, its rank regressed on the rank of a same-card key's earlier rows, its later rows, and time, over fit and dev rows. The controls were added after the pre-registration, to read the look at all, and don't touch any score: a count that is past-only by construction scores earlier 0.68, later 0.02, and a future-only one scores earlier 0.21, later 0.49.

| group | columns | median size on earlier rows | median size on later rows | share leaning later than earlier | largest later-leaning |
|---|---:|---:|---:|---:|---|
| C | 14 | 0.095 | 0.123 | 71% | C9 (-0.16, 0.16), C5 (-0.14, 0.16), C13 (-0.08, 0.14) |
| D | 15 | 0.041 | 0.034 | 40% | D1 (-0.07, 0.14) |
| V | 339 | 0.115 | 0.068 | 19% | V285 (-0.02, 0.09), V310 (-0.03, 0.09) |
| card3, card5, dist | 4 | 0.096 | 0.027 | 25% | none above 0.1 |

Within the same key, C1 falls between consecutive rows in 25% of pairs, and D1 moves within a day of the elapsed time in 22% of pairs. So this key isn't one real card, or these columns aren't plain running counts.

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays every decision and the review budget from the traces with its own arithmetic. 19 setups, 8 windows: 0 rows decided differently, largest cost gap 0.03 USD per window (rounding), 0 customer counts differ. A second bootstrap with another seed gives -1,880 to -400 on the primary. (Observed)

## What it means (Inferred)

- **Use the bound, not the point.** Eight windows give the headline figure a wide interval. The two reads put the cost of losing all three groups at about 30-50%, so a deployed number that uses them should carry that as its known risk until someone at the data's source confirms what the columns mean.
- **The C result is a warning, not a finding of leakage.** The look's largest loadings are about a third of the future-only control's, and the key is rough. A C column could lean later simply because a card that will be used more is a card in more active use today. But the same columns give logreg the most, so this is the doubt worth settling first.
- **What you stand to lose is mostly fraud caught, and customers.** Without C, D and V, frauds stopped fall 142 → 108 and good customers declined more than double, 391 → 865.
- **V's 339 columns add little on top of C and D here.** AUC falls from 0.871 to 0.854, and the cost doesn't move on new windows. A setup with V dropped would have 58% fewer features, at +221 USD per window on held-out dev. Whether that trade is worth it depends on why they're doubted.
- **The predictions mostly held.** Distinguishably dearer, by 20-50%: held, at 50% here and 29% on held-out dev. Dropping V alone costs little, C more than D: held. **Beats approve-all: a miss**, lower on the mean but not shown. **Leak look: a miss**, predicted C and D to lean on past rows, and C leans later.

## Negative results

- **The leak look cannot clear C, and the data can't settle this.** (Observed; Inferred)
- **Without C, D and V, the model isn't shown to be better than doing nothing**, on these 8 windows. (Observed)
- **8 windows and 244 frauds are few.** Several rows may simply lack the power to show a difference. Held-out dev's row for V differs from the new windows' by more than 200 USD, inside both intervals only barely. (Observed)

## Limits

- **These are bounds within logistic regression.** A stronger model may lose more or less from the same columns. Phase 19 and Phase 20 report their own sets without C, D and V where it matters. (Implementation choice)
- **New windows are chains, not contiguous.** A chain spans more time than 1,000 rows would. Cost per window in earlier phases may differ slightly for that reason, so don't compare across phases. (Observed; Inferred)
- **Held-out dev isn't test.** The p fit and lines come from nearby dev rows, so it may read kinder than a later period. (Inferred)
- **The meaning of the columns is still from memory of the data page, not verified.**

## Not run

- Dropping single C columns, or the V blocks one by one.
- Asking Vesta or reading the data page, which isn't reachable by search and is out of scope for this phase.
