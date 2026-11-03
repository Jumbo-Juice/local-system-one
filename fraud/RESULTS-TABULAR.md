# Results: a stronger tabular model, numpy boosted trees (Phase 20)

Eval `20261103-235132` (`python -m fraud.eval_tabular`). Pre-registration: the `fraud/eval_tabular.py` docstring, committed in `c12f7db` before any setup here scored a new window, with its rerun note in `45fe701`.
Numbers: `fraud/results/eval_tabular/20261103-235132.json|.md`, dev look `dev_look.json`. Model scores are uncalibrated. A review costs 10 USD. 10 new test windows, 10,000 rows, 386 frauds.

The owner's constraint, set during this phase: the final setup must be a language decision model at heart. So these trees are a reference and a possible signal, not a candidate to deploy alone.

## Short answer

- **Trees on the same columns give no shown saving over logreg.** 2,402 against 2,498: -97 USD per window, CI -385 to 182, 6 windows to 4. (Observed)
- **Dev promised much more.** -295 on held-out dev became -97, and AUC 0.914 became 0.873. Logreg fell less, 0.888 to 0.870. So the trees' lead is mostly a lead on rows close to their fit. (Observed; the reason is Inferred)
- **Trees with the extra features have the lowest mean: 2,072.** Against the same-columns trees that's -330, CI -773 to 50, not distinguishable. Against logreg it's -427, CI -680 to -176, 9 windows to 1, which excludes 0, **but this comparison was not pre-registered and I picked it after seeing the table**, so it's a lead, not a finding. (Observed)
- **The customer line again costs little or nothing here.** With trees, good customers declined fall from 595 to 113, 81% fewer, at -162 USD, CI -570 to 94, not distinguishable. Frauds stopped fall 247 → 212, fraud dollars 42,108 → 40,367, 4%. (Observed)
- **Three substantial phases in a row, 18, 19 and 20, have no distinguishable dollar saving on a pre-registered question.** The owner's stop rule would end the search for tabular cost here, and the language-model phase runs because of the owner's constraint rather than an expected saving. (Implementation choice)

## Setup (Implementation choice unless cited)

- `fraud/gbdt.py`: histogram boosting, numpy only. 800 trees, depth 6, learning rate 0.05, half the rows and 30% of the columns per tree, seed 0. Fit rows only. Card and address ids stay out.
- p's fit and two lines from Phase 14's 30 dev windows. For trees, lambda from all 118 dev windows, because on 30 it proved unstable; logreg keeps Phase 14's procedure.
- New windows: chains tabular0-9, the last of the 38.
- **The first run was stopped by a bug seen on dev.** It scored trees in float64 against bin edges from float32, which moved every row's score. Its traces on these windows were deleted unread, the fix is in `fraud/gbdt.py` with a test, and this run reproduces the dev look's 1,830 and AUC 0.9139 exactly. No test result prompted the rerun. (Observed)

## Results (Observed)

| setup, with the rule | mean cost | 95% CI | AUC | frauds stopped, of 386 | fraud USD stopped, of 56,254 | good customers declined | reviews | mean cost, two lines |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| trees, all columns + extra features | 2,072 | 1,855–2,295 | 0.874 | 243 | 43,739 | 449 | 448 | 3,155 |
| trees, all columns, customer line | 2,240 | 2,040–2,476 | 0.873 | 212 | 40,367 | **113** | 492 | |
| trees, all columns | 2,402 | 2,041–2,916 | 0.873 | 247 | 42,108 | 595 | 427 | 3,379 |
| **logreg, all columns (the base)** | **2,498** | 2,202–2,875 | 0.870 | 247 | 40,327 | 616 | 337 | 3,481 |
| trees, without C, D, V, + extra features | 2,751 | 2,352–3,162 | 0.824 | 217 | 39,568 | 788 | 366 | 4,039 |
| logreg, without C, D, V | 3,049 | 2,701–3,460 | 0.800 | 216 | 39,159 | 1,074 | 423 | 5,157 |
| approve all | 5,625 | 4,132–7,313 | n/a | 0 | 0 | 0 | 0 | |

Paired, with the rule. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| **trees, all columns** | **logreg (primary)** | 6 / 4 | -97 | -385 to 182 | not distinguishable |
| **trees + extra** | **trees, all columns (second)** | 6 / 4 | -330 | -773 to 50 | not distinguishable |
| **trees, without C, D, V** | **logreg, without C, D, V (third)** | 7 / 3 | -298 | -747 to 106 | not distinguishable |
| **trees, customer line** | **trees, all columns (fourth)** | 5 / 5 | -162 | -570 to 94 | not distinguishable |
| trees, all columns, rule | its two lines | 10 / 0 | -977 | -1,469 to -551 | distinguishable |
| trees, without C, D, V | approve all | 9 / 1 | -2,875 | -4,333 to -1,521 | distinguishable |
| trees, customer line | logreg | 8 / 2 | -259 | -552 to 13 | not distinguishable |
| trees + extra (not pre-registered) | logreg | 9 / 1 | -427 | -680 to -176 | a lead only |

Only the first four rows are pre-registered questions.

Held-out dev, 88 chains, no bar. Lambda for trees was picked with these rows in, so their read is flattered:

| scorer | rule mean | AUC | against logreg | 95% CI | good customers declined | frauds stopped |
|---|---:|---:|---:|---|---:|---:|
| logreg, all columns | 2,125 | 0.8882 | 0 | | 4,003 | 2,347 |
| trees, all columns | 1,830 | 0.9139 | -295 | -411 to -173 | 3,003 | 2,445 |
| trees + extra | 1,880 | 0.9164 | -245 | -385 to -96 | 3,150 | 2,474 |
| logreg, without C, D, V | 2,752 | 0.8289 | +626 | 475 to 780 | 7,315 | 2,019 |
| trees, without C, D, V + extra | 2,494 | 0.8626 | +369 | 217 to 522 | 5,752 | 2,105 |

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 12 setups from the traces with its own arithmetic. 0 rows decided differently, largest cost gap 0.03 USD per window (rounding), 0 customer counts differ. A second bootstrap gives -393 to 184 on the primary. (Observed)

## What it means (Inferred)

- **Dev flatters whichever model fits fine detail.** Trees lost 0.041 of AUC from dev to test, logreg 0.018. This is the third phase where dev's ordering didn't carry over (14, 18, 20). So the final test matters more than any number here, and rows further from the fit look harder for the strongest model.
- **Without the doubtful columns, trees give back something.** -298 here, -258 on held-out dev, the same direction both ways, though not distinguishable here. Logreg got nothing back from these features in Phase 19. Their cost is still above logreg with all the columns.
- **The customer line is the steadiest finding so far.** It has now cut good customers declined by 68% (logreg, Phase 18) and 81% (trees here), and on two sets of new windows its dollar difference was within noise.
- **The signal for the language model is a weaker lead than dev suggested.** It may well add less than the dev numbers imply. Phase 21 was registered to use all-columns trees, and that stays, even though extra features had the lower mean on these windows: changing it now would be choosing on test.
- **The predictions partly held.** Lower mean than logreg by less than dev's 295: held, not distinguishable. Rich ties all: a miss, its mean was lower by 330. Without C, D, V lower by about 250: held at 298. The customer line costs a little more: a miss, it's lower. A third of the customers: 19%, a stronger cut than predicted.

## Negative results

- **The primary question has no distinguishable saving,** after a stronger model and a large dev gain. (Observed)
- **A bug of mine produced a wrong run.** It was caught because a dev read disagreed with the earlier look. (Observed)
- **Three phases in a row, no distinguishable dollar saving.** (Observed)

## Limits

- **10 windows, 386 frauds, a heavy tail.** One window contributes 4,520 for trees and 3,891 for logreg. (Observed)
- **Chains, not contiguous windows.** Don't compare across phases.
- **Trees read C, whose timing Phase 17 could not clear.** (Inferred)
- **A numpy version, not LightGBM or XGBoost.** It has no tuned early stopping, and those packages may do differently. They were not tried: new packages need the owner's approval. (Implementation choice)
- **One seed, one size.** Nothing here measures how much those would move it.

## Not run

- LightGBM or XGBoost.
- Trees with a monotone constraint, or fitted on fit and dev rows together for later windows.
- Delayed labels as features, which the repo's rule rules out without the owner's decision.
