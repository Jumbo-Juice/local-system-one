# Results: history and category features for logreg (Phase 19)

Eval `20261103-212822` (`python -m fraud.eval_features`). Pre-registration: the `fraud/eval_features.py` docstring, committed in `4826ea2` before any setup here scored a new window.
Numbers: `fraud/results/eval_features/20261103-212822.json|.md`, dev look `dev_look.json`. Model scores are uncalibrated. A review costs 10 USD. 10 new test windows, 10,000 rows, 333 frauds.

## Short answer

- **History and more categories give logreg no shown saving.** Rich features 2,725 against 2,776: -50 USD per window, CI -176 to 60, 5 windows each way. With the D1-keyed history 2,679: -97, CI -338 to 164. Neither is distinguishable. AUC barely moves: 0.8429, 0.8424, 0.8437. (Observed)
- **They don't replace the doubtful columns.** Without C, D and V, adding them changes the cost by +23, CI -333 to 553, though AUC rises from 0.779 to 0.799. It is still dearer than all columns by 604, CI 257 to 981. Dev's -196 didn't carry over. (Observed)
- **Customers don't change.** Good customers declined 501 → 517, frauds stopped 199 → 201. (Observed)

## Setup (Implementation choice unless cited)

- Keys, from columns the row holds: card = card1 card2 card3 card5 card4 card6; card_addr adds addr1; addr = addr1 addr2; card_email = card1 P_emaildomain; device = DeviceInfo; uid = card1 addr1 and the day D1's counter began. There are no card numbers in the data.
- For each key: seconds since the last payment, counts in the last hour, day and 7 days, amount against the key's past. `steady` leaves out lifetime counts, because their level keeps rising with the row number. `extra`: cents and calendar shape. `identity`: `train_identity.csv`'s numeric columns, missing flags and top categories from fit rows only. `inter`: product by amount and by velocity.
- No label is an input: a test flips every label and the features don't move. A test also changes later rows and earlier rows' history doesn't move. FINAL rows are read as history for later rows, unscored and with no label read.
- Logreg refitted on fit rows; rule and lines from Phase 14's 30 dev windows. New windows: chains features0-9.

## Results (Observed)

| setup, with the rule | mean cost | 95% CI | AUC | frauds stopped, of 333 | fraud USD stopped, of 57,030 | good customers declined | reviews | mean cost, two lines |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| all + rich + interactions + uid | 2,679 | 1,973–3,477 | 0.8437 | 205 | 39,221 | 477 | 468 | 3,340 |
| all + rich | 2,725 | 2,063–3,483 | 0.8424 | 201 | 38,407 | 517 | 363 | 3,310 |
| **all (the base)** | **2,776** | 2,123–3,511 | 0.8429 | 199 | 37,750 | 501 | 359 | 3,315 |
| without C, D, V | 3,357 | 2,613–4,286 | 0.7789 | 150 | 32,771 | 726 | 428 | 4,259 |
| without C, D, V, + rich + interactions | 3,379 | 2,533–4,357 | 0.7992 | 160 | 32,390 | 684 | 398 | 4,200 |
| approve all | 5,703 | 4,190–7,348 | n/a | 0 | 0 | 0 | 0 | |

Paired, with the rule. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| **all + rich** | **all (primary)** | 5 / 5 | -50 | -176 to 60 | not distinguishable |
| **all + rich + uid** | **all (second)** | 6 / 4 | -97 | -338 to 164 | not distinguishable |
| **without C, D, V** | **the same + rich (third)** | 3 / 7 | -23 | -553 to 333 | not distinguishable |
| **all** | **without C, D, V + rich (fourth)** | 8 / 2 | -604 | -981 to -257 | distinguishable |
| all, rule | all, two lines | 9 / 1 | -540 | -816 to -254 | distinguishable |
| without C, D, V + rich | approve all | 9 / 1 | -2,324 | -3,401 to -1,286 | distinguishable |

Only the first four rows are pre-registered questions.

Held-out dev, 88 chains, no bar:

| scorer | rule mean | AUC | against all | 95% CI | good customers declined | frauds stopped |
|---|---:|---:|---:|---|---:|---:|
| all | 2,125 | 0.8882 | 0 | | 4,003 | 2,347 |
| all + rich | 2,071 | 0.8893 | -54 | -116 to 0 | 3,919 | 2,353 |
| all + rich + interactions + uid | 2,044 | 0.8903 | -81 | -176 to 8 | 3,655 | 2,381 |
| without C, D, V | 2,752 | 0.8289 | +626 | 475 to 780 | 7,315 | 2,019 |
| without C, D, V + rich + interactions | 2,555 | 0.8460 | +430 | 291 to 570 | 6,320 | 2,065 |

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 11 setups from the traces with its own arithmetic. 0 rows decided differently, largest cost gap 0.03 USD per window (rounding), 0 customer counts differ. A second bootstrap gives -180 to 60 on the primary. (Observed)

## What it means (Inferred)

- **A linear model can't do much with these.** A velocity count means little until it's crossed with product or amount, and only a few crosses were tried. So this is a verdict on logreg, not on the signal. Phase 20 tests the same columns in trees.
- **Without C, D and V, a logistic model isn't helped by rows nearby.** So the doubt's cost from Phase 17 stands for this family of models: 604 USD a window here even with the extra columns.
- **The direction is steady but too small to matter.** Both reads have rich features lower by about 50 USD, about 2%. A saving that size can't be told apart from noise on 10 windows and wouldn't change a decision.
- **Lifetime counts were dropped on an argument and a dev look.** A later period would test that argument. All of test is at most 40-80 days later than dev.
- **The pre-run statement mostly held.** Lower mean, not distinguishable: held for both. "Gets back about a third": a miss, nothing came back on new windows.

## Negative results

- **The primary question has no distinguishable saving.** (Observed)
- **Nothing here gives back what losing C, D and V costs,** on new windows. (Observed)
- **Two substantial phases in a row, 18 and 19, have given no distinguishable dollar saving.** The owner's stop rule also asks that what's left be unlikely to change that. It isn't met yet: a tree model reads these same columns differently, so Phase 20 runs. (Implementation choice)

## Limits

- **Keys are rough.** Phase 17's look suggests a same-card key mixes real cards.
- **id_ columns are anonymised too, and weren't audited.** Their meaning is unverified.
- **10 windows, 333 frauds, heavy tail.** Don't compare across phases.

## Not run

- Labels of earlier rows as features, even delayed ones. The repo's rule says no prior `isFraud` is an input (`fraud/DATA-IEEE.md`). Fraud clusters by card in this data, so a version respecting a realistic label delay may be the largest untested gain, and it's the owner's decision. (Inferred)
- Crosses beyond product by amount and velocity.
