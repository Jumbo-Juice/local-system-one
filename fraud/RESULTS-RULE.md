# Results: a better use of the rule (Phase 18)

Eval `20261103-211736` (`python -m fraud.eval_rule`). Pre-registration: the `fraud/eval_rule.py` docstring, committed in `77e1734` before any setup here scored a new window.
Numbers: `fraud/results/eval_rule/20261103-211736.json|.md`, dev looks `dev_look.json`, `dev_look_combined.json`. Model scores are uncalibrated. A review costs 10 USD. 10 new test windows, 10,000 rows, 361 frauds. One scorer: logreg on all columns.

## Short answer

- **Nothing beat Phase 14's rule on dollars, and two changes that looked good on dev got worse.** Combined (primary): 2,974 against 2,423, +550, CI -16 to 1,294. Paced: 2,769, +346, CI -112 to 1,041. Neither is distinguishable, but both means went the wrong way against a dev gain of about -90. (Observed)
- **A decline line that counts customers is the useful change.** A 10 USD flat price in the decision takes good customers declined from 476 to 153, 68% fewer, for +119 USD per window, CI -2 to 245, not distinguishable. Cost: frauds stopped fall 234 → 180, but fraud dollars stopped only 41,821 → 38,757, 7%. (Observed)
- **That line is distinguishably cheaper as soon as a wrong decline carries any real flat cost.** Re-priced with its decisions unchanged, at a true 10 USD per customer: -204, CI -347 to -50; at 30 USD: -850, CI -1,093 to -606. (Observed)
- **The long-run prior does shrink the overshoot.** Sum of p 416 → 380 against 361 frauds. It lowers customers declined 476 → 400 and moves cost +58, CI -11 to 140, not distinguishable. On held-out dev, where fraud ran higher, it undershoots slightly: 3,368 against 3,409. (Observed) Isotonic and a fit with the amount don't change the level at all, and an EM prior from recent rows made cost worse on dev. (Observed)

## Setup (Implementation choice unless cited)

- Scores: logreg on all columns, refitted, matching Phase 14's dev scores. New windows: chains rule0-9.
- Fits and picks on Phase 14's 30 dev windows only, with held-out dev chains read beforehand (see the pre-registration's list of what was seen).
- A flat price changes the decision only. What a setup is charged doesn't include it, unless a row below says "true flat price".

## Results (Observed)

| setup | mean cost | 95% CI | frauds stopped, of 361 | fraud USD stopped, of 57,800 | good customers declined | reviews | fraud let through | good declined USD | review fees | sum of p |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rule-base, Phase 14's** | **2,423** | 1,964–2,879 | 234 | 41,821 | 476 | 378 | 15,980 | 4,475 | 3,780 | 416 |
| rule-longrun | 2,482 | 2,010–2,949 | 222 | 40,300 | 400 | 354 | 17,501 | 3,774 | 3,540 | 380 |
| **rule-customers, flat 10** | 2,543 | 2,026–3,066 | 180 | 38,757 | **153** | 421 | 19,043 | 2,172 | 4,210 | 416 |
| rule-paced | 2,769 | 2,096–3,454 | 238 | 40,600 | 461 | 479 | 17,201 | 5,702 | 4,790 | 416 |
| rule-combined | 2,974 | 2,222–3,748 | 209 | 38,578 | 291 | 489 | 19,222 | 5,625 | 4,890 | 417 |
| two lines | 3,600 | 2,869–4,288 | 234 | 34,465 | 351 | 464 | 23,335 | 8,028 | 4,640 | |
| approve all | 5,780 | 4,273–7,379 | 0 | 0 | 0 | 0 | 57,800 | 0 | 0 | |

Paired. "Distinguishable" means the 95% window-bootstrap CI excludes 0, the pre-registered rule.

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| **rule-base** | **rule-combined (primary)** | 6 / 4 | -550 | -1,294 to 16 | not distinguishable |
| **rule-base** | **rule-paced (second)** | 6 / 4 | -346 | -1,041 to 112 | not distinguishable |
| **rule-base** | **rule-longrun (third)** | 6 / 4 | -58 | -140 to 11 | not distinguishable |
| **rule-base** | **rule-customers (fourth)** | 7 / 3 | -119 | -245 to 2 | not distinguishable |
| rule-base | two lines | 10 / 0 | -1,177 | -1,740 to -688 | distinguishable |
| rule-customers | two lines | 9 / 1 | -1,058 | -1,638 to -576 | distinguishable |
| rule-combined | two lines | 9 / 1 | -627 | -923 to -335 | distinguishable |
| rule-combined | approve all | 9 / 1 | -2,806 | -4,513 to -1,307 | distinguishable |

Only the first four rows are pre-registered questions.

Re-priced at a true flat price per good customer declined, decisions unchanged, mean per window. In brackets, against rule-base at the same price:

| setup | 0 USD | 10 USD | 20 USD | 30 USD | 50 USD |
|---|---:|---:|---:|---:|---:|
| rule-base | 2,423 | 2,899 | 3,375 | 3,851 | 4,803 |
| rule-customers | 2,543 (+119) | 2,696 (-204) | 2,849 (-527) | 3,002 (-850) | 3,308 (-1,496) |
| rule-longrun | 2,482 | 2,882 | 3,282 | 3,682 | 4,482 |
| two lines | 3,600 | 3,951 | 4,302 | 4,653 | 5,355 |

The rule-customers gaps at 10 USD and up each have an interval excluding 0: -347 to -50, -716 to -332, -1,094 to -606, -1,864 to -1,147.

Held-out dev, 88 chains, no bar, partly used to pick these setups:

| setup | mean | against rule-base | 95% CI | good customers declined | frauds stopped | sum of p, of 3,409 |
|---|---:|---:|---|---:|---:|---:|
| rule-base | 2,125 | 0 | | 4,003 | 2,347 | 3,672 |
| rule-longrun | 2,175 | +50 | -5 to 115 | 3,312 | 2,282 | 3,368 |
| rule-paced | 2,033 | -92 | -167 to -21 | 3,676 | 2,385 | 3,672 |
| rule-combined | 2,032 | -94 | -168 to -24 | 1,961 | 2,210 | 3,674 |
| rule-customers | 2,128 | +3 | -33 to 44 | 933 | 1,973 | 3,672 |

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 7 setups from the traces with its own arithmetic, including the shift, the pacing and the flat price. 0 rows decided differently, largest cost gap 0.22 USD per window (rounding), 0 customer counts differ. A second bootstrap gives -1,288 to 16 on the primary. (Observed)

## What it means (Inferred)

- **One row explains most of the paced setup's loss.** Of its 3,458 USD extra over the 10 windows, 3,067 is a single 3,077 USD fraud that the paced rule approved after spending reviews earlier, where the base rule reviewed it. So the mean difference says little either way. It does show the risk: a rule that spends reviews faster has fewer left for a very large payment. (Observed; Inferred)
- **Dev was a poor guide for fine changes, again.** Phase 14 said the same about which scorer wins. Effects of about 100 USD per window sit inside the noise of one large fraud. So this phase chooses by large, stable effects only. Customers declined is one: a fall of 68% here and 77% on held-out dev.
- **The customer line works because the cost curve is flat near the decline line.** With 10% friction a payment near p = 1/11 costs about the same approved or declined, so moving the line costs little in expectation and removes many good customers. It mostly gives up small frauds. Frauds stopped fall 23%, their dollars 7%.
- **The overshoot is a level problem, and a long-run rate is a reasonable fix.** But it trades cost in a heavy window for cost in a light one, and it isn't shown to help cost.
- **The pre-run statement mostly missed.** "Combined has a lower mean": a miss. "Paced is lower": a miss. "Longrun closes much of the overshoot and moves cost little": held. "Customers about the same cost, about a quarter of the good customers, about 15% fewer frauds": cost held and customers about a third, frauds 23% fewer. On the means the rule-base held up better than I expected.

## Negative results

- **The primary question has no saving: combined's mean is 550 USD higher**, and it isn't distinguishable. (Observed)
- **Reviews aware of the budget gave no shown saving, and spent 27% more reviews.** (Observed)
- **No calibration tried makes the rule cheaper.** (Observed)
- **The customer line lets more fraud through:** 3,063 USD more over the 10 windows, 54 more frauds. (Observed)

## Limits

- **10 windows, 361 frauds and a heavy tail.** Approve-all's interval runs from 4,273 to 7,379. One row moved a setup by 300 USD a window. (Observed)
- **Pacing needs a window's size up front**, which a real day's traffic only estimates. (Implementation choice)
- **A flat price here is a knob, not a measured cost.** 10 USD was picked from a dev curve. What a declined customer costs is still unknown. (Implementation choice)
- **The rule still reads every column, so Phase 17's doubt applies.** (Observed)
- **Don't compare across phases.** These windows are chains and hold more fraud dollars than Phase 14's. (Observed)

## What carries forward

- rule-base stays the dollar baseline. rule-customers goes on as the customer-friendly candidate. Paced and combined are dropped. Long-run isn't carried, because it brings no shown saving and its benefit depends on which window comes next. (Implementation choice)

## Not run

- A budget share that adapts through the day, or lambda varying with the amount.
- Any fix for the very large payment risk beyond the rule's review price.
