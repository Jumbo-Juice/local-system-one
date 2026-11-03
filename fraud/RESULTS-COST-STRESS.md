# Results: where the amount-aware rule stops winning (Phase 16)

Run `20261103-210247` (`python -m fraud.eval_cost_stress`). Pre-registration: the `fraud/eval_cost_stress.py` docstring, committed in `b19c153` before anything was priced at a new cost.
Numbers: `fraud/results/eval_cost_stress/20261103-210247.json|.md`. From Phase 14's recorded scores on its 20 test windows and 30 dev windows. No model ran, and no new window was read.

## Short answer

Each row is logreg with the rule minus logreg on two lines, USD per 1,000-row window at that cost. Negative means the rule is cheaper. (Observed)

| if you change only this | rule as deployed (not told) | rule told the new costs |
|---|---|---|
| Friction rate, 0% to 100% | **Never stops winning.** -600 at 0%, -825 at 10%, -2,700 at 100% | Wins on the mean everywhere. Not distinguishable at 0% and 1% (-36) |
| Friction minimum, 0 to 50 USD | Not distinguishable from **30 USD** (-345, CI -852 to 50). Mean flips by **50 USD** (+58) | Distinguishable everywhere tested |
| Flat price per good customer declined, on top of friction | Not distinguishable from **30 USD** (-335, CI -862 to 86). Mean flips between **50 and 60 USD** (-8, +156). Distinguishably dearer by 100 USD (+810, CI 151 to 1,427) | Distinguishable everywhere tested, up to 200 USD |
| Flat price alone, no percentage | Not distinguishable from **20 USD** (-290, CI -756 to 73). Mean flips by **50 USD** (+201). Distinguishably dearer by 75 USD (+610, CI 46 to 1,102) | Wins on the mean everywhere. Not distinguishable at 1 USD (-36) |
| Review fee, 1 to 200 USD | **Never stops winning.** The gap widens: -716 at 1 USD, -1,914 at 100 USD | Not distinguishable from **75 USD** (-151, CI -434 to 115), still lower on the mean |
| Review share, 0.5% to 20% | Not distinguishable at **0.5%** (-135, CI -371 to 84); distinguishable from 1% up | Distinguishable everywhere tested |

- **The rule's real risk is a price per customer, not a percentage.** Its declined dollars are fewer than the lines' (89,680 against 131,400 USD of good payments), so a higher friction rate helps it. But it declines more customers (982 against 655), so anything charged per customer counts against it. (Inferred from the two Observed counts)
- **The break-even is about 50 USD per good customer declined**, on top of 10% friction, if the rule keeps today's settings. It stops being a shown win at about 30 USD. (Observed)
- **Telling the rule the real cost removes that risk.** Re-fitted to the true cost, it has the lower mean at every cost set tested, and it declines far fewer customers: 96 of them at a 20 USD flat price, against 982. (Observed) So the risk is in guessing the cost wrong and leaving it wrong, not in the rule itself. (Inferred)

## Setup (Implementation choice unless cited)

- Scores: `ieee-logreg-all` and `ieee-ft-ieee-full`, recorded on the laptop, in eval `20261103-164830`. No scores from this machine, so two machines are never mixed.
- As deployed: Phase 14's decisions stay fixed (rule at lambda 10 assuming 10% friction; lines 0.0586 and 0.1141); only the prices differ.
- Re-picked: each setup knows the costs. Lines come from the 30 dev windows, and so do lambda and the rule's expected costs. The logistic fit of p doesn't use costs, so it stays. Lines here come from all 30 dev windows, so the base row reads -818 where as-deployed reads -825.
- 66 cost sets across six families, and a grid of friction rate by flat price, tested without correcting for multiple comparisons. So this is a map, not 66 findings. Phase 14's test windows have been read before, and nothing here was chosen on them.

## Results (Observed)

Logreg as deployed, the flat-price row by row. Customers stay at 982 and 655 in every row, because the decisions don't change.

| flat price per good customer declined | rule minus lines | 95% CI | windows cheaper of 20 |
|---:|---:|---|---:|
| 0 | -825 | -1,307 to -464 | 18 |
| 10 | -662 | -1,158 to -283 | 18 |
| 20 | -498 | -1,012 to -100 | 14 |
| 30 | -335 | -862 to 86 | 12 |
| 50 | -8 | -572 to 464 | 8 |
| 60 | +156 | -423 to 656 | 8 |
| 100 | +810 | 151 to 1,427 | 6 |

Logreg told the costs, what the customers do. Lines and rule are each re-picked on dev:

| flat price | rule minus lines | good customers declined, rule | good customers declined, lines | lambda |
|---:|---:|---:|---:|---:|
| 0 | -818 | 982 | 660 | 10 |
| 1 | -866 | 696 | 660 | 5 |
| 5 | -682 | 448 | 310 | 10 |
| 10 | -682 | 256 | 152 | 10 |
| 20 | -746 | 96 | 152 | 10 |
| 50 | -799 | 28 | 29 | 20 |
| 100 | -824 | 17 | 29 | 20 |

The adapter, on the same scale. It has less room, as Phase 14 found:

| | as deployed | told the costs |
|---|---|---|
| Phase 14's costs | -397, CI -937 to 14, not distinguishable | -420, CI -964 to -5 |
| Friction rate where the mean flips | between 20% and 30% | never in the range |
| Friction rate where it's distinguishably dearer | 50% (+798, CI 38 to 1,477) | never in the range |
| Flat price where the mean flips | between 10 and 15 USD | never in 0-200, except a tie at 60 (+16) |

Friction rate by flat price, logreg as deployed. A lower friction rate leaves less room for a flat price, and the mean flips near 50 USD at most rates:

| rate | flat 0 | flat 10 | flat 20 | flat 50 | flat 100 |
|---:|---:|---:|---:|---:|---:|
| 1% | -630 | -467 | -303 (n.d.) | +187 (n.d.) | +1,005 |
| 5% | -720 | -557 | -393 | +97 (n.d.) | +915 |
| 10% | -825 | -662 | -498 | -8 (n.d.) | +810 |
| 20% | -1,033 | -870 | -706 | -216 (n.d.) | +602 (n.d.) |
| 50% | -1,658 | -1,495 | -1,331 | -841 | -23 (n.d.) |

n.d. = the interval includes 0.

Checked apart from the report code: `fraud/dev/verify_cost_stress.py` imports no cost, rule or report code. It re-prices the traces with its own arithmetic and budget replay, and re-derives the rule from each header's recorded fit. 264 cost-set views, 0 differ by more than rounding. A separate bootstrap with another seed gives -1,297 to -468 at Phase 14's costs. (Observed)

## What it means (Inferred)

- **Which cost you are unsure of matters more than how unsure.** A business that loses a percentage of what it declines has no reason to fear the rule. A business that loses a customer's lifetime value on a wrong decline does.
- **A flat cost is why customers matter here.** At 30 USD or more per customer the rule's extra 327 declines cost about what its fraud saving brings in. Phase 18 tests a decline line that counts customers, on new windows.
- **A dearer review helps the rule as deployed** because it uses fewer reviews (702 against 944). Told a very high fee, both setups stop reviewing and the rule's edge goes, which fits a rule whose gain mostly comes from choosing what to review.
- **A very small review budget is the one budget where it isn't shown to win as deployed.** At 0.5%, lambda 10 asks for far more reviews than exist, first come, first served. Told the budget, it's -499.
- **The predictions held, with one miss.** (1) a higher friction rate favours the rule: held. (2) A flat price near 50 USD breaks even, and it stops being distinguishable earlier: held, at 30. (3) Re-picked wins further out, possibly everywhere: held on the mean. (4) A dearer review narrows the gap: held only re-picked; as deployed the gap widens, which is a miss.

## Negative results

- **As deployed, at a flat price of 100 USD or more per customer, the rule is distinguishably dearer than the lines.** (Observed)
- **The adapter's rule is not shown to win as deployed at any friction rate of 10% or more**, and it is distinguishably dearer at 50% and up. (Observed)
- **Told a friction of 0-1%, the rule's advantage is too small to show** (-36 for both). Almost anything suspicious is simply declined, for both setups, 8,300 to 9,400 good customers. (Observed)

## Limits

- **These are the same 20 windows as Phase 14**, with its heavy tail: the 10 biggest row savings were 61% of the saving there. Nothing was chosen on them here, but every row of these tables shares that luck. (Inferred)
- **Reviews still find the truth, at a flat fee, with no delay.** A friction curve that isn't one of these shapes wasn't tried. (Implementation choice)
- **The cost of a wrong decline is unknown, and this doesn't measure it.** It maps what follows from each guess. The owner's business is the only source for the real number. (Inferred)
- **Same period, as before.** The windows come from the same stretch of time as all the others. (Observed)

## Not run

- A flat price that varies with the amount or the customer's history.
- Any change to which scores the rule reads. Phases 17-20 test that.
