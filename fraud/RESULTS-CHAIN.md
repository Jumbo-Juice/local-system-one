# Results: no review, a tightened cash-out rule (Phase 10)

Eval `20261102-163455` (`python -m fraud.eval_chain`, run once). Pre-registration: the `fraud/eval_chain.py` docstring, committed
before any test-split hour was read. Numbers: `fraud/results/eval_chain/20261102-163455.json|.md`; diagnosis `20261102-163455_diagnosis.json`.

## Short answer

- **Bar 1 PASS, bar 2 FAIL: the pre-registered test fails.** Chain F1 0.578 with no reviews, against 0.558 for today's rules with
  a human on 17,042 queue rows, and against the 0.90 floor. In-sample it was 0.98. (Observed)
- **The rule under test held; another rule failed.** The tightened cash-out line made 5 wrong declines. The old "round transfer"
  line made 2,178, all but 2 of them legit transfers of exactly 10,000,000 in a few hours. (Observed)
- **No clean way to re-test here** (Inferred): removing that line is a change prompted by test hours, which needs untouched
  hours, and PaySim has none left in its test split.

## Results on 282 untouched hours (Observed)

1,641,601 rows at the natural rate, 3,020 fraud rows, 1,516 chains.

| setup | reviews | chain F1 (95% over hours) | chain P | chain R | wrong declines | chains missed | row F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 17,042 | 0.5575 (0.345–0.931) | 0.3869 | 0.9967 | 2,394 | 5 | 0.7154 |
| approve | 0 | 0.5561 (0.344–0.930) | 0.3862 | 0.9934 | 2,394 | 10 | 0.4498 |
| **tight** | **0** | **0.5784 (0.352–0.989)** | 0.4081 | 0.9927 | 2,183 | 11 | 0.4636 |
| decline | 0 | 0.1438 (0.116–0.183) | 0.0775 | 0.9967 | 17,992 | 5 | 0.2510 |

Training hours for reference (in-sample): tight chain F1 0.9805, human 0.8129, tight wrong declines 30.

## Why it dropped (Observed, after the result; it explains and decides nothing)

- Of tight's 2,183 wrong declines: **5** from the tightened cash-out line, **2,178** from the round-transfer line.
- 2,176 are of exactly 10,000,000, in hours 304 (1,309), 305 (804), 376 (48), 348 (12). That line made 2 wrong declines in all 300 training hours.
- The line's training evidence was 37 of 39 round transfers being fraud. It was written from those rows, and those hours had no such burst.
- The interval over hours (0.35–0.99) is wide for the same reason: a few hours decide the score.

## Interpretation

- **Inferred:** the tightened cash-out line generalised: its wrong declines fell from 216 (today's line, same hours) to 5,
  with only 1 more chain missed than `approve`. A line resting on 39 training rows did not generalise.
- **Inferred:** with no reviews, tight still beats human on chain F1 by 0.02, which saves 17,042 reviews for no loss on this metric.
  On row F1 the human setup wins clearly (0.72 vs 0.46), as pre-stated.
- **Negative:** the 0.98 in-sample figure given to the owner did not hold. The bar existed for this.
- **Limits, unchanged:** the chain metric assumes money can still be stopped at the cash-out. The exact-amount cash-out within the hour looks
  like a PaySim generator property, so none of this shows real fraud behaves this way.

## What was not done

- No line was removed and nothing re-run on these hours. The score without the round-transfer line was not computed as a result:
  it would be post-hoc, with no untouched hours left to confirm it.
