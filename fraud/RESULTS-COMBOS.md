# Results: combinations on dev, and what goes to the final test (Phase 22)

Eval `20261104-080932` (`python -m fraud.eval_combos`). How candidates are chosen was stated in the script's docstring and committed in `7890284`, before `lm-signal`'s scores existed. One interim run on the three other adapters came first, as a check of the code, and is recorded in that docstring.
Numbers: `fraud/results/eval_combos/20261104-080932.json|.md`. Dev only: 88 held-out chains, 88,000 rows, 3,409 frauds. No new window was used, because none is left. Model scores are uncalibrated. A review costs 10 USD. Lambda comes from all 118 dev windows except logreg's, so every read here is flattered.

## Short answer

- **Going to the final test: `lm-signal` with the rule, `lm-signal` with the customer line, `lm-rich` with the rule,** against logreg with Phase 14's rule. (Implementation choice, by the stated rule)
- **No combination beat its cheapest member by a distinguishable amount, except by adding the tabular score to `lm-rich`.** Averaging adapters gave nothing shown: the two seeds are 54 dearer than the better seed, CI -38 to 138, and the three adapters are 8 dearer than `lm-rich`, CI -95 to 101. (Observed)
- **Adding the tabular score to `lm-rich` at the rule saves 244, CI -341 to -160, but the score then carries most of the weight.** The language model has 25% of the weight with `lm-rich` and 42% with `lm-signal`. Both are under the 50% line, so neither can go on. (Observed; the line is an Implementation choice)
- **The customer line costs money on dev, and saves a lot of customers.** On `lm-signal`, good customers declined fall from 3,241 to 827, 74% fewer, for +63 USD per window, CI 18 to 111. That's distinguishable, unlike on Phases 18 and 20's test windows. (Observed)

## Results (Observed)

| setup | mean cost | 95% CI | AUC | frauds stopped, of 3,409 | fraud USD stopped | good customers declined | reviews | lambda | p90 ms | language model's share |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trees, reference | 1,830 | 1,693–1,982 | 0.9139 | 2,445 | 484,173 | 3,003 | 3,665 | 5 | n/a | none |
| lm-rich + trees, fitted | 1,868 | 1,732–2,017 | 0.9119 | 2,441 | 481,907 | 2,982 | 3,736 | 5 | 59 | 0.25, fails |
| lm-signal + trees, fitted | 1,898 | 1,753–2,059 | 0.9110 | 2,441 | 480,977 | 2,977 | 3,827 | 5 | 62 | 0.42, fails |
| **lm-signal** | **1,920** | 1,780–2,074 | 0.9102 | 2,410 | 475,508 | 3,241 | 3,039 | 10 | 62 | |
| lm-rich and lm-signal, mean | 1,923 | 1,777–2,082 | 0.9087 | 2,414 | 480,539 | 3,191 | 3,876 | 5 | 121 | |
| **lm-signal, customer line** | **1,983** | 1,829–2,153 | 0.9102 | 2,049 | 452,284 | 827 | 3,462 | 10 | 62 | |
| lm-rich + trees, customer line | 1,998 | 1,843–2,171 | 0.9119 | 2,118 | 459,614 | 786 | 4,088 | 5 | 59 | 0.25, fails |
| lm-signal + trees, customer line | 2,037 | 1,870–2,221 | 0.9110 | 2,101 | 456,565 | 759 | 4,120 | 5 | 62 | 0.42, fails |
| lm-rich and lm-signal, customer line | 2,044 | 1,877–2,227 | 0.9087 | 2,100 | 457,664 | 840 | 4,190 | 5 | 121 | |
| **lm-rich** | **2,112** | 1,945–2,294 | 0.8932 | 2,324 | 465,486 | 3,459 | 3,855 | 5 | 59 | |
| lm-phase12, lm-seed1 and lm-rich, mean | 2,120 | 1,976–2,277 | 0.8938 | 2,276 | 465,909 | 3,163 | 4,033 | 5 | 170 | |
| logreg, the baseline | 2,125 | 1,980–2,277 | 0.8882 | 2,347 | 461,773 | 4,003 | 2,997 | 10 | n/a | none |
| lm-rich, customer line | 2,226 | 2,044–2,428 | 0.8932 | 1,996 | 442,092 | 945 | 4,196 | 5 | 59 | |
| lm-phase12 and lm-seed1, mean | 2,240 | 2,093–2,394 | 0.8824 | 2,186 | 449,438 | 3,362 | 3,223 | 10 | 111 | |
| the three adapters, customer line | 2,254 | 2,097–2,417 | 0.8938 | 1,946 | 441,662 | 821 | 4,261 | 5 | 170 | |
| the two seeds, customer line | 2,294 | 2,141–2,455 | 0.8824 | 1,840 | 427,359 | 789 | 3,683 | 10 | 111 | |

A mean's p90 is the sum of its members' (Inferred: they'd run one after another).

Paired, on 88 chains, no bar:

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| lm-signal | logreg | 65 / 23 | -205 | -299 to -104 | distinguishable |
| lm-rich | logreg | 51 / 37 | -13 | -142 to 125 | not distinguishable |
| lm-rich and lm-signal, mean | logreg | 64 / 24 | -202 | -308 to -94 | distinguishable |
| lm-rich + trees | logreg | 70 / 18 | -257 | -349 to -159 | distinguishable |
| trees | lm-signal | 52 / 36 | -90 | -175 to -9 | distinguishable |
| trees | lm-rich and lm-signal, mean | 51 / 37 | -93 | -206 to 18 | not distinguishable |
| trees | lm-rich + trees | 50 / 38 | -38 | -141 to 64 | not distinguishable |
| lm-signal, no customer line | lm-signal, customer line | 45 / 43 | -63 | -111 to -18 | distinguishable |
| lm-rich, no customer line | lm-rich, customer line | 49 / 39 | -114 | -180 to -56 | distinguishable |

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 16 setups over 88 chains from the traces. 0 rows decided differently, largest cost gap 0.04 USD per window, 0 customer counts differ. (Observed)

## What it means (Inferred)

- **`lm-signal`'s lead over logreg comes from the score it reads.** Phase 21 shows that: rank correlation 0.984 with its signal, and no gain on it. So `lm-rich` is the fair price of a language model with no trees in it, and it about ties logreg on dev.
- **The customer line's price looks larger on dev than it did on test.** Two sets of test windows showed no distinguishable cost, 88 dev chains do. With more windows a small cost becomes visible. On `lm-signal`, across 88 chains, the line stops 361 fewer frauds and declines 2,414 fewer good customers.
- **Averaging adapters isn't worth its latency here.** Two seeds gain 0.001–0.003 of AUC on their members, at twice the time, and no distinguishable dollars.
- **The predictions partly held.** `lm-signal` the cheapest setup passing the line, close to the trees: held, 90 dearer. A fit leaning on trees: held. Means gain 0.005–0.01 of AUC: a miss, 0.001–0.003, and the `lm-rich`/`lm-signal` mean lost AUC. The customer line at no distinguishable cost: a miss on dev.

## Negative results

- **The trees stay ahead of every setup that passes the line.** They're 90 cheaper than `lm-signal`, CI -175 to -9, and 93 cheaper than the `lm-rich` and `lm-signal` mean, CI -206 to 18, which isn't distinguishable. (Observed)
- **Means of adapters don't help.** No mean is distinguishably cheaper than its cheapest member. (Observed)

## Limits

- **Dev only.** The ordering on dev has failed to carry over in Phases 14, 18 and 20. (Observed)
- **The 50% line is my own definition of "at heart".** Standardised weight in a logistic fit is one reasonable measure, not the only one. An owner who reads it differently might accept `lm-rich` + trees, which on dev is 244 cheaper than `lm-rich` and 52 cheaper than `lm-signal`. It would need its own test, because FINAL runs once on the three chosen here. (Implementation choice)
- **`lm-signal` passes the line because it has no fit to weigh, yet it mostly follows its score.** The line doesn't catch that, Phase 21's report does. (Inferred)

## Not run

- Trees as a candidate on FINAL, ruled out by the owner's constraint and the limit of three.
- Weighted means of adapters, which would need more choosing on the same dev rows.
