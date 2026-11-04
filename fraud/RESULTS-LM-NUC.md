# Results: the language model on this machine's GPU (Phase 21)

Eval `20261104-090217` (`python -m fraud.eval_lm_nuc --report`). Pre-registration and two owner amendments: the `fraud/eval_lm_nuc.py` docstring, committed in `a732720`, `2e5c700` and `26408f4`, before any new adapter was scored.
Numbers: `fraud/results/eval_lm_nuc/20261104-090217.json|.md`. Dev only: 88 held-out chains, 88,000 rows, 3,409 frauds. No new window was used, because none is left. Model scores are uncalibrated. A review costs 10 USD. Lambda comes from all 118 dev windows except logreg's, so every read here is flattered.

The owner's constraint, set during Phase 20: the final setup must be a language decision model at heart, and it may read a tabular score as a signal. So no setup here had to beat the trees to go on. The two cheapest on held-out dev went on.

## Short answer

- **More prompt columns made a real gain.** `lm-rich` is 176 USD per window cheaper than Phase 12's recipe, CI -281 to -58, and AUC rises 0.880 → 0.893. It catches logreg: -13, CI -142 to 125, not distinguishable, with 14% fewer good customers declined (3,459 against 4,003). (Observed)
- **`lm-signal` is the cheapest language-model setup: 1,920, against logreg's 2,125 (-205, CI -299 to -104).** (Observed)
- **But `lm-signal` adds nothing shown over the score it reads.** Against the signal alone on the same rows and rule it's 67 dearer, CI -33 to 166, not distinguishable. Rank correlation with the signal is 0.984, and 2,233 of 88,000 decisions differ. **On dev it mostly repeats its signal, and this isn't a language-model win.** (Observed; the reading is Inferred)
- **No language-model setup beats the trees.** The nearest, `lm-signal`, is 90 dearer, CI 9 to 175. (Observed)
- **A second seed moves cost by about 100,** -102, CI -205 to 6. That's a useful yardstick for how much one run's adapter differs from another's. (Observed)
- **Frozen d1-Omni-2B is far behind:** 3,697, AUC 0.754. Fine-tuning is worth 1,409 USD per window, CI 1,188 to 1,641. (Observed)
- **Every setup is well inside the latency bar:** p90 54–62 ms against 250 ms, on this GPU. (Observed)

## Setup (Implementation choice unless cited)

- d1-Omni-2B, option order `rev` as Phase 12 chose. LoRA, one pass, the recipe of Phase 12. All training on this RTX 3060 Ti in bf16.
- `lm-phase12`: Phase 12's adapter, re-scored here, so no two machines mix. `lm-seed1`: seed 1, a different draw of legit rows. `lm-rich`: adds six counters, five day columns, distance, and what earlier rows say about the card and address, no label. `lm-signal`: the same, plus one line with the trees' score (`fraud/lm_signal.py`, 400 trees, out of fold on training rows, from all fit rows on dev). `lm-frozen`: no adapter. One prompt per option, written before any model read it.
- Withdrawn by the owner before it was scored: two passes.
- Larger models: none is on disk, and each has needed its own reader, readout and reproduction check. Not run. (Implementation choice)
- Scored in batches of 16, 118,000 dev rows per setup. Latency on lone prompts, 950 rows of dev0, with the CPU free.

Run times here (Observed): training 84.5 min for the plain prompt, 123 min for rich, 134.6 min for signal, peak 3.8–4.1 GiB. Scoring 54 min for the plain prompt, 81–85 min for the longer ones. My first estimates were 85 and 20 min. No run passed 8 hours.

## Results (Observed)

| scorer, with the rule | mean cost | 95% CI | AUC | frauds stopped, of 3,409 | fraud USD stopped | good customers declined | reviews | mean cost, two lines | lambda | p90 ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trees, reference | 1,830 | 1,693–1,982 | 0.9139 | 2,445 | 484,173 | 3,003 | 3,665 | 2,972 | 5 | n/a |
| the signal alone (trees at 400) | 1,853 | 1,718–1,999 | 0.9108 | 2,452 | 483,377 | 3,052 | 3,708 | 2,829 | 5 | n/a |
| **lm-signal** | **1,920** | 1,780–2,074 | 0.9102 | 2,410 | 475,508 | 3,241 | 3,039 | 3,063 | 10 | 62 |
| **lm-rich** | **2,112** | 1,945–2,294 | 0.8932 | 2,324 | 465,486 | 3,459 | 3,855 | 3,093 | 5 | 59 |
| logreg, the baseline | 2,125 | 1,980–2,277 | 0.8882 | 2,347 | 461,773 | 4,003 | 2,997 | 3,085 | 10 | n/a |
| lm-seed1 | 2,186 | 2,036–2,341 | 0.8815 | 2,194 | 456,055 | 3,253 | 3,426 | 3,405 | 10 | 55 |
| lm-phase12 | 2,288 | 2,129–2,457 | 0.8795 | 2,169 | 445,087 | 3,644 | 3,010 | 3,296 | 10 | 56 |
| lm-frozen | 3,697 | 3,432–3,971 | 0.7540 | 1,808 | 330,359 | 9,586 | 3,216 | 5,635 | 5 | 54 |

Paired, on 88 chains, no bar:

| lower cost | against | windows cheaper / dearer | mean difference | 95% CI | |
|---|---|---:|---:|---|---|
| trees | lm-signal | 52 / 36 | -90 | -175 to -9 | distinguishable |
| trees | lm-rich | 63 / 25 | -282 | -423 to -151 | distinguishable |
| trees | lm-seed1 | 66 / 22 | -355 | -490 to -218 | distinguishable |
| trees | lm-phase12 | 76 / 12 | -457 | -585 to -334 | distinguishable |
| trees | lm-frozen | 86 / 2 | -1,867 | -2,111 to -1,633 | distinguishable |
| lm-signal | logreg | 65 / 23 | -205 | -299 to -104 | distinguishable |
| lm-rich | logreg | 51 / 37 | -13 | -142 to 125 | not distinguishable |
| logreg | lm-seed1 | 53 / 35 | -61 | -186 to 64 | not distinguishable |
| logreg | lm-phase12 | 51 / 37 | -163 | -292 to -34 | distinguishable |
| logreg | lm-frozen | 84 / 4 | -1,572 | -1,819 to -1,336 | distinguishable |
| **the signal alone** | **lm-signal** | 52 / 36 | -67 | -166 to 33 | not distinguishable |
| lm-seed1 | lm-phase12 | 55 / 33 | -102 | -205 to 6 | not distinguishable |
| lm-rich | lm-phase12 | 59 / 29 | -176 | -281 to -58 | distinguishable |
| lm-signal | lm-rich | 55 / 33 | -192 | -325 to -67 | distinguishable |
| lm-phase12 | lm-frozen | 84 / 4 | -1,409 | -1,641 to -1,188 | distinguishable |

Latency and the batch check:

| setup | p50 ms | p90 ms | p99 ms | under 250 ms | batch vs lone, largest gap | mean gap |
|---|---:|---:|---:|---|---:|---:|
| lm-phase12 | 39 | 56 | 64 | yes | 0.00996 | 0.00115 |
| lm-seed1 | 40 | 55 | 63 | yes | 0.00877 | 0.00102 |
| lm-rich | 53 | 59 | 72 | yes | 0.01494 | 0.00138 |
| lm-signal | 54 | 62 | 78 | yes | 0.00513 | 0.00078 |
| lm-frozen | 38 | 54 | 59 | yes | 0.00879 | 0.00095 |

Batching doesn't hide a difference: on 500 rows each setup's AUC moves by at most 0.002. `lm-phase12` here against its laptop trace on dev0, for the record and not a comparison: largest score gap 0.016, mean 0.001.

Checked apart from the report code: `fraud/dev/verify_tabular.py` replays all 16 setups over 88 chains from the traces. 0 rows decided differently, largest cost gap 0.03 USD per window, 0 customer counts differ. A second bootstrap agrees to within a few dollars. (Observed)

## What it means (Inferred)

- **The prompt's columns matter more than a second seed.** Rich lifts AUC by 0.014 and a seed by 0.002. So what the model reads is a better lever than how many times it's trained.
- **The language model's gap to logreg on dev came largely from what it was shown.** That's the most hopeful finding here for a model that decides on its own, and it's a dev finding. `lm-rich` reads fewer than a tenth of logreg's columns and about ties it. The next test is the final one.
- **`lm-signal` is a tabular score with a language model around it, on this evidence.** It costs about what its score costs, with 62 ms of GPU on each row and a trees pipeline as well. Its gain over `lm-rich` comes through that one line. If the owner wants a decision model whose judgement is its own, `lm-rich` is that setup and `lm-signal` is not. It's still within the owner's terms, which allowed reading the score as a signal.
- **A longer prompt costs little on this GPU.** It adds 5% at p90 and 36% at p50 (59 against 56 ms, 53 against 39), which leaves plenty of room under the bar.
- **The predictions held.** No option beats the trees: held. Seeds about 0.005 of AUC apart: 0.002. Rich gains 0.01-0.02: 0.014. Frozen far behind: held. `lm-signal` cheapest, close to the trees, adding little over its signal, leaning on that line almost entirely: held.

## Negative results

- **No language-model setup beats the trees or its own signal.** (Observed)
- **Frozen is no option:** 9,586 good customers declined, against 3,241–3,644 fine-tuned. (Observed)
- **Two estimates of mine were badly off.** Scoring was 54–85 min, not 20, and rich and signal training were 123–135 min, not 85. The finish moved from about 05:45 to about 10:45. (Observed)

## Limits

- **Dev only, and every read here is flattered by lambda.** Dev's ordering hasn't carried over in Phases 14, 18 and 20. (Observed)
- **One seed per option, except the seed comparison.** A seed's 102 is close to the size of some differences here. Rich's 176 excludes 0 on the windows but rests on one adapter. (Observed; the reading is Inferred)
- **The signal on training rows comes from 4 time-block models, on dev from one fitted on all fit rows.** So the adapter trains on a slightly weaker score than it reads later. (Implementation choice)
- **The prompts were written once and never tuned on scores.** That keeps them clean. It also means nobody knows how good a better prompt could be. (Implementation choice)
- **Latency is this machine's, an RTX 3060 Ti, and isn't comparable with the laptop's.** (Observed)

## Not run

- Two passes: withdrawn by the owner.
- Larger models: none on disk, and no setup here showed the gain that would justify a new reader.
- More prompts, or more seeds of `lm-rich`.
