# Results: d1-Omni-2B fine-tuned for one hour (Phase 9)

Eval `20261102-162345` (`python -m fraud.eval_d1ft --adapter ...`). Pre-registration: the `fraud/eval_d1ft.py` docstring,
committed before training. Numbers: `fraud/results/eval_d1ft/20261102-162345.json|.md`. Model scores are uncalibrated.

## Short answer

- **Gate PASS, bars A, B and C PASS.** (Observed)
- **It is the fastest and cheapest fine-tune so far, but it does not beat the hand rules**: mean cost 810 against
  rules 808, hybrid-ft 830, hybrid-d3ft 835. It ties the rules in 19 of 20 windows and is dearer in the other. (Observed)
- **It learned the rules rather than going past them** (Inferred): on the rules' review queue its AUC is 0.59, close to the others.

## Setup (Implementation choice)

- `LiquidAI/d1-Omni-2B` @ 431f1d667d, BF16 text layers, in our own torch code (`fraud/d1torch.py`). transformers has no LFM3
  class and the repo's remote code is not run. Reproduction check first, on the Phase 8 llama-server traces, eval
  `20261102-152059`: **PASS** (tokens equal on 2,194 of 2,194 rows; AUC fwd 0.9809 vs 0.9830, rev 0.9843 vs 0.9847; median score diff
  0.0140 and 0.0113).
- Phase 5's rows, LoRA r16 on the attention and convolution projections, lone prompts, loss on the temperature-scaled scores.

## Results (Observed)

| | value |
|---|---|
| Training | 1,510 of 1,510 batches, a full epoch, not stopped early; 56.1 min process (cap 60); 2.21 s/batch; peak 3.69 GiB |
| Loss | 0.162 over the first 100 batches → 0.054 over the last 100 |
| Dev AUC, 2,194 rows | both 0.9953, fwd 0.9953, rev 0.9952 → fwd; frozen d1-Omni 0.9847; frozen 1.5B 0.9028; logreg 0.9916 |
| Thresholds | review ≥ 0.05, decline ≥ 0.95 |
| Bar A, cost ≤ logreg | 19 of 20: PASS |
| Bar B, p90 ≤ 250 ms | 92 ms (p50 81 ms), one order: PASS |
| Bar C, cost ≤ hybrid-ft | 20 of 20 (8 cheaper, 12 ties): PASS |
| Reviews over 20 windows | 324 (hybrid-ft 332); no fraud approved, no legit declined |

| setup | mean cost | recall | precision | alert rate |
|---|---:|---:|---:|---:|
| rules | 808 | 1.00 | 0.87 | 0.060 |
| **hybrid-d1ft** | **810** | 1.00 | 0.87 | 0.060 |
| hybrid-ft | 830 | 1.00 | 0.87 | 0.060 |
| hybrid-d3ft (Phase 7) | 835 | 1.00 | 0.87 | 0.060 |
| logreg | 190,910 | 0.89 | 0.85 | 0.056 |
| hybrid-1.5b (frozen) | 5,146,777 | 1.00 | 0.14 | 0.386 |

- Against the frozen 1.5B + filter: dev AUC 0.995 vs 0.903, and mean test cost 810 vs 5.1M. Same prompt and type filter.
- Faster than hybrid-ft (128 ms p90) and hybrid-d3ft (184 ms), reading one order where they read both or a slower layer type.

## Negative and limits

- It does not beat rules-only on cost: 19 ties and 1 dearer window. All of its cost is review fees (324 × 50 ÷ 20 = 810).
- Pre-stated "near half an epoch" was wrong: the steps ran twice as fast as d3-nano's estimate, so a full epoch fit.
- These 20 windows have now compared four models. A fresh comparison would need other hours; Phase 10 used the remaining ones.
- A report bug from this phase's refactor mislabelled the setup as `approve-all` and broke the paired table on the first
  build (a loop variable reused `setup`); fixed and rebuilt from the same traces. Bar values were not affected.
- The console does not list `fraud/results/eval_d1ft/` yet.
