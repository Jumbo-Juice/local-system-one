# Results: the fine-tuned d3-nano (Phase 7)

Eval `20261021-155222` (`python -m fraud.eval_d3ft --adapter fraud/models/20261021-141001_lora-d3`).
Pre-registration: the `fraud/eval_d3ft.py` docstring and the `fraud/finetune_d3.py` docstring,
committed with the PROGRESS pre-statement in `7bd8719` before the reproduction check and before
training. Numbers: `fraud/results/eval_d3ft/20261021-155222.json|.md` and
`fraud/results/eval_d3ft/20261021-130745_repro.json|.md`. The traces (`fraud/runs/20261021-*`), the
checkpoint (`fraud/models/d3-nano-hf/`) and the adapter (sha256 `8d15cf49…`) are git-ignored. Every
number below is **Observed** unless labelled otherwise. Model scores are uncalibrated.

## The question and the answer

Fine-tuned like the Phase 5 Qwen2.5-1.5B (same rows, LoRA, one training run), does d3-nano decide as
cheaply as logistic regression and as the Phase 5 fine-tune (hybrid-ft)? **Yes to both bars, but
not more cheaply than hybrid-ft:** it tied hybrid-ft in 18 of 20 windows and cost 50 more in 2.

| step | rule | value | result |
|---|---|---|---|
| reproduction check | R1 tokens, R2 AUC ± 0.01, R3 median score diff ≤ 0.02 | R3: 0.0207 / 0.0203 | **FAIL** (owner: proceed, see below) |
| cap rule | N = min(1,510, floor(6,600 s / smoke s/batch)) ≥ 755 | 4.87 s → N = 1,356 | trained |
| gate | dev AUC(chosen variant) ≥ dev AUC(logreg) | 0.9953 vs 0.9916 | **PASS** |
| A | cost ≤ logreg in ≥ 10 of 20 windows | **19 of 20** | **PASS** |
| B | p90 latency ≤ 250 ms | **184 ms** | **PASS** |
| C | cost ≤ hybrid-ft in ≥ 10 of 20 windows | **18 of 20** | **PASS** |

## Deviation: the reproduction check failed

Before training, the frozen d3-nano in our own torch code (`fraud/d3torch.py`, BF16) was compared
with the Phase 6 llama-server traces (GGUF Q8_0) on the 2,194 dev model rows:

| | pre-registered | result |
|---|---|---|
| R1 prompt tokens equal on every row | all rows | 2,194 of 2,194: PASS |
| R2 \|AUC − Phase 6 AUC\| per order | ≤ 0.01 | fwd 0.6971 vs 0.6886, rev 0.8816 vs 0.8748: PASS |
| R3 median \|decline score diff\| per order | ≤ 0.02 | fwd 0.0207, rev 0.0203: **FAIL** |

The pre-registration says nothing is trained until the owner has seen why. The why (exploratory, not
pre-registered, `fraud/dev/d3_q8sim.py`):
- The difference is systematic, not noise: correlation 0.997, and our decline score is higher in both
  orders (logit offset +0.14, sd 0.06–0.08), in the option "yes: decline it" (code C in one order, A in
  the other), so it is not a code-position effect.
- With every weight matrix of our model quantized like llama.cpp's Q8_0 (blocks of 32, int8 + an
  fp16 scale), the median difference to the Phase 6 traces drops to **0.0030 / 0.0043** on 150 dev0
  rows (0.0204 / 0.0208 in BF16).
- **Inferred:** our prompt, masks and readout reproduce llama-server; the R3 gap is the reference's
  quantization, which lowers the decline score by about 0.02 against BF16.
- A second reference (vllm-sr's own BF16 runtime, `d3_runtime.py`) was not run: the permission
  check refused to execute the downloaded code.

The owner then chose to proceed exactly as pre-registered, with this failure reported as a deviation.
No tolerance, gate or bar was changed.

## Training (Implementation choice: the pre-registered recipe)

- d3-nano's own wrapper refuses training (Documented, `modeling_d3.py` of `vllm-sr/d3-nano` @
  462efc3662), so `fraud/d3torch.py` loads the plain Qwen3.5 backbone and the FP32 readout and
  re-implements the "d3" prompt and the non-causal softmax layers of `d3_runtime.py`.
- Phase 5's rows (12,084: all 3,021 training fraud + 3× legit, row hash 983ed4fa…) and its seeded
  order coin (6,075 reversed). LoRA r16 on q/k/v/o (6 softmax layers) and on in_proj_qkv/in_proj_z/
  out_proj (18 Gated DeltaNet layers); readout frozen; loss over the three option codes.
- flash-linear-attention 0.5.2 (approved) does not run on this XPU (triton-xpu needs a C compiler;
  the machine has none), so the DeltaNet layers ran transformers' PyTorch reference.
- Smoke run: 4.87 s per batch → N = 1,356 batches (90% of the epoch). The training run did all 1,356
  in **101.8 min** for the whole process, not stopped early (cap 120 min). 4.48 s per batch mean,
  peak 4.31 GiB on the iGPU. Loss: 0.174 over the first 100 batches, 0.057 over the last 100.
- For comparison, Phase 5's 1.5B: all 1,510 batches in 52 min at 2.0 s per batch, peak 3.38 GiB.

## Setups (the 20 Phase 5 test windows, 500 transactions each, ~5% fraud)

Other setups: the committed per-window results of eval `20261005-022333` on the same windows.

| setup | mean cost | median cost | recall | precision | alert rate | p50 / p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| **hybrid-d3ft** (filter + fine-tuned d3-nano, fwd order, dev thresholds) | **835** | 825 | 1.00 | 0.87 | 0.060 | 164 / 184 |
| hybrid-ft (filter + fine-tuned 1.5B) | 830 | 825 | 1.00 | 0.87 | 0.060 | 125 / 128 |
| rules-only | 808 | 800 | 1.00 | 0.87 | 0.060 | – |
| logistic regression | 190,910 | 167,148 | 0.89 | 0.85 | 0.056 | – |
| hybrid-1.5b (frozen) | 5,146,777 | 4,988,782 | 1.00 | 0.14 | 0.386 | 122 / 125 |
| random | 19,605,342 | 14,211,557 | 0.58 | 0.14 | 0.222 | – |
| approve-all | 42,190,884 | 34,751,285 | 0.00 | 0.00 | 0.000 | – |

Paired window counts (two-sided sign test, ties dropped): hybrid-d3ft vs logreg 19 cheaper, 1
dearer (p = 4.0e-5); vs rules 0 cheaper, 9 dearer, 11 ties (p = 0.0039); vs hybrid-ft 0 cheaper,
2 dearer, 18 ties (p = 0.5); vs the frozen 1.5B and random 20 of 20 cheaper (p = 1.9e-6).

## What the fine-tune changed

- **Ranking.** Dev AUC of the decline score from 0.8748 (frozen, best order) to **0.9953**; on the
  pooled test model rows 0.9952 against hybrid-ft's 0.9949 on the same rows (reported, no bar).
- **Option order.** The frozen nano ranked TRANSFERs below chance in the forward order (Phase 6:
  dev AUC 0.689 overall). After the fine-tune both orders give the same ranking (fwd 0.9953, rev
  0.9950, both 0.9952) and the argmax never flips between orders on dev. The debias rule therefore
  kept one order, which halves the forward passes.
- **Scores (uncalibrated).** ECE of the top score 0.014 (hybrid-ft 0.011); top option right for
  97.9% of the test decisions (hybrid-ft 97.8%).
- **Speed.** p90 184 ms with one order through the PyTorch DeltaNet reference (frozen nano, both
  orders, in the reproduction check: 406 ms). It passes Bar B but is slower than the 1.5B (128 ms).

## Insight: the same rule table, plus two zero-amount reviews

- On the 3,862 transactions the model judged, hybrid-d3ft wanted **the same action as hybrid-ft for
  3,860 (99.9%)** and as rules-only for 3,851 (99.7%).
- Its 9 differences from rules-only are hybrid-ft's: fraud TRANSFERs of exactly 10,000,000, which it
  scores 0.888–0.893, under its 0.9 decline threshold, so it reviews instead of declines (50 more each).
- Its 2 differences from hybrid-ft are **fraud CASH_OUTs of amount 0.00** (test212, test222). It sends
  them to review (decline scores 0.72 and 0.78); hybrid-ft and rules approve them. A review costs 50
  and the fraud is worth nothing, so finding it costs 50 more than missing it. That is the whole cost
  gap to hybrid-ft (835 vs 830 mean).
- **Inferred:** two different base models, fine-tuned on the same labels, converge on the same
  decisions; the data (two signals that nearly separate PaySim fraud, `fraud/docs/data.md`) sets the
  result more than the base model. The cost ceiling stays rules-only's, which pays only review fees
  in most windows (`fraud/docs/results.md`).

## Limits

- The reproduction check failed its pre-registered R3 bound; the diagnosis above is exploratory and
  Inferred, and a BF16 reference run of vllm-sr's runtime was not done.
- One training run, one seed, 90% of an epoch (the cap); Phase 5 trained the full epoch.
- Bars A and C compare against costs recorded by another eval (Phase 5) on the same windows; those
  setups were not re-run (Implementation choice, pre-registered). The 20 windows were used once
  before, by Phase 5, and nothing was tuned on them in either phase.
- Latency was measured on the owner's laptop with other apps using most of the RAM (free RAM fell to
  0.4 GB in the smoke run); the reproduction check's dev8 window swapped (1,895 s instead of ~80 s).
- Model scores are softmax values of a model trained at a 1:3 fraud share on enriched windows; they
  are not fraud probabilities.

## Summary

The fine-tuned d3-nano passes the gate and all three bars: cheaper than logistic regression in 19 of
20 windows, never dearer than the Phase 5 fine-tune except for two zero-amount frauds it chose to
review, and 184 ms at p90. It is not better than hybrid-ft: same decisions on 99.9% of rows, a little
slower, and a training run twice as long. The fine-tune fixed the frozen model's option-order bias.
The pre-registered reproduction check failed narrowly before training, which the owner accepted as
a deviation after a diagnosis that points to the Q8_0 reference.
