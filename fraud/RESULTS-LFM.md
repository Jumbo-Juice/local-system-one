# Results: the LFM3 family, frozen (Phase 8)

Eval `20261030-122539` (`python -m fraud.eval_lfm`). Pre-registration: the `fraud/eval_lfm.py` docstring,
committed with the PROGRESS pre-statement (commit 8bb8348) before any LFM run. Numbers:
`fraud/results/eval_lfm/20261030-122539.json|.md`. Model scores are uncalibrated.

## Short answer

- **d1-Omni-2B ranks fraud better than the frozen Qwen2.5-1.5B + type filter on dev**: AUC 0.985 vs
  0.903 (+0.082), with no fine-tuning and the same prompt. (Observed, dev only.)
- **Gate: FAIL.** 0.9847 is below logreg's 0.9916, so the test windows were not run and there is no
  pre-registered test result. (Observed)
- **LFM3-1.7B: no result.** The owner dropped it as too slow before any of its dev traces existed
  (smoke: 454–1,061 ms per transaction vs 172–174 ms for d1-Omni-2B). (Observed)

## Setup (Implementation choice)

- `LiquidAI/d1-Omni-2B-GGUF` Q8_0 (sha256 aedf1a7c…), text only, on llama.cpp `llama-server` b11977
  (winget, Vulkan, Arc 140V; owner-approved deviation from the pre-registered b11960 zip),
  `/v1/systemone`, one slot, one request at a time. Same client as Phase 6 (`fraud/d3.py`).
- Prompt v4 unchanged as one `choice` question; decline score = P("yes: decline it"). Type filter
  first. 2,194 dev model rows (dev0–dev9), 260 fraud. About 338 prompt tokens per order.

## Dev results (Observed)

| | AUC both orders | forward | reversed | chosen | AUC TRANSFER | AUC CASH_OUT | dev cost / window (dev thresholds) |
|---|---|---|---|---|---|---|---|
| frozen 1.5B (Phase 5 dev traces) | 0.9028 | 0.4897 | 0.9022 | both | 0.9354 | 0.8846 | 2,701,634 |
| d1-Omni-2B | 0.9840 | 0.9830 | **0.9847** | reversed | 0.9385 | 1.0000 | 269,831 |
| d3-mini (Phase 6, for reference) | 0.9380 | 0.9378 | 0.9387 | reversed | 0.9356 | 1.0000 | 270K |
| logreg (the gate) | 0.9916 | | | | | | 163K (Phase 6) |

- **Order debias:** not needed. The two orders differ by 0.002 AUC, and averaging gains −0.0007, so
  the rule keeps the reversed order alone. The 1.5B's forward order alone is at chance (0.49).
- **Per type:** CASH_OUT is ranked perfectly (as both D3 models did); TRANSFER 0.939, about the
  same as the 1.5B (0.935) and d3-mini (0.936).
- **Dev cost** (thresholds picked on dev, in-sample for both): d1-Omni-2B 270K per window
  (review ≥ 0.5, decline ≥ 0.7), the 1.5B 2.70M (0.7 / 0.9), 5.35M with argmax.
- **Latency** dev0 p90: 200 ms both orders, **91 ms one order** (d3-mini 327 ms, d3-nano 143 ms one order).

## Interpretation

- **Inferred:** as for d3-mini, the gain over the 1.5B is CASH_OUT, where the decisive cue is
  explicit in the text. On TRANSFER no model gets past what the flags carry (≈ 0.94).
- **Inferred:** pooled AUC (0.985) is much higher than d3-mini's (0.939) with nearly the same per-type
  AUCs, so d1-Omni-2B puts the two types on a more compatible score scale. It is the closest any
  frozen model has come to logreg, and the fastest one-order model tested.
- **Negative:** it still fails the logreg gate, and the 1.5B comparison is on dev only (not a test result).
  LFM3-1.7B was not measured.

## What was not done

- No test windows (gate failed). Nothing tuned after seeing these results; a follow-up (for example
  test windows for d1-Omni-2B against hybrid-1.5b without the logreg gate, or a fine-tune) needs new
  pre-registration under `fraud/CLAUDE.md`.
