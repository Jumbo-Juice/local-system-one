# Results: the Decision 3.0 models (Phase 6)

Eval `20261021-020632` (`python -m fraud.eval_d3`). Pre-registration: the `fraud/eval_d3.py` docstring,
committed with the PROGRESS pre-statement (commit 325ac1d) before any D3 run. Numbers:
`fraud/results/eval_d3/20261021-020632.json|.md`. Model scores are uncalibrated.

## Short answer

- **Gate: FAIL for both models.** Neither reached logreg's dev AUC (0.9916), so the test windows
  were not run and there is no test bar result. (Observed)
- **d3-mini ranks better than the frozen 1.5B on dev**: AUC 0.939 vs 0.903, with no fine-tuning and
  the same prompt. **d3-nano ranks worse**: 0.875 with its best option order. (Observed, dev only.
  This is a dev comparison, not a pre-registered test result.)
- Both are slower than the 250 ms bar of the 1.5B setups when they read both option orders
  (dev0 p90: nano 369 ms, mini 631 ms). With one order, nano is 143 ms and mini 327 ms. (Observed)

## Setup (Implementation choice, owner's choices of 2026-10-21)

- vllm-sr Decision 3.0 d3-nano (2.21B) and d3-mini (4.54B), GGUF Q8_0 (`ggml-org/vllm-d3-*-GGUF`,
  sha256 pinned in `fraud/d3.py`), llama.cpp `llama-server` build 11748 with Vulkan on the Arc 140V,
  `/v1/systemone`, one slot, one request at a time.
- Prompt v4 unchanged as one `choice` question: context + the transaction's flags, the question,
  the three options. The decline score is P("yes: decline it"). It is about 228 prompt tokens per order.
- Type filter first, as in every hybrid setup. 2,194 dev model rows (dev0–dev9), 260 of them fraud.

## Dev results (Observed)

| | AUC both orders | forward | reversed | chosen | AUC TRANSFER | AUC CASH_OUT |
|---|---|---|---|---|---|---|
| frozen 1.5B (Phase 5 dev traces) | 0.9028 | | | both | 0.9354 | 0.8846 |
| d3-nano | 0.8019 | 0.6886 | **0.8748** | reversed | 0.9193 | 1.0000 |
| d3-mini | 0.9380 | 0.9378 | **0.9387** | reversed | 0.9356 | 1.0000 |
| logreg (the gate) | 0.9916 | | | | | |

The type columns are for the chosen variant (both orders for the 1.5B).

- **Debias rule** (gain ≥ 0.005 to keep both orders): neither model keeps both orders. For d3-nano,
  averaging the orders *lowers* AUC by 0.073. Its forward order ranks TRANSFERs worse than chance
  (AUC 0.454), and the reversed order, which lists "yes: decline it" first, ranks them at 0.919.
  d3-mini barely depends on the order (gain −0.0007, argmax flip rate 4.5%).
- **CASH_OUT is ranked perfectly** by both D3 models (AUC 1.000, 1,674 rows, 130 fraud). The 1.5B
  scored 0.885 there. On TRANSFER, d3-mini equals the 1.5B (0.936 vs 0.935).
- **Dev cost** with thresholds picked on dev (in-sample, so optimistic), mean per window:
  d3-mini 270K (review ≥ 0.2, decline ≥ 0.9), d3-nano 578K (0.2 / 0.7). logreg is 163K and the
  frozen 1.5B with argmax 5.35M (Phase 5).
- **Latency** on dev0 (p90): both orders, nano 369 ms and mini 631 ms. One order, nano 143 ms and mini 327 ms.

## Interpretation

- **Inferred:** d3-mini's whole AUC gain over the 1.5B comes from CASH_OUT, where the decisive cue
  ("the SAME amount was already moved … earlier this hour") is explicit in the text. On TRANSFER,
  fraud and legit rows differ less in the shown flags, and neither model goes beyond what the flags
  carry. Logreg also sees four account-count signals that the prompt does not show (the known
  asymmetry from Phase 5).
- **Inferred:** pooled AUC mixes the two types. A perfect CASH_OUT ranking plus a 0.936 TRANSFER
  ranking pools to 0.939 because the D3 scores of the two types sit on different scales. A
  per-type threshold could use that, but it was not pre-registered and is not claimed here.
- **Negative:** reading both option orders, the debias that the 1.5B needed, does not help either
  D3 model. For d3-nano it costs 0.073 AUC and doubles the latency.

## What was not done

- No test windows (the gate failed), so there is no pre-registered test comparison with hybrid-1.5b
  or logreg. The dev numbers above are not test results.
- Nothing was tuned after seeing these results. Under `fraud/CLAUDE.md`, a follow-up (for example
  per-type thresholds, the `noul` question type, or d3-flash) would need new pre-registration and
  untouched windows.
