# PROGRESS: the fraud app (append-only)

## 2026-10-03: Grilling and specs

Decisions from the grilling session with the owner (each one theirs unless marked Implementation
choice):
- Showcase, not production. Its own app in this repo (`fraud/`), clearly separate from the playful
  game demos; it shares only the engine, `.venv` and pytest.
- PaySim (CC BY-SA 4.0) with the four balance columns excluded, as the dataset page says.
- Approve / review / decline per transaction; one tier; order debias.
- Hybrid: the rules only filter fraud-free types; the model judges the rest. Rules-only, numpy
  logistic regression, random, approve-all and a 3B hybrid are reported next to it.
- Prompt: transaction fields plus signals derived from earlier rows.
- Enriched windows (~500 tx, ~5% fraud) from a held-out test period; costs also reweighted to the
  natural rate.
- Simple money cost model with a review budget.
- Bars: hybrid beats rules-only on cost in ≥ 16/20 test windows; p90 ≤ 250 ms.
- Model scores measured and labelled uncalibrated.
- Viewer: replay console + eval page, replay only.
- The owner asked for autonomous work without stops for small decisions. "Done" = the console
  replaying a run at 1× recorded speed.

## 2026-10-03: Phase 0, data look

- PaySim downloaded without a login from Kaggle's public endpoint (186 MB zip, 494 MB CSV) into
  `fraud/data/raw/` (git-ignored). `python -m fraud.data` builds `fraud/data/paysim.npz` in about
  20 s; a test proves no balance column or value reaches the cache.
- Observed facts are in `fraud/DATA.md`. The ones that shape the build:
  CASH_IN/DEBIT/PAYMENT are fraud-free; origin accounts almost never repeat (no per-account tier);
  each fraud TRANSFER is followed by a CASH_OUT of the same amount in the same hour, with unlinked
  ids; legitimate volume collapses after step ~400.
- Implementation choice: train = steps 1–300, test = steps 301–743 (time split; the test period is
  sparser and higher-fraud, reported, not hidden).
- Signals were written test-first (`fraud/signals.py`, no look-ahead test). Same-amount-this-hour
  and new-receiver nearly separate fraud. Expectation stated before any model run (Inferred):
  rules-only will be hard to beat.

## 2026-10-03: Phase 1 done, Phase 2 in progress (paused by the owner)

Built and tested (pytest green): `windows.py`, `costs.py` (DEFAULT: friction 10% min 10, review
50, budget 5%), `rules.py` (type filter + rules-only table from training cells), `baselines.py`
(numpy logistic regression with type × signal interactions, random, approve-all), `capture.py`,
`eval.py` (pre-registration in its docstring, **not yet committed as final and not yet run**),
`brain.py` (first prompt), `runs.py`.

Dev window dev0 (Observed):
- rules-only: cost 1,000 (20 reviews), recall 1.00. Logistic regression: cost 144k, recall 0.92.
  Random: cost 8.3M. Approve-all: cost 46.3M.
- hybrid 1.5B (prompt v0): **declines every TRANSFER and CASH_OUT**, cost 4.38M. Latency p50
  142 ms, p90 153 ms (under the 250 ms bar).

Memory (Observed): the iGPU reported 5.42 GiB total and 3.66 GiB free, not the 8,097 MB in
docs/machine.md. The 1.5B with a float32 LM head ran out of memory, so capture/eval use
`head_dtype = "model"` (bfloat16) for now (`HEAD_DTYPE` in capture.py). The 3B (6.2 GB) cannot load
at 5.42 GiB. **The owner is reconfiguring the shared GPU memory override and restarting.** After
that, check `torch.xpu.get_device_properties(0).total_memory`. If it is back to ~8 GB, consider
returning `HEAD_DTYPE` to "float32" (the repo default) before the eval is pre-registered.

Prompt development (dev1+dev2, 389 model decisions, 52 fraud; `fraud/dev/promptdev.py`):
| variant | change | AUC of review+decline score | top action |
|---|---|---:|---|
| v0 | the current brain.py prompt (raw counts) | 0.58 | decline 389/389 |
| v1 | signals as flags ("NEW account…", "SAME amount already moved…") | 0.85 | decline 389/389 |
| v2 | v1 + context naming the two fraud patterns | 0.86 | decline 389/389 |
| v3 | v2 + options reworded (normal / suspicious / fraud) | 0.65 | decline 389/389 |
v4/v6/v7 (a "does it match the fraud pattern?" question; a base-rate sentence) were started and
stopped before finishing. Inferred: the 1.5B reads the flags (the ranking improves) but its top
option stays "decline". Next: finish v4–v7. If argmax never moves, bring the owner a design
question (keep argmax and report the negative result, or a decision rule on the scores tuned
on dev windows). Don't decide that alone: argmax was agreed in the grilling.

Next after that: freeze brain.py, commit the eval pre-registration, run `python -m fraud.eval`
(Phase 3), then the viewer (Phase 4).

## 2026-10-03: XPU memory back; prompt variants v0–v7 finished

- XPU memory (Observed): `total_memory` 7.18 GiB ("Arc 140V GPU (8GB)"), 5.36 GiB free at idle. The
  1.5B with the **float32 LM head** (the repo default) now loads and runs. The 3B (6.2 GB) still
  cannot fit in 5.36 GiB free (Inferred, not tried this session).
- `fraud/dev/promptdev.py`: `HEAD=float32|model` env switch, a v5 (the v4 question with the original
  options, to separate the question from the options), `SAVE=<dir>` to keep the scores. Rerun of all
  variants with the float32 head on dev1+dev2 (389 model decisions, 52 fraud), Observed:

| variant | change | AUC review+decline | AUC decline | top action | p50 / p90 ms | tokens |
|---|---|---:|---:|---|---:|---:|
| v0 | brain.py prompt (raw counts) | 0.579 | 0.634 | decline 389/389 | 146 / 161 | 209 |
| v1 | signals as flags | 0.852 | 0.850 | decline 389/389 | 140 / 144 | 195 |
| v2 | v1 + both fraud patterns in context | 0.863 | 0.871 | decline 389/389 | 138 / 139 | 215 |
| v3 | v2 + options normal/suspicious/fraud | 0.637 | 0.833 | decline 389/389 | 137 / 139 | 216 |
| v4 | v2 + "Does this match the fraud pattern?" no/unsure/yes | **0.927** | 0.926 | decline 389/389 | 137 / 139 | 214 |
| v5 | v2 + the v4 question, original options | 0.885 | 0.904 | decline 389/389 | 138 / 140 | 216 |
| v6 | v2 + base rate ("about 1 in 10 … is fraud") | 0.869 | 0.896 | decline 389/389 | 162 / 168 | 235 |
| v7 | v4 + base rate | 0.927 | 0.928 | decline 389/389 | 162 / 164 | 234 |

  v0–v3 match the bfloat16-head numbers of the previous entry within 0.02 (the head dtype does not
  explain the argmax). **Negative result:** no prompt variant moves the 1.5B's top option off the
  third option; its ranking is good (v4: AUC 0.93) but its argmax is constant. All variants stay well
  under the 250 ms p90 bar.
- Rules-only and logistic regression on all ten dev windows (Observed): rules-only has recall 1.00
  and no wrong decline in **every** dev window; its cost is review fees only (750–1,000). Logistic
  regression (thresholds 0.01 / 0.2) costs 7k–393k.
- Inside the cell rules-only reviews (TRANSFER to a new receiver, not round; training steps, 40,092
  rows, 1,649 fraud), no simple subset is pure enough to decline: hours 0–6 are 42% fraud, amounts
  ≥ 5M 56%, the rest 2–14%. **Inferred:** with these signals and costs (a wrong decline costs 10% of
  ~0.5M, a missed fraud its full amount, a review 50), rules-only is close to the cheapest possible
  policy. A decider beats it in a window only if it gets every one of that window's ~15–20 reviewed
  transfers right without a review. Bar 1 ("cheaper in ≥ 16 of 20", ties not counted) looks
  structurally out of reach for any decider on these signals, the model included.
- A score-threshold rule instead of argmax (Observed, offline on saved v4 scores, dev windows only):
  v4 on dev0–dev9 (2,194 model decisions, 260 fraud): AUC 0.90, argmax decline 2,194/2,194, p90
  135 ms. Its decline score spans only 0.63–0.86. Thresholds on the decline score from the logreg-style
  grid, tuned on dev0–4 (review ≥ 0.30, decline ≥ 0.70), checked on dev5–9: cost **0.70M–1.71M** per
  check window vs rules-only 750–1,000; cheaper than rules in **0 of 10** dev windows. It reviews 25
  (the budget) and then declines legit transfers by fallback. Even with a dev-tuned rule, the 1.5B's
  scores do not separate well enough to compete with rules-only or logistic regression.
- **Open for the owner** (not decided alone; argmax was agreed in the grilling): (1) keep argmax
  and report "declines everything" as the negative result, or switch to a dev-tuned score rule
  (also a clear loss); (2) Bar 1 as pre-registered looks structurally unreachable (see above):
  keep it and expect FAIL, or restate it before any test window runs.

## 2026-10-03: owner decisions; Phase 2 done

Owner decisions (answers to the open questions above):
- **Argmax stays**, prompt **v4** frozen in `brain.py` (question "Does this transaction match the
  fraud pattern?", options no: approve / unsure: analyst / yes: decline, the v2 context, signals as
  flags). The dev score-rule result above stays a note; it is not an eval setup.
- **Bar 1 is kept as pre-registered**, with the expected FAIL stated in `fraud/eval.py` before the run.
- **hybrid-3b is dropped** from the eval: the 3B (6.2 GB) does not fit the 5.36 GiB free (Observed
  limit of this machine, not a result about the 3B).

Implementation choice: `HEAD_DTYPE` back to "float32" (the repo default) now that it fits.
`promptdev.py` keeps its own copy of the v0 prompt so the table above stays reproducible.

Phase 2 acceptance (Observed, `python -m fraud.capture --setup hybrid --window dev0`):
- mock (`--config config/mock.toml`): finished trace, cost 24.6M, recall 0.62 (random choices).
- Qwen2.5-1.5B, prompt v4, float32 head: finished trace; cost 4.38M, recall 1.00, alert rate 0.51,
  0 reviews (every model row declined); latency **p50 125 ms, p90 126 ms**, max 188 ms over 257
  model decisions (bar: p90 ≤ 250 ms).
- New tests `fraud/tests/test_brain_capture.py`: the prompt shows flags and no account ids, the
  option→action order, a mock model decision, a mock capture writes a finished trace.

## 2026-10-03: Phase 3, the eval (20261003-052158)

The pre-registration was committed in `47c32b1`, then `python -m fraud.eval` ran all 5 setups on
test100–test119 once (about 9 min). Full write-up: `fraud/RESULTS.md`. Observed:
- **Bar 1 FAIL** (as stated before the run): hybrid-1.5b cheaper than rules-only in 0 of 20.
  **Bar 2 PASS**: p90 127.2 ms (p50 121, max 203; 3,800 model decisions).
- The hybrid declined 3,800 of 3,800 model rows. Read in the listed order the model always picked
  the last option ("decline"); read reversed it picked the last option ("approve") 71% of the time.
  The averaged argmax stayed "decline" because the listed-order readings were more confident.
- Mean / median cost: rules 400,845 / 825; logreg 549,247 / 134,044; hybrid 6.37M / 5.40M; random
  17.9M / 14.4M; approve-all 31.7M / 27.7M. Rules cheaper than logreg in 18 of 20.
- Rules-only failed once: test109 (steps 327–328) holds a burst of legit 10,000,000 TRANSFERs (the
  PaySim cap); the "round transfer" rule declined 8 of them (8.0M, 99.8% of rules-only's total).
  Not fixed: a test result doesn't change the rules (`fraud/CLAUDE.md`).
- A bug found after the run: printing the report crashed on the Windows console (cp1252 can't
  encode "→"), after the results files were written. `main()` now reconfigures stdout to UTF-8,
  and `--report 20261003-052158` rebuilt the files from the traces. No result changed.

Next: Phase 4, the analyst console.

## 2026-10-03: Phase 4, the analyst console

`python -m fraud.viewer` (stdlib server on 127.0.0.1:8766; `fraud/viewer/server.py`,
`index.html`, `console.js`, `console.css`; no code shared with the game viewer). Pages: start (runs
table with setup filter and an "include eval runs" switch, latest eval with its bars, auto-demo),
replay, eval, auto-demo (model runs at 1×, pinned first, 3 s between runs).
Replay: a play/pause/step/restart/speed control with a scrubber; the transaction feed (newest
first, click to inspect); a decision card (decider, rule or "model scores (uncalibrated)" with
both option orders, outside mass, tokens, latency, truth, cost, signals); running totals (cost
split into missed fraud / wrong declines / reviews, confusion matrix, reviews left); a latency
chart with the 250 ms bar; "About this run" (banner, fact cards with hints, notes, the prompt).

Checked (Observed):
- 1× replay of the pinned dev0 run, driven by hand with `frame(ts)` (the hidden pane does not
  animate): 17 transactions shown at 1.0 s, 106 at 6.0 s, 265 at 16.0 s, all 500 at 32.2 s, the
  sum of the recorded decision times. Each transaction appears when its recorded decision finished.
- Start page, full replay, auto-demo (first run = the pinned one) and eval page render
  (headless Edge, 1440 px). No console errors. At 375 px no page scrolls sideways (scrollWidth =
  375 on the start page, eval page, a model run and a rules run).
- Fixed on the way: at the end of a run the hero cost now shows the trace's own total (the
  per-row costs are rounded, and the running sum was 1 unit off); latencies below 0.001 ms read
  "<0.001 ms"; the rule-table note no longer nests parentheses.
- Pinned: `fraud/runs/20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl`, listed in
  `fraud/runs/README.md` with the PaySim attribution (CC BY-SA 4.0).
- Tests: `fraud/tests/test_viewer.py` (served paths only, listing, own files only).

## 2026-10-04: promptdev.py v0 fixed and rerun

- `fraud/dev/promptdev.py` did not parse from 47c32b1 (Phase 2) until c19556c (Observed): the v0
  copy had a raw line break inside its `"\n".join` string. So the claim in the Phase 2 entry, that
  the copy keeps the v0 row reproducible, was not true in that span. The v0 rows above came from
  `brain.py`'s own prompt before it was frozen on v4, not from this copy (Inferred from the history).
- Fixed: the copy now matches the pre-freeze `brain.py` `describe()` line for line (Observed, compared
  against 98bc801).
- Rerun, `HEAD=float32 python -m fraud.dev.promptdev 1,2 v0` (Observed): 389 decisions, 52 fraud;
  AUC review+decline 0.579, AUC decline 0.634, decline 389/389 (52 fraud, 337 legit), p50 147 /
  p90 155 ms, 209 tokens. AUCs, actions and tokens match the 2026-10-03 table exactly; latency
  differs by ≤ 6 ms (146 / 161 there).

## 2026-10-05: Phase 5 opened, the fine-tuned brain (pre-registration)

Owner's goal: a LoRA fine-tune of the 1.5B, measured against logistic regression (Bar A: cost ≤
logreg in ≥ 10 of 20 new windows; Bar B: p90 ≤ 250 ms). Action from dev thresholds, not argmax.
New windows share no hour with test100–test119. Training set: all fraud + 3× legit, 1 epoch.

- Spike (Observed, throwaway script outside the repo): Unsloth did not install on this machine
  (its install replaced the XPU torch with a CPU build; `triton-windows` and `triton-xpu` clash).
  `peft` 0.21.2 + `accelerate` 1.15.0 in the repo `.venv` left torch 2.14.0+xpu intact. A toy
  4-option task in the engine's prompt format: 100 LoRA steps (r16, batch 8, ~113 tokens) took
  1.33 s/step at 3.16 GiB peak and moved held-out accuracy 53.5% → 88.0%. Before training the
  model chose the last option in 82 of 200 items; after, picks were spread evenly. A reloaded
  adapter gave the same argmax and accuracy; label logits differed by up to 0.375 (bf16 on the
  XPU differs by up to 0.25 between two identical passes, so the pre-set 1e-2 check was unreachable).
- Packages: `peft` and `accelerate` approved by the owner for this phase (vetted: upstream
  huggingface/peft and huggingface/accelerate document `pip install peft` / `accelerate`).
- Training set (Observed, `python -m fraud.finetune --dry-run`): 12,084 rows (3,021 fraud, 9,063
  legit; 3,211 TRANSFER, 8,873 CASH_OUT), row sha256 983ed4fa…; prompts 214 tokens mean, 223 max;
  6,075 of 12,084 shown with reversed options.
- New test windows (Observed, `python -m fraud.eval_ft --windows`): test200, 201, 202, 204, 208,
  210–217, 219, 221, 222, 224, 228, 232, 233; seeds 203, 205–207, 209, 218, 220, 223, 225–227,
  229–231 skipped for shared hours.
- Smoke run (10 steps; adapter deleted): 2.13 s/step, 3.37 GiB peak, so the full 1,510 steps take
  ~54 min. The merged smoke adapter ran the dev path at 128 ms p50 (same as the base model).
- The pre-registration is the `fraud/eval_ft.py` docstring, committed with this entry before the
  adapter is trained.

## 2026-10-05: Phase 5, the adapter and the dev gate

- Adapter `fraud/models/20261005-013044_lora` (git-ignored), trained once with the committed
  configuration (Observed, its manifest): 1,510 steps in 52.4 min (1.99 s/step p50), peak 3.38 GiB,
  loss mean 0.416 over the first 100 steps → 0.054 over the last 100; adapter sha256 02a0c5d9…;
  torch 2.14.0+xpu, peft 0.21.2.
- Dev gate (Observed, `python -m fraud.eval_ft`, dev0–dev9, 2,194 model rows, 260 fraud): AUC of
  the order-averaged decline score 0.9028 (frozen hybrid-1.5b) → **0.9950** (hybrid-ft). Gate
  PASS. For reference (computed separately, same rows): logistic regression 0.9916.
- Thresholds picked on dev (Observed): review ≥ 0.05, decline ≥ 0.95.
- With argmax (the dev traces), mean dev cost per window fell from 5.35M (hybrid-1.5b) to 0.31M
  (hybrid-ft); recall 1.00 for both. Latency p90 128 ms (hybrid-ft) vs 139 ms (hybrid-1.5b) on dev.

## 2026-10-05: Phase 5 eval and results

Eval `20261005-022333` (`python -m fraud.eval_ft --adapter fraud/models/20261005-013044_lora`,
~27 min). Write-up: `fraud/RESULTS-FT.md`. Observed:
- Bar A PASS: hybrid-ft cost ≤ logreg in 19 of 20 new windows (sign test p = 4.0e-5).
  Bar B PASS: p90 127.9 ms. Old bar 1 (vs rules) FAIL as stated: 0 cheaper, 9 dearer, 11 ties.
- Mean cost: hybrid-ft 830, rules 808, logreg 190,910, hybrid-1.5b 5.15M. Recall hybrid-ft 1.00.
- hybrid-ft wanted the same action as rules-only on 3,853 of 3,862 model rows. The 9 differences
  are fraud 10M TRANSFERs to new accounts (score 0.92–0.93 < 0.95 → review, +50 each).
- 57 of logreg's 59 approved frauds were TRANSFERs to a new receiver scored ~0.009, under its
  dev-picked review threshold 0.01 (the grid floor).
- Console: `hybrid-ft` label, a fourth eval-chart series, a dev-gate panel, generic bar titles;
  the old and new eval pages render, no console errors, 375 px without sideways scroll.
Phase 5 done.

## 2026-10-05: exploratory check on test109 (owner's request; written before running)

Question: in test109 (steps 327–328) rules-only declined 8 legit TRANSFERs of exactly 10,000,000
(cost 8.0M, Phase 3). What does hybrid-ft do with them? Not pre-registered and not a bar: one run of
hybrid-ft on test109 with the frozen adapter (sha256 02a0c5d9…) and the frozen dev thresholds
(review ≥ 0.05, decline ≥ 0.95). Nothing (adapter, thresholds, rules, prompt) changes on its
result. Reported: total cost vs rules-only / logreg / hybrid-1.5b from eval 20261003-052158, the
action and decline score of each 10M row, and whether the review budget (5% = 25) runs out.
Result (Observed; trace `fraud/runs/20261005-030728_explore_hybrid-ft_test109.jsonl`, git-ignored):
- hybrid-ft cost **8,000,800**, identical to rules-only: its final action equals rules-only's on all
  500 rows. It declined all 8 legit 10M TRANSFERs, decline scores 0.989–0.994 (rules and logreg
  declined them too). Review budget not exhausted (16 wanted, 9 left). p90 173 ms.
- Negative for the guess in results-ft.md: the model is *more* sure on these legit 10M rows than
  on the fraud 10M rows of the new windows (0.92–0.93). **Inferred:** the legit burst carries "the
  SAME amount was already moved 41–322 times earlier this hour" and a round amount, the strongest
  fraud cues it learned; its hesitation is on new receivers without a repeated amount.

## 2026-10-05: data source reference changed (owner)

The owner checked that Kaggle `sriharshaeedala/financial-fraud-detection-dataset`
(`Synthetic_Financial_datasets_log.csv`) is the same file as the PaySim CSV used so far, and chose it
as the cited source (its Kaggle usability rating is 10.0). Every source reference now points to it
(`fraud/data.py`, `DATA.md`, PRD, `runs/README.md`, the console footer, RUN-GUIDE); the PaySim
authors stay credited (CC BY-SA). The local raw file was renamed to the new name; `find_csv` still
accepts the old `PS_*.csv` name. No result changes: same bytes, same cache.

## 2026-10-21: Phase 6 pre-statement, the Decision 3.0 models (owner; written before any D3 run)

Question: does an off-the-shelf decision model, vLLM Semantic Router's Decision 3.0 (d3-nano 2.21B,
d3-mini 4.54B, Apache-2.0), rank fraud better than the frozen 1.5B (dev AUC 0.9028), with no
fine-tuning? Owner's choices from a grilling session on 2026-10-21. The full pre-registration is
the `fraud/eval_d3.py` docstring, committed with this entry before the first D3 run.
- Runtime (Implementation choice, owner-approved): llama.cpp `llama-server` build 11748 (winget,
  Vulkan on the Arc 140V), native `/v1/systemone` endpoint, GGUF Q8_0 from
  `ggml-org/vllm-d3-{nano,mini}-GGUF` (sha256 4867d306… / f9f99be4…, checked on download) in
  `fraud/models/d3/` (git-ignored). One slot (`-np 1`), one request at a time: llama.cpp issue
  #30402 (open) reports that a D3 answer changes when another request shares its batch.
  vLLM itself does not run natively on Windows (Documented, docs.vllm.ai).
- Prompt v4 unchanged: state = CONTEXT + describe(), instructions = QUESTION, criteria = OPTIONS
  (descriptions null) as one `choice` question. Fraud score = P("yes: decline it").
- Order debias: each transaction is asked in both option orders (two questions, one request). The
  server reads a vllm-d3 choice in one order only, the order given (Documented, server-decision.cpp).
  Rule: keep debias for a model only if its order-averaged dev AUC beats the better single order
  by >= 0.005; otherwise use that single order. Flip rate between orders reported, not used.
- Dev gate (exploratory part): dev0–dev9 model rows. A model goes on only if its frozen variant's
  dev AUC >= logreg's dev AUC on the same rows (0.9916 in Phase 5). Each model is gated alone.
- Thresholds on dev as in eval_ft (GRID, min dev cost), frozen before the test run.
- Test (only for models that pass): the 20 original windows test100–test119, once. Bar: pooled AUC
  over the model rows > hybrid-1.5b's pooled AUC on the same rows (recomputed from eval
  20261003-052158 traces). Reported, no bar: logreg's pooled AUC, cost per window vs rules, logreg
  and hybrid-1.5b, recall, latency p50/p90.
- Stated before running (Inferred): the gate is at logreg level, so both models may stop at dev.
  That result gets written up as a negative.

## 2026-10-21: Phase 6 results, gate FAIL for both D3 models

Eval `20261021-020632` (`python -m fraud.eval_d3`, ~11 min nano + ~23 min mini). Write-up:
`fraud/RESULTS-D3.md`. Observed:
- Setup notes: llama-server answered 401 at first because the owner's user environment sets
  `LLAMA_API_KEY` (llama-server then requires it); `fraud/d3.py` now sends it as a bearer token.
  The smoke run showed that one order asked alone gives the same probabilities as in the pair.
- Dev AUC (2,194 model rows, 260 fraud): d3-nano both 0.8019 / fwd 0.6886 / rev 0.8748 → rev;
  d3-mini 0.9380 / 0.9378 / 0.9387 → rev. Debias rule: neither keeps both orders.
- Gate (>= logreg 0.9916): **FAIL for both**. The test windows were not run.
- d3-mini ranks better than the frozen 1.5B on dev (0.939 vs 0.903): CASH_OUT AUC 1.000 vs 0.885,
  TRANSFER 0.936 vs 0.935. d3-nano's forward order ranks TRANSFER below chance (0.454).
- Latency dev0 p90: both orders nano 369 / mini 631 ms; one order 143 / 327 ms.
- Nothing changes on these results. A follow-up needs a new pre-registration and untouched windows.
Phase 6 done.

## 2026-10-21: Phase 7 pre-statement, the fine-tuned d3-nano (owner; written before training)

Question: fine-tuned like the Phase 5 Qwen2.5-1.5B, does d3-nano (frozen dev AUC 0.8748, rev order)
decide as cheaply as logistic regression and as hybrid-ft? Owner's choices from a grilling session
on 2026-10-21. The full pre-registration is the `fraud/eval_d3ft.py` docstring, committed with this
entry before the training run.
- Runtime: d3-nano's own wrapper refuses training (Documented, its `modeling_d3.py`), so
  `fraud/d3torch.py` loads the plain Qwen3.5 backbone + FP32 readout of vllm-sr/d3-nano @ 462efc3662
  (sha256 from the hub metadata, checked on download and on load) and re-implements the "d3"
  prompt and the non-causal softmax layers of `d3_runtime.py` (read, not run). Torch on the XPU,
  adapter merged at inference; no GGUF conversion (no new `gguf` package).
- flash-linear-attention 0.5.2 (approved): vetted (upstream fla-org README/INSTALL.md
  `pip install flash-linear-attention[xpu]`, PyPI 0.5.2 = latest, no attestations), installed
  without touching torch 2.14.0+xpu, then **removed**. Observed: triton-xpu 3.8.0 needs a C
  compiler to build its driver and the machine has none (no Visual Studio, clang, gcc or oneAPI),
  so fla falls back to CPU device detection and its kernel fails on XPU tensors ("PyTorch was
  compiled without CUDA support"). While installed, transformers binds it in place of its PyTorch
  reference, so the model would crash. The DeltaNet layers run transformers' PyTorch reference.
- Observed before writing the pre-registration: our "d3" prompt gives exactly llama-server's token
  counts on all 257 model rows of dev0 (227 or 230 per order).
- Data: Phase 5's exact rows and order coin (row hash 983ed4fa…, `--dry-run` reproduces it).
- LoRA r16/alpha 32/dropout 0.05 on q/k/v/o_proj (6 softmax layers) and in_proj_qkv/in_proj_z/
  out_proj (18 Gated DeltaNet layers), readout frozen; Phase 5's optimiser and schedule.
- The 2-hour cap (training run only): a 20-batch smoke run measures s/step; N = min(1,510,
  floor(6,600 s / s per step)), the LR schedule sized to N; below 755 batches nothing is trained and
  the owner decides; the process stops stepping at 2 h minus 2 min and saves.
- Reproduction check first (R1 tokens, R2 AUC ± 0.01, R3 median score diff <= 0.02 per order).
- Debias rule of Phase 6 on dev; gate: dev AUC >= logreg (0.9916); then the 20 Phase 5 windows.
  Bars: A cost <= logreg in >= 10/20, B p90 <= 250 ms, C cost <= hybrid-ft in >= 10/20.

## 2026-10-21: Phase 7 reproduction check FAIL (R3), nothing trained

`python -m fraud.eval_d3ft --repro`, check `20261021-130745`
(`fraud/results/eval_d3ft/20261021-130745_repro.md`): the frozen d3-nano through `fraud/d3torch.py`
(BF16, XPU) against the Phase 6 llama-server Q8_0 traces, 2,194 dev model rows. Observed:
- R1 prompt tokens equal on 2,194 of 2,194 rows: PASS.
- R2 AUC fwd 0.6971 vs 0.6886, rev 0.8816 vs 0.8748 (|diff| <= 0.01): PASS.
- R3 median |decline score diff| fwd 0.0207, rev 0.0203 (<= 0.02): **FAIL**. So the check fails and,
  as pre-registered, nothing is trained until the owner has seen why.
- Latency, both orders in one pass: p50 335 ms, p90 406 ms. The machine ran out of RAM from about
  13:21 (0.1 of 15.6 GB free; this process 8.1 GB, a WSL/Docker `vmmem` 4.0 GB): dev8 took 1,895 s
  instead of ~80 s, so latency is inflated there.
Why (exploratory, not pre-registered, after the failure):
- The differences are systematic, not noise: correlation 0.997, and our decline score is higher in
  both orders (logit offset +0.137 fwd / +0.144 rev, sd 0.06 / 0.08). Decline is code C in one order
  and code A in the other, so the shift follows the option, not the code position.
- `fraud/dev/d3_q8sim.py`: our forward with every weight matrix quantized as llama.cpp Q8_0 (blocks
  of 32, int8 + fp16 scale, back to bf16), first 150 dev0 model rows. Median |decline diff| to the
  Phase 6 traces: **0.0030 fwd / 0.0043 rev** (BF16: 0.0204 / 0.0208). **Inferred:** our prompt,
  masks and readout reproduce llama-server; the R3 gap is the reference's own Q8_0 quantization,
  which lowers the decline score by about 0.02 against BF16.
- Running vllm-sr's own BF16 runtime (`d3_runtime.D3`) as a second reference was refused by the
  permission check (it executes downloaded code); not done.
- Owner's decision (2026-10-21, after seeing the diagnosis): proceed to the smoke run, training and
  eval exactly as pre-registered, with the R3 failure and its Q8_0 diagnosis reported as a stated
  deviation. No tolerance, gate or bar changes.

## 2026-10-21: Phase 7 smoke run and the cap rule

`python -m fraud.finetune_d3 --steps 20` → `fraud/models/20261021-140723_lora-d3-smoke` (git-ignored).
Observed: rows 12,084 (hash 983ed4fa…, as Phase 5), 6,075 reversed, prompts 228 tokens mean / 237
max; batches 4–20 took 4.87 s mean (median 4.83; batch 1: 10.5 s); peak XPU memory 4.31 GiB;
loss 1.10 at batch 1, 0.03–0.36 over batches 11–20; process 2.4 min. Free RAM fell to 0.4 GB
during the run (the owner's other apps: a WSL/Docker VM 4.0 GB, Steam, Discord), but the step times
stayed steady. Cap rule: N = min(1,510, floor(6,600 / 4.87)) = **1,356 batches** (90% of the
epoch, >= 755), about 110 min of steps. The training run started at 14:11 with `--plan`.

## 2026-10-21: Phase 7 training run (within the cap)

`python -m fraud.finetune_d3 --plan fraud/models/20261021-140723_lora-d3-smoke` →
`fraud/models/20261021-141001_lora-d3` (git-ignored), adapter sha256 `8d15cf49…`. Observed (manifest):
1,356 of 1,356 planned batches, **not stopped early**; process 101.8 min (training steps 101.3 min)
against the 120 min cap; 4.48 s/batch mean (median 4.19); peak XPU memory 4.31 GiB; loss 0.174 over
the first 100 batches → 0.057 over the last 100 (means per 200 batches: 0.138, 0.075, 0.065, 0.085,
0.056, 0.080, 0.065). The eval (`python -m fraud.eval_d3ft --adapter …`) started at 15:52.

## 2026-10-21: Phase 7 results, gate PASS and bars A/B/C PASS

Eval `20261021-155222` (started 15:52, all traces done before 16:16). Write-up: `fraud/RESULTS-D3FT.md`.
Observed:
- Dev AUC (2,194 model rows): fwd 0.9953, rev 0.9950, both 0.9952 → debias gain −0.0001, variant
  fwd; argmax flip rate 0.0. Gate vs logreg 0.9916: **PASS**. Thresholds review ≥ 0.05, decline ≥ 0.9.
- Bar A cost ≤ logreg in **19/20**: PASS. Bar B p90 **184 ms**: PASS. Bar C cost ≤ hybrid-ft in
  **18/20**: PASS (18 ties, 2 dearer by 50, none cheaper). Mean cost 835 (hybrid-ft 830, rules 808).
- Same wanted action as hybrid-ft on 3,860 of 3,862 model rows. The 2 differences: fraud CASH_OUTs of
  amount 0.00 that nano-ft reviews (50 each) and hybrid-ft approves. Its 9 differences from rules are
  hybrid-ft's: 10M fraud TRANSFERs scored 0.888–0.893, reviewed instead of declined.
- The report step first crashed after all traces had run (`model_calibration` read `outside_mass`,
  which D3 decisions do not record); fixed with a test, results rebuilt with `--resume`.
- Negative and limits: the reproduction check failed R3 (owner-approved deviation); 90% of an epoch;
  no gain over hybrid-ft; slower (184 vs 128 ms) and twice the training time. Phase 7 done.

## 2026-10-21: console shows the fine-tuned d3-nano (owner)

- `fraud/viewer/server.py` serves and lists evals from `fraud/results/eval/` and
  `fraud/results/eval_d3ft/` (`EVAL_DIRS`); files without `totals` (the repro check) are not evals.
- `console.js`: `hybrid-d3ft` label and a fifth chart series; the d3ft eval page merges the Phase 5
  setups of eval 20261005-022333 (same windows) and links each cost to the eval that recorded it;
  a training card and a dev-gate panel by option order; bar C. D3 traces (Phase 6 and 7) now replay:
  their per-order scores are objects and may hold one order, which broke the replay before.
- Checked in the browser pane (Observed): start page, the d3ft eval page (140 of 140 per-window
  links resolve to traces on disk), the Phase 3 and Phase 5 eval pages, a d3ft and a Phase 6 d3-nano
  replay played to the end with the decision panel, auto-demo starts, no console errors, no sideways
  scroll at 375 px.

## 2026-10-24: write-ups moved to the fraud/ root (owner)

- The fraud docs were in two places: the planning docs at `fraud/` and the write-ups one level down
  in `fraud/docs/`. Now every fraud doc sits at the `fraud/` root, in uppercase (an uppercase `.md`
  is a doc, lowercase is code; `DATA.md` stays apart from `data.py` and `data/`):
  `docs/data.md` → `DATA.md`, `docs/results.md` → `RESULTS.md`, `docs/results-ft.md` →
  `RESULTS-FT.md`, `docs/results-d3.md` → `RESULTS-D3.md`, `docs/results-d3ft.md` →
  `RESULTS-D3FT.md`. `fraud/docs/` is gone.
- Only path text changed, including the older entries above and one path in the pre-registration
  docstring of `fraud/eval_ft.py` (owner-approved); no wording, seed, window or rule changed. The
  console's write-up line names the new paths. New fraud write-ups go at the `fraud/` root as
  `RESULTS-<setup>.md`; raw eval output still goes to `fraud/results/<script>/`.

## 2026-10-30: Phase 8 pre-statement, the LFM3 family frozen (owner; written before any LFM run)

Owner's request: try LiquidAI's LFM3 family, LFM3-1.7B and d1-Omni-2B, run the same test as
Phase 6, no fine-tuning, and say whether either beats the base Qwen 1.5B + type filter. The full
pre-registration is the `fraud/eval_lfm.py` docstring, committed with this entry.
- Models (Documented, Hugging Face model cards): `LiquidAI/d1-Omni-2B` is a decision model built on
  LFM3-Omni-2B that answers typed questions on llama.cpp's `/v1/systemone` (as the D3 models);
  `LiquidAI/LFM3-1.7B` is a chat model. Both need llama.cpp b11960 or newer. Q8_0 GGUFs from
  `LiquidAI/*-GGUF` (sha256 aedf1a7c… / 8f6b2418…) in `fraud/models/lfm/` (git-ignored).
- Runtime (Implementation choice): llama.cpp b11960 Vulkan x64 release zip (sha256 bac6914b…),
  unzipped into `fraud/models/llama-b11960/`, so the winget b11748 of Phase 6 stays as it was.
  transformers 5.17 has no `lfm3` model class and the repo's remote code is not run, so LFM3-1.7B
  goes through the engine's llama.cpp backend (HF tokenizer, next-token label scores) instead of
  the HF backend the 1.5B used. Q8_0 vs the 1.5B's bf16 weights is a difference in the comparison.
- Protocol: Phase 6 unchanged (dev0–dev9, both orders, debias rule ≥ 0.005, gate ≥ logreg dev AUC,
  thresholds on dev, test100–test119 for passers, bar pooled AUC > hybrid-1.5b). Added: the
  headline comparison against the frozen 1.5B's dev AUC on the same rows (0.9028, recomputed from
  eval 20261005-022333 before any LFM run: 0.9028, TRANSFER 0.9354, CASH_OUT 0.8846), per-type AUCs,
  and dev cost with dev-picked thresholds for both (in-sample).
- Stated before running (Inferred): both models may stop at the logreg gate, as both D3 models did.

## 2026-10-30: Phase 8 runtime deviation (owner; before any LFM run)

Running the downloaded b11960 release binary from `fraud/models/llama-b11960/` was blocked by the
session's permission check. The owner chose to upgrade the winget llama.cpp instead: b11748 →
**b11977** (commit 7ac5727d6). `fraud/eval_lfm.py` now finds llama-server as Phase 6 does
(`d3.find_server`). Consequence: Phase 6 can no longer be re-run on its exact b11748 build. The
GGUF hashes were checked after download (both match the pins). Nothing else changed.

## 2026-10-30: Phase 8, LFM3-1.7B dropped (owner; during the dev runs, before any LFM3 trace)

Eval `20261030-122539` was stopped while d1-Omni-2B was on its dev windows (logreg and the first
d1-Omni windows finished). The owner dropped LFM3-1.7B as too slow: its smoke run took 454–1,061 ms
per transaction with both orders (5 dev0 rows, Observed), against 172–174 ms for d1-Omni-2B (also
both orders). No LFM3-1.7B dev trace exists, so there is no LFM3 result and nothing was seen that
could bias the rest. The eval resumes with `--resume 20261030-122539 --models d1-omni-2b`; the
d1-Omni-2B protocol is unchanged.

## 2026-10-30: Phase 8 results, d1-Omni-2B beats the frozen 1.5B on dev, gate FAIL

Eval `20261030-122539` (resumed with `--models d1-omni-2b`, about 9 min of d1-Omni runs). Write-up:
`fraud/RESULTS-LFM.md`. Observed:
- Dev AUC (2,194 model rows, 260 fraud): both 0.9840 / fwd 0.9830 / rev 0.9847 → rev (gain −0.0007).
  vs frozen 1.5B 0.9028: **+0.0820, better**. TRANSFER 0.9385, CASH_OUT 1.0000.
- Gate (>= logreg 0.9916): **FAIL**. The test windows were not run.
- Dev cost with dev thresholds (0.5 / 0.7): 269,831 per window; frozen 1.5B 2,701,634 (0.7 / 0.9).
- Latency dev0 p90: both orders 200 ms, one order 91 ms.
- LFM3-1.7B: no result (dropped by the owner before any trace). Phase 8 done.

## 2026-11-02: Phase 9 pre-statement, d1-Omni-2B fine-tuned for one hour (owner; before training)

Owner's request: "train d1 omni for an hour". Everything else follows Phase 7 (Implementation choice):
the full pre-registration is the `fraud/eval_d1ft.py` docstring, committed with this entry.
- Runtime (Observed): transformers has no LFM3 class up to 5.22.1, so the only ready-made torch model is the
  repo's remote code, which this project does not run (Phase 8). `fraud/d1torch.py` re-implements the text path
  from reading that source at revision 431f1d667d: layers, prompt, "set" masks, code readout, temperature.
  Only the weights, tokenizer, config and calibration files are downloaded, to `fraud/models/d1-omni-hf/`.
- Reproduction check first, as Phase 7: frozen torch vs the Phase 8 llama-server traces on dev0–dev9, packed
  the way the server packs both orders. R1 tokens, R2 AUC within 0.01, R3 median score within 0.02. A failure
  stops training until the owner has seen why.
- Training: Phase 5's rows, LoRA r16 on attention and convolution projections, lone prompts. CE on the
  temperature-scaled scores (3.383), the function the decider reads. N from the smoke run's s/step; the process
  takes at most one hour of wall clock. No minimum N: the hour is what was asked for. `peft` is used
  as in Phases 5 and 7 (`finetune_d1.py` imports it through `finetune_d3.train`; `d1torch.merge_adapter`).
- Eval: lone prompts per order, debias rule, gate vs logreg dev AUC, thresholds on dev, the 20 Phase 5 windows
  if the gate passes, bars A (vs logreg), B (p90 <= 250 ms), C (vs hybrid-ft).
- Noted while reading the source (Inferred): llama-server packs a body's questions after one state, so Phase 8's
  "rev" scores were read after the forward question (token count 338 for two orders). Its fwd scores are close
  to the lone prompt (smoke: within 0.001). Phase 8 reached no test windows, so nothing there relied on it.
- Stated before running (Inferred): one hour is likely near half an epoch, Phase 7's run was 90%.

## 2026-11-02: Phase 9 correction, before the reproduction check ran

The pre-statement above says llama-server packs both orders after one state. That Inference was wrong. Observed:
the Phase 8 traces report 169 tokens for one order and 338 for two, and the server's source sets `no_batching` for
LFM3 d1 and reads each question as its own prompt (Documented, server-decision.cpp at 7ac5727d6). So Phase 8's "rev"
scores were not read after the forward question, and its rev variant needs no caveat. A 6-row check of the frozen
model showed this (packed: 216 tokens, scores 0.08 off; lone: 338 tokens as the traces, scores about 0.01 off). The
check used dev0 rows and only tested the prompt format. `fraud/d1torch.py` lost its packing, and the reproduction check
in the `fraud/eval_d1ft.py` docstring now compares lone prompts. R1–R3 and their tolerances did not change.

## 2026-11-02: review-tier idea, a no-training check (owner; exploratory, dev and training steps only)

Owner's idea: a second decision model reviews in place of the human, escalating when unsure, so a correct call costs
0 instead of 50. All of `hybrid-ft`'s test cost is review fees (332 reviews, 830 per window, Observed), so this
targets the whole remaining cost. `fraud/reviewer_check.py`, results in `fraud/results/reviewer_check/`. Observed:
- Dev queue (the rules' review rows): 173 rows, 128 fraud, 45 legit, 2.85% fraud at the natural rate. AUC on it:
  logreg 0.56, frozen d1-Omni 0.47, frozen d3-mini 0.43, hybrid-ft 0.56, hybrid-d3ft 0.58. These are the same models
  whose AUC is 0.94–1.00 over all rows; the queue is what is left after their cues are used.
- Cross-fold blocking loses money for every scorer but d3-mini, which blocks nothing: one wrong block costs more than all
  the reviews it saves. Blocking is not viable on these signals, and approving even less (a miss costs about 1.6M).
- Training-step queue: 40,133 rows, 1,649 fraud (4.1%). Amount alone AUC 0.505, the count signals 0.50, hour 0.37 (fraud
  is flat over the day, legit is not). At decision time the queue's fraud barely differs from its legit.
- After the decision: a CASH_OUT of exactly this amount follows in the same hour for 99.76% of fraud and 0.010% of legit.
Inferred: the review queue is not hard for lack of a better model; the signal arrives after the transfer, so a reviewer
on the same inputs cannot save much however it is trained. The follow-up cash-out is a PaySim property (its generator
scripts transfer-then-cash-out with equal amounts) and is unlikely to be this clean on real data. Using it means a design
change, not a tuning step: the no-look-ahead rule, and the cost model, which charges a fraud TRANSFER's approval as its
full amount even though the existing cash-out rule then stops the money. Those are the owner's decisions. Nothing built.

## 2026-11-02: Phase 10 pre-statement, no review by F1 (owner; before any test-split hour)

Owner: minimise review, forget cost, focus on F1. Exploratory on training steps 1–300, natural rate (Observed): the
cash-out rule needs only that an earlier TRANSFER of exactly this amount sits in the hour to keep all 1,682 fraud cash-outs it
caught while its legit declines fall from 741 to 28. With the review queue approved that gave no reviews and chain F1 0.98
(one theft = a transfer + its cash-out), but row F1 0.67; today's rules with a human on all 40,133 queue rows have row F1 0.90.
These numbers are in-sample: the rule was found on these rows. The pre-registration is the `fraud/eval_chain.py` docstring:
untouched test-split hours, once; bar 1 tight chain F1 > human chain F1; bar 2 tight chain F1 >= 0.90; row F1 reported.
Stated before running (Inferred): it will shrink, and a pass says the rule fits PaySim, whose generator seems to cash out
the exact amount within the hour. It says nothing about real fraud. The chain metric assumes money can be stopped at the cash-out.

## 2026-11-02: Phase 9 results, gate and bars A/B/C PASS, no gain over the rules

Eval `20261102-162345`, write-up `fraud/RESULTS-D1FT.md`. Observed: reproduction check PASS; full epoch in 56.1 min; dev AUC 0.9953 (frozen
0.9847, frozen 1.5B 0.9028, logreg 0.9916); bars A 19/20, B 92 ms, C 20/20; mean test cost 810 (rules 808, hybrid-ft 830,
hybrid-d3ft 835); 324 reviews, no fraud approved, no legit declined. The first report build mislabelled the setup `approve-all`
and broke the paired table (a loop variable reused `setup` in `eval_d3ft.report` after this phase's refactor); fixed, rebuilt from the
same traces, bar values unchanged. Review-tier check with this model (`20261102-163454`): queue AUC 0.59; blocking still loses money.

## 2026-11-02: Phase 10 results, bar 1 PASS, bar 2 FAIL

Eval `20261102-163455`, write-up `fraud/RESULTS-CHAIN.md`. Observed on 282 untouched hours: tight chain F1 0.578 with no reviews, human 0.558
with 17,042 reviews, floor 0.90 not met; in-sample it was 0.98. Of tight's 2,183 wrong declines, 5 came from the tightened cash-out line and
2,178 from the old round-transfer line (2,176 legit transfers of exactly 10,000,000, mostly hours 304–305). That line's evidence was 39
training rows. Nothing removed or re-run: it would be a change prompted by test hours, and no untouched hours remain.


## 2026-11-02: console shows the fine-tuned d1-Omni-2B (owner)

`fraud/viewer/server.py` lists `fraud/results/eval_d1ft/` (`EVAL_DIRS`); `console.js` has the `hybrid-d1ft` label, chart series (it shares the
fifth color with `hybrid-d3ft`: one page never shows both) and write-up link; the training card no longer names a 2-hour cap. Phase 10 has no
eval page: its result has no per-window setups. Checked in the browser pane (Observed): start page, the new eval page (20 of 20 trace links), the
d3-nano eval page, a d1-Omni replay, no console errors, no sideways scroll at 375 px.


## 2026-11-02: Phase 11 pre-statement, first real data (owner; before any model scored an IEEE row)

Owner: test one random frozen and one random fine-tuned model on IEEE-CIS, to see which is better and whether retraining is necessary. The owner
placed the dataset in `fraud/data/ieee-cis/` and did not answer the handoff's three open questions, so they are Implementation choices in the
`fraud/eval_ieee.py` docstring: the adapter runs as is (transfer, not a fine-tune on real data); costs in USD with a 5 USD review, friction 10% at least 1
USD, sensitivity reported including PaySim's numbers as they are; retraining counts as not necessary only if the better model is no dearer than an
IEEE-fitted logistic regression in >= 10 of 20 windows and beats approve-all. All three can be re-read from the traces if the owner decides otherwise.
Draw, seed 20261102: frozen Qwen2.5-1.5B, fine-tuned d1-Omni-2B. Different base models, so the comparison changes two things at once (stated now).
Stated before running (Inferred): both likely lose to logistic regression.
