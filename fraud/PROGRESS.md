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
- Observed facts are in `fraud/docs/data.md`. The ones that shape the build:
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
test100–test119 once (about 9 min). Full write-up: `fraud/docs/results.md`. Observed:
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
