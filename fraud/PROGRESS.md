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
