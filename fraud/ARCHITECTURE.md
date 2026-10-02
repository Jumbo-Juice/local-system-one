# ARCHITECTURE: the fraud app

## Stack
| Layer | Choice | Why |
|---|---|---|
| Engine | `system_one` from this repo (`make_engine`, `decide_batch`, `order_debias`) | The thing being showcased |
| Model | Qwen2.5-1.5B-Instruct (`config/default.toml`, LM head in float32) | Repo default; the latency bar is sized for it. The 3B was dropped: it does not fit the free iGPU memory (`fraud/PROGRESS.md`) |
| Data | PaySim CSV → `fraud/data/paysim.npz` (numpy) | No pandas; the cache loads in about a second |
| Baselines | numpy only | No new dependencies (`fraud/CLAUDE.md`) |
| Viewer | stdlib `http.server` + one HTML page + plain JS | Same approach as the game viewer, separate code |
| Tests | pytest, `fraud/tests/` (added to `pytest.ini` testpaths) | One `pytest` run covers both apps |

## Modules (`fraud/`)
| file | does |
|---|---|
| `data.py` | reads the CSV (drops the balance columns), writes or reads the `.npz` cache; `python -m fraud.data` |
| `signals.py` | one pass in file order → per-row signals that use earlier rows only |
| `windows.py` | train/test cut; seeded enriched windows; weights back to the natural rate |
| `costs.py` | the cost model and review budget; per-window scoring |
| `rules.py` | the type filter and the rules-only table |
| `baselines.py` | logistic regression (numpy), random, approve-all |
| `brain.py` | transaction + signals → `Decision` prompt; model decisions with timing |
| `runs.py` | trace paths and writing (`fraud/runs/`) |
| `capture.py` | one window, one setup → one trace; `python -m fraud.capture` |
| `eval.py` | the pre-registered eval → `fraud/results/eval/`; `python -m fraud.eval` |
| `viewer/` | `server.py`, `index.html`, `console.js`; `python -m fraud.viewer` |

## Data model
**Transaction** (one PaySim row, balances dropped): `row` (index in the CSV = arrival order),
`step` (hour 1..743), `type` (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), `amount`, `orig`,
`dest` (account ids as ints; the first letter of a PaySim name, C = customer, M = merchant, kept as
`dest_merchant`), `is_fraud` (truth), `flagged` (PaySim's own `isFlaggedFraud`: a transfer over
200,000).

**Signals** (row *i*, from rows before *i* only): hour of day; the origin's prior transaction count;
the destination's prior incoming count and incoming amount; whether the destination had received a
TRANSFER earlier; whether the same amount appeared earlier in the same step; whether the amount is
round. The final list is fixed in Phase 1 and recorded here.

**Window**: `id`, `seed`, `split` (dev or test), the row indices in arrival order, and the
per-class weights that map its counts back to the natural fraud rate of its time span.

**Action**: `approve` | `review` | `decline`. Each decision records who made it (`filter`, `rules`,
`model`, `logreg`, `random`, `all`), the option scores (for the model: the order-averaged softmax
and both orders), `outside_mass`, prompt tokens and the measured latency in ms.

**Cost model** (constants fixed in `fraud/costs.py` in Phase 1, before any test window):
approving fraud loses its amount; declining a legit transaction costs a fixed friction fee; a
review costs a fixed analyst fee and finds the truth (fraud blocked, legit approved). Reviews have
a budget (a share of the window's transactions, in arrival order). Once it is used up, a `review`
becomes the better-scored of approve/decline. Every setup is scored with the same function.

## Trace format (`fraud/runs/<time>_<label>.jsonl`)
Line 1, header: `{"type":"header","app":"fraud","setup","window":{...},"backend":{...},"costs":{...},
"created","example_prompt"}`. One line per transaction: `{"type":"tx","i","row","step","tx":{...},
"signals":{...},"by","action","scores","orders","outside_mass","prompt_tokens","ms","truth",
"cost","budget_left"}`. The last line: `{"type":"end","summary":{...}}` with totals, the confusion
matrix, latency p50/p90 and the natural-rate reweighted cost.

The replay clock is the sum of the recorded `ms`. At 1× each transaction appears when its decision
finished in the recorded run.

## Security and data boundary
- Everything is local. No remote inference and no telemetry. The viewer binds to 127.0.0.1 by default.
- PaySim is synthetic and has no PII. Raw rows and the cache stay in `fraud/data/` (git-ignored).
  Traces hold single rows from windows. Only pinned traces are committed (CC BY-SA attribution in
  `fraud/runs/README.md`).
- The viewer serves only `fraud/viewer/*`, `fraud/runs/*.jsonl` and `fraud/results/**/*.json`.
