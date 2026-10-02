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
