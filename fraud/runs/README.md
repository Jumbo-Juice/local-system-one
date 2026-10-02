# Fraud runs

One trace per run, `<time>_<label>.jsonl`, written by `fraud/runs.py` (captures and evals). Runs
are git-ignored; only `*.pinned.jsonl` is committed, and each one is listed here with the reason.
Watch any run with `python -m fraud.viewer`. The trace format is in `fraud/ARCHITECTURE.md`.

Traces hold single PaySim rows (balance columns excluded). PaySim: E. A. Lopez-Rojas, A. Elmir and
S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection", Kaggle
`ealaxi/paysim1`, licence CC BY-SA 4.0. The pinned traces are shared under the same licence.

| pinned run | why |
|---|---|
| `20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl` | The console's example and the auto-demo's first run: the hybrid with Qwen2.5-1.5B (prompt v4, float32 head, order debias) on dev window dev0. It shows the main finding: the model declines all 257 transactions it judges, at p50 125 ms / p90 126 ms. 32 s at 1×. |
