"""The language model on this machine's GPU: a second seed, a second pass, more prompt columns, frozen d1-Omni-2B. Dev only → fraud/results/eval_lm_nuc/.

    python -m fraud.finetune_ieee --train --full --seed 1     # one adapter per option (fraud/models/<time>_lora-d1-ieee-full-*)
    python -m fraud.finetune_ieee --train --full --rich
    python -m fraud.eval_lm_nuc --score <setup> [<adapter dir>]   # score the dev rows here, into fraud/data/ieee-cis/lm_scores/ (git-ignored)
    python -m fraud.eval_lm_nuc --report                          # the comparison, from the saved scores and refits

Results: fraud/results/eval_lm_nuc/report.json + .md, dev rows only.

PRE-REGISTRATION (Phase 21; committed before any new adapter was trained and before this script scored a row).
Owner, 2026-11-03: "language model, given this machine's hardware: more prompt columns with retraining, a second pass, a second seed, frozen d1-Omni-2B on IEEE windows, larger models if they fit. Each must keep p90 <= 250 ms. Estimate the run time first and tell me if a run exceeds about 8 hours. The rule above already beats the adapter, so run these after the tabular work unless evidence says otherwise."
- Hardware (Observed): RTX 3060 Ti, 8 GB. Smoke, while the CPU was busy: 0.82 s per batch, peak 3.79 GiB. Estimate: one pass 82 min, the rich prompt about 85 min, two passes 164 min, so about 5.5 h of training, plus about 20 min to score each setup. No single run passes 8 hours.
- Setups, each d1-Omni-2B in option order `rev` as Phase 12 chose: lm-phase12 (Phase 12's adapter, re-scored here, so no two machines are mixed); lm-seed1 (its recipe, seed 1, a different draw of legit rows); lm-rich (the same rows and recipe with ieee_brain.describe_rich, which adds six counters, five day columns, dist1 and what earlier rows say about the card and address; no label; one prompt, written before any model read it, never tuned on scores); lm-pass2 (seed 0, two passes, LR schedule over both); lm-frozen (no adapter).
- Larger models: none is on disk, and each model here has needed its own reader, readout and reproduction check. Not run in this phase unless a smaller option shows the language model gaining (Implementation choice).
- Dev only. The 38 chains are spent on Phases 17-20. Fit rule and lines on Phase 14's 30 dev windows, read on the 88 held-out dev chains, against the best tabular setup from Phase 20 and logreg on all columns on the same rows.
- Scoring in batches of 16 for speed, on the rows' scores only. So batching doesn't hide a difference: a check on 500 rows against lone prompts is reported, largest score gap and AUC. Latency is measured on lone prompts, 1,000 rows of dev0, after the CPU is free, against the bar p90 <= 250 ms.
- The question: does any option beat the best tabular setup on held-out dev, in rule cost and in AUC? An option goes on to Phase 22 only if its rule cost is lower by the CI of the mean paired difference excluding 0, or if a blend with the best tabular setup is. Costs: friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed.
- Limits stated now: dev isn't test, and Phases 14 and 18 both found dev's ordering didn't carry over. One seed per option, except the seed comparison itself.
- Stated before running (Inferred): no option beats the best tabular setup. The seeds differ by about 0.005 AUC. The rich prompt gains the most, about 0.01-0.02 AUC. A second pass gains little. Frozen is far behind.

AMENDMENT, owner, 2026-11-03, before any new adapter was scored: skip the two-pass run, to finish about 3 hours sooner. lm-pass2 is withdrawn and was not trained. Everything else stands.

AMENDMENT, owner, 2026-11-03, before any new adapter was scored: the final setup must be a language decision model at heart, and yes, it may read a tabular risk score as a signal. So:
- New setup lm-signal: the rows and recipe of lm-rich, with one more prompt line, the tabular score from fraud/lm_signal.py (Phase 20's trees on all columns at 400 trees, out of fold on training rows). The language model still decides. One prompt, written before any model read it.
- The question changes. No language model now needs to beat the trees to reach the final test. The two best language-model setups on held-out dev go on, chosen by rule cost with lambda from all 118 dev windows as for trees, and each stays subject to p90 <= 250 ms.
- What the language model adds is reported, whatever it shows: lm-signal against its signal alone, on the same rows and the same rule, in cost, AUC and customers. A setup that only echoes the score must be reported as that, not as a language-model win.
- Stated before running (Inferred): lm-signal is the cheapest language-model setup, close to the trees, and adds little or nothing on top of its signal. lm-rich gains on lm-phase12 but stays dearer than the trees. Trained on its own signal, the model may lean on that line almost entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_ieee_amount as am, history, ieee_brain, ieee_data, tabular as tb
from .brain import OPTIONS
from .runs import read_run

RESULTS = Path(__file__).resolve().parent / "results" / "eval_lm_nuc"
SCORES = ieee_data.RAW / "lm_scores"
BATCH = 16
VARIANT = "rev"
PHASE12_ADAPTER = Path(__file__).resolve().parent / "models" / "20261103-091222_lora-d1-ieee-full"


def needs_history(setup: str) -> bool:
    return setup in ("lm-rich", "lm-signal")


def order_of(variant: str) -> tuple[str, ...]:
    return OPTIONS[::-1] if variant == "rev" else OPTIONS


def decline_at(variant: str) -> int:
    return order_of(variant).index(OPTIONS[2])


def score_rows(model, t, rows: np.ndarray, variant: str = VARIANT, hist: dict | None = None, batch: int = BATCH, log=print) -> np.ndarray:
    """The decline score per row, batches of similar lengths. Same prompt as the lone reader, padded."""
    import torch

    opts = list(order_of(variant))
    seqs = []
    for i in rows:
        d = ieee_brain.decision(t, int(i), hist)
        seqs.append(model.encode(f"{d.context}\n\n{d.state}", d.instruction, opts))
    order = np.argsort([len(s) for s in seqs], kind="stable")
    at = decline_at(variant)
    q = np.empty(len(rows))
    t0 = time.perf_counter()
    for k in range(0, len(order), batch):
        part = order[k:k + batch]
        with torch.no_grad():
            p = torch.softmax(model.logits([seqs[j] for j in part], len(opts)), dim=-1)[:, at].cpu().numpy()
        q[part] = p
        if log and (k // batch) % 500 == 0:
            log(f"{k + len(part):,} of {len(order):,} rows, {time.perf_counter() - t0:.0f} s")
    return q


def load_model(adapter: Path | None):
    from .d1torch import D1Torch
    from .eval_d3ft import warmed

    m = D1Torch.load()
    return warmed(m.merge_adapter(adapter / "adapter") if adapter else m)


def score(setup: str, adapter: Path | None, log=print) -> dict:
    t = ieee_data.load()
    n = len(t)
    windows = tb.dev30(n) + tb.heldout_dev(n)
    rows = np.unique(np.concatenate([w.rows for w in windows]))
    hist = history.load() if needs_history(setup) else None
    if setup == "lm-signal":
        from .lm_signal import load as load_signal

        hist["signal"] = load_signal()
    model = load_model(adapter)
    t0 = time.perf_counter()
    q = score_rows(model, t, rows, hist=hist, log=log)
    s_score = time.perf_counter() - t0
    sample = rows[:500]
    lone = np.empty(len(sample))
    for j, i in enumerate(sample):
        body = ieee_brain.request(t, int(i), VARIANT, hist)
        lone[j] = model.post(body)["answers"][VARIANT]["probabilities"][OPTIONS[2]]
    y = t.is_fraud[sample].astype(int)
    d0 = am.old_dev(n)[0]
    ms = []
    for i in d0.rows:
        body = ieee_brain.request(t, int(i), VARIANT, hist)
        t1 = time.perf_counter()
        model.post(body)
        ms.append((time.perf_counter() - t1) * 1000)
    ms = np.array(ms[50:])
    info = {"setup": setup, "adapter": adapter.name if adapter else None, "rows": int(len(rows)), "score_s": round(s_score, 1), "batch_vs_lone_max_gap": float(np.abs(lone - q[:500]).max()), "batch_vs_lone_mean_gap": float(np.abs(lone - q[:500]).mean()),
            "batch_auc": ev.auc(q[:500], y), "lone_auc": ev.auc(lone, y), "latency_ms": {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99))}, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES / f"{setup}.npz", rows=rows, q=q, info=np.array(json.dumps(info)))
    log(json.dumps(info))
    return info


def saved(setup: str, n: int) -> tuple[np.ndarray, dict]:
    with np.load(SCORES / f"{setup}.npz") as z:
        q = np.full(n, np.nan)
        q[z["rows"]] = z["q"]
        return q, json.loads(str(z["info"]))


def trace_check(q: np.ndarray, n: int) -> dict:
    """lm-phase12 here against its dev0 trace from the laptop, for the record. Not a comparison between machines."""
    w = am.old_dev(n)[0]
    was = ev.view([read_run(ev.trace_file(am.PHASE12, "dev", am.FT, w.id))], VARIANT)["scores"][0]
    return {"max_gap": float(np.abs(q[w.rows] - was).max()), "mean_gap": float(np.abs(q[w.rows] - was).mean())}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", nargs="+", metavar=("SETUP", "ADAPTER"), help="score the dev rows with this setup; no adapter for lm-frozen")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.score:
        score(a.score[0], Path(a.score[1]) if len(a.score) > 1 else None, log)


if __name__ == "__main__":
    main()
