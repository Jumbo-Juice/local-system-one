"""The gate on a new machine, before any new work (owner, 2026-11-03) → fraud/results/gate_nuc/gate.json.

    python -m fraud.dev.gate_nuc --logreg       # gate 3: refit logreg, against Phase 11's dev scores and Phase 14's test scores
    python -m fraud.dev.gate_nuc --sha          # gate 4: the adapter file against its manifest, D1Torch.load's own sha check
    python -m fraud.dev.gate_nuc --score        # gate 5: Phase 12's adapter on dev0 here, against its trace; latency here

No new window and no new label: everything read is already in a trace. Gate traces are not written to fraud/runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .. import eval_ieee as ev, eval_ieee_amount as am, ieee_data
from ..runs import read_run

OUT = Path(__file__).resolve().parents[1] / "results" / "gate_nuc" / "gate.json"
ADAPTER = Path(__file__).resolve().parents[1] / "models" / "20261103-091222_lora-d1-ieee-full"
PHASE12_DEV0 = ev.trace_file(am.PHASE12, "dev", am.FT, "dev0")
PHASE14 = "20261103-164830"


def save(key: str, value: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps({key: value}, indent=1), flush=True)


def gate_logreg() -> None:
    t = ieee_data.load()
    n = len(t)
    t0 = time.perf_counter()
    lr = ev.fit_logreg(t, ev.features_all)
    fit_s = time.perf_counter() - t0
    gaps = {}
    for w in am.old_dev(n):
        recorded = ev.view([read_run(ev.trace_file(am.PHASE11, "dev", am.LR, w.id))])["scores"][0]
        gaps[w.id] = float(np.abs(lr.predict(ev.features_all(t, w.rows)) - recorded).max())
    test = {}
    for w in am.test_windows(n):
        recorded = ev.view([read_run(ev.trace_file(PHASE14, "eval", am.LR, w.id))])["scores"][0]
        test[w.id] = float(np.abs(lr.predict(ev.features_all(t, w.rows)) - recorded).max())
    worst = max(gaps.values())
    save("logreg", {"fit_s": round(fit_s, 1), "features": int(len(lr.w) - 1), "dev_max_gap": worst, "dev_tolerance": 1e-6, "pass": worst <= 1e-6,
                    "phase14_test_max_gap": max(test.values()), "dev_gaps": gaps})


def gate_sha() -> None:
    from ..d1torch import D1Torch

    man = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
    got = hashlib.sha256((ADAPTER / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    t0 = time.perf_counter()
    model = D1Torch.load()  # raises if a pinned sha256 differs
    save("sha", {"adapter_file": got, "manifest": man["adapter_sha256"], "adapter_pass": got == man["adapter_sha256"], "starts_as_stated": got.startswith("10970d702ee9990f"),
                 "base_load_s": round(time.perf_counter() - t0, 1), "base_sha_check": "passed", "device": str(model.device)})


def gate_score() -> None:
    import torch

    t = ieee_data.load()
    w = am.old_dev(len(t))[0]
    recorded = [r for r in read_run(PHASE12_DEV0) if r.get("type") == "tx"]
    model = ev.D1Fine(ADAPTER)
    both = [model.score(t, int(i), "both") for i in w.rows]
    rev = [model.score(t, int(i), "rev") for i in w.rows]
    y = np.array([r["truth"] for r in recorded])
    if not np.array_equal(np.array([r["row"] for r in recorded]), w.rows):
        raise SystemExit("dev0's rows differ from the trace")
    was_fwd, was_rev = (np.array([r["orders"][k]["decline"] for r in recorded]) for k in (0, 1))
    here_fwd, here_rev = (np.array([o["orders"][k]["decline"] for o in both]) for k in (0, 1))
    tr, td = am.s10_thresholds(am.FT)
    decide = lambda q: np.array([ev.act(float(v), tr, td) for v in q])
    head = read_run(PHASE12_DEV0)[0]
    ms = np.array([o["ms"] for o in rev[50:]])
    save("score", {"window": w.id, "device": str(model.model.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "platform": platform.platform(), "torch": torch.__version__,
                   "rev_max_gap": float(np.abs(here_rev - was_rev).max()), "rev_mean_gap": float(np.abs(here_rev - was_rev).mean()),
                   "fwd_max_gap": float(np.abs(here_fwd - was_fwd).max()), "fwd_mean_gap": float(np.abs(here_fwd - was_fwd).mean()),
                   "rev_vs_fresh_rev_max_gap": float(np.abs(np.array([o["q"] for o in rev]) - here_rev).max()),
                   "auc_rev_trace": ev.auc(was_rev, y), "auc_rev_here": ev.auc(here_rev, y), "auc_fwd_trace": ev.auc(was_fwd, y), "auc_fwd_here": ev.auc(here_fwd, y),
                   "thresholds": [tr, td], "decisions_differ_rev": int((decide(was_rev) != decide(here_rev)).sum()), "rows": len(y),
                   "latency_here_rev_ms": {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99))},
                   "trace_backend_device": head["backend"].get("device")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logreg", action="store_true")
    ap.add_argument("--sha", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.logreg:
        gate_logreg()
    if a.sha:
        gate_sha()
    if a.score:
        gate_score()


if __name__ == "__main__":
    main()
