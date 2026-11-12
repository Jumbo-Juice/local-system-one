"""How long the baseline takes to decide one payment → fraud/results/time_logreg/.

    python -m fraud.time_logreg

The final test (fraud/eval_final.py) timed every language-model setup on lone prompts and recorded none for
logreg: it scored logreg a window at a time, so its traces say 0 ms. This times the baseline the same way the
model was timed, one payment at a time.

METHOD (stated before running; Implementation choice):
- The baseline of the final test: final-logreg-amt's model, tab-logreg (logistic regression on all columns,
  fit rows only), fitted as fraud/eval_final.py fits it.
- Rows: the 30 dev windows (tabular.dev30), 30,000 rows. Not FINAL: only the final phase may score those
  (fraud/final_test.py), and a dot product's time doesn't depend on which row it reads.
- Timed per payment: building that row's features and the model's predict, called on one row at a time.
  The model's time (eval_final.LoneModel.score) covers its forward pass only, not building the prompt, so this
  counts more work for logreg than the model's time counts for the model.
- The first 200 calls warm up and aren't kept. Reported: p50, p90, p99 and mean in ms, and the machine.
- Not comparable hardware: the model was timed on an RTX 3060 Ti, this runs on whatever CPU runs it. The
  result file names the CPU, and every chart that uses it must say so.
- Stated before running (Inferred): under 1 ms at p90, so more than 50 times faster than the model's 56 ms.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from . import eval_tabular as et, history, ieee_data, tabular as tb

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "time_logreg"
WARMUP = 200


def run(log=print) -> dict:
    t = ieee_data.load()
    hist = history.load()
    kind, parts, drop = et.SETS["tab-logreg"]
    m, f = et.fit("tab-logreg", kind, parts, drop, t, hist, log)
    rows = np.concatenate([w.rows for w in tb.dev30(len(t))])
    ms = []
    for k, i in enumerate(rows):
        one = np.array([i])
        t0 = time.perf_counter()
        m.predict(f(t, one))
        if k >= WARMUP:
            ms.append((time.perf_counter() - t0) * 1000)
    ms = np.array(ms)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = {"run": stamp, "script": "fraud/time_logreg.py", "setup": "final-logreg-amt", "model": "tab-logreg",
           "rows": int(len(ms)), "warmup": WARMUP, "rows_from": "the 30 dev windows (tabular.dev30)",
           "latency_ms": {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)), "mean": float(ms.mean())},
           "timed": "one row's features and predict, one payment at a time",
           "machine": {"cpu": platform.processor() or platform.machine(), "system": f"{platform.system()} {platform.release()}",
                       "python": platform.python_version(), "numpy": np.__version__},
           "not_comparable_with": "the model's times in eval_final, taken on an RTX 3060 Ti"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"logreg, one payment at a time over {len(ms)} rows: p50 {out['latency_ms']['p50']:.3f} ms, p90 {out['latency_ms']['p90']:.3f} ms: {RESULTS / (stamp + '.json')}")
    return out


if __name__ == "__main__":
    run()
