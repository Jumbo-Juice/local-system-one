"""Paper revision, Stage 0: every IEEE-CIS test-split window any eval has scored, and what is left → fraud/results/window_audit.json.

    python -m fraud.dev.window_audit

Reads no label. Windows are rebuilt from the code that drew them (fraud/eval_ieee.py, eval_ieee_full.py, eval_ieee_amount.py,
final_test.py) and checked row for row against the [first row, last row] each results JSON recorded. "Scored" means an eval
computed decisions and costs on the rows, so their labels were read. A row whose features were read only to build history for
later rows, or that has a cached tree score with no label read, is not scored (fraud/final_test.py states the same rule).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import eval_ieee as ev, eval_ieee_amount as am, eval_ieee_full as fu, final_test as ft, ieee_data, tabular as tb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "window_audit.json"

# (name, phase, script, results JSON, how the windows are built)
SOURCES = (
    ("phase11", "Phase 11", "fraud/eval_ieee.py", "fraud/results/eval_ieee/20261102-183431.json"),
    ("phase12", "Phase 12", "fraud/eval_ieee_full.py", "fraud/results/eval_ieee_full/20261103-142622.json"),
    ("phase14", "Phase 14 (and Phase 16 cost stress, re-priced from Phase 14's traces)", "fraud/eval_ieee_amount.py", "fraud/results/eval_ieee_amount/20261103-164830.json"),
    ("audit", "Phase 17 column audit", "fraud/eval_column_audit.py", "fraud/results/eval_column_audit/20261103-210600.json"),
    ("rule", "Phase 18 rule tests", "fraud/eval_rule.py", "fraud/results/eval_rule/20261103-211736.json"),
    ("features", "Phase 19 feature tests", "fraud/eval_features.py", "fraud/results/eval_features/20261103-212822.json"),
    ("tabular", "Phase 20 tabular tests (and its first run, deleted unread, on the same chains)", "fraud/eval_tabular.py", "fraud/results/eval_tabular/20261103-235132.json"),
    ("final", "Phase 23 final test (and the read-outs after it)", "fraud/eval_final.py", "fraud/results/eval_final/20261104-090649.json"),
)


def windows(n: int, name: str) -> list[ev.Win]:
    if name == "phase11":
        return ev.make_windows(n, "test", ev.TEST_WINDOWS)
    if name == "phase12":
        return fu.new_windows(n)
    if name == "phase14":
        return am.test_windows(n)
    if name == "final":
        return ft.final_windows(n)
    return ft.phase_windows(n, name)


def runs_of(mask: np.ndarray, offset: int) -> list[list[int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [[int(a) + offset, int(b) + offset] for a, b in zip(edges[::2], edges[1::2])]


def main() -> None:
    t = ieee_data.load()
    n = len(t)
    lo, hi = ieee_data.split_rows(n, "test")
    d_lo, d_hi = ieee_data.split_rows(n, "dev")
    hits = np.zeros(hi - lo, np.int32)
    sources = []
    for name, phase, script, results in SOURCES:
        ws = windows(n, name)
        recorded = {k: v for k, v in json.loads((ROOT.parent / results).read_text(encoding="utf-8"))["windows"].items() if not k.startswith(("dev", "xdev"))}
        matches = len(recorded) == len(ws) and all(recorded.get(w.id) == [int(w.rows[0]), int(w.rows[-1])] for w in ws)
        if not matches:
            raise SystemExit(f"{name}: rebuilt windows differ from {results}")
        rows = np.concatenate([w.rows for w in ws])
        if rows.min() < lo or rows.max() >= hi:
            raise SystemExit(f"{name}: rows outside the test split")
        hits[rows - lo] += 1
        sources.append({"name": name, "phase": phase, "script": script, "results": results, "windows_n": len(ws), "rows_n": int(len(rows)),
                        "matches_recorded": matches,
                        "windows": [{"id": w.id, "rows_n": int(len(w.rows)), "first_row": int(w.rows[0]), "last_row": int(w.rows[-1]),
                                     "contiguous": bool(w.rows[-1] - w.rows[0] + 1 == len(w.rows))} for w in ws]})
    free = hits == 0
    free_runs = runs_of(free, lo)
    dev_used = np.concatenate([w.rows for w in tb.dev30(n) + tb.heldout_dev(n)])
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "fraud/dev/window_audit.py",
        "label": "audit (no label read)",
        "dataset": "IEEE-CIS train_transaction.csv, time split by row count (fraud/ieee_data.py)",
        "split_rows": {"fit": list(ieee_data.split_rows(n, "fit")), "dev": [d_lo, d_hi], "test": [lo, hi]},
        "window_size": ev.WINDOW_SIZE,
        "test": {
            "rows": int(hi - lo),
            "full_windows_possible": int((hi - lo) // ev.WINDOW_SIZE),
            "windows_scored": int(sum(s["windows_n"] for s in sources)),
            "rows_scored": int((hits > 0).sum()),
            "rows_scored_twice": int((hits > 1).sum()),
            "rows_never_scored": int(free.sum()),
            "never_scored_runs": free_runs,
            "untouched_windows": int(sum((b - a) // ev.WINDOW_SIZE for a, b in free_runs)),
            "untouched_windows_if_chains_allowed": int(free.sum() // ev.WINDOW_SIZE),
        },
        "dev": {
            "rows": int(d_hi - d_lo),
            "windows_used": {"dev30 (rule and line fits)": len(tb.dev30(n)), "heldout_dev (hdev, the 88 dev chains)": len(tb.heldout_dev(n))},
            "rows_never_used": int(d_hi - d_lo - len(np.unique(dev_used))),
        },
        "sources": sources,
        "notes": [
            "Phases 11, 12, 14 and FINAL use 1,000 contiguous rows. Phases 17-20 use chains: 1,000 untouched rows in arrival order that skip rows used earlier, so a chain spans more than 1,000 rows (fraud/final_test.py).",
            "Not counted as scored: fraud/data/ieee-cis/signal.npz holds the Phase 20 tree score (tab-all) for every fit and dev row and for the FINAL rows, about 16.9% of test rows; no label is read to make it.",
            "Not counted as scored: fraud/history.py reads feature columns of every row to build history for later rows.",
            "Phase 16 (fraud/eval_cost_stress.py) and the post-FINAL read-outs (fraud/results/eval_final/rich_cust_readout.json, extras.json) re-price saved traces of Phase 14 and FINAL windows; they add no window.",
            "Phases 1-10 ran on PaySim and are outside this audit.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    s = out["test"]
    print(f"test rows {s['rows']:,}: {s['windows_scored']} windows scored, {s['rows_scored']:,} rows; {s['rows_never_scored']} rows never scored; "
          f"{s['untouched_windows']} untouched windows of {ev.WINDOW_SIZE:,} -> {OUT}")


if __name__ == "__main__":
    main()
