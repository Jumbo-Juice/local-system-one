"""Checks an eval from its raw traces, apart from the report code. Imports nothing from the fraud app but the trace folder's location.

    python -m fraud.dev.verify_tabular <results json> [<eval id>]

For every setup and window in the results file: replays each decision from the trace's score and the header's rule or lines, replays the review budget, re-prices with its own arithmetic, recounts customers, and compares with the
trace's own actions and with the results file. Then a second bootstrap with another seed for each pre-registered comparison. Prints what differs. Costs are read from the trace header.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def decide_rule(q, a, pol, c):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    z = (np.log(q / (1 - q)) - pol["fit_mean"][0]) / pol["fit_sd"][0]
    p = 1 / (1 + np.exp(-(pol["fit_weights"][0] + pol["fit_weights"][1] * z)))
    appr, decl = p * a, (1 - p) * (np.maximum(c["friction_min"], c["friction_rate"] * a) + pol.get("flat", 0.0))
    fb = np.where(appr <= decl, "approve", "decline")
    if pol.get("gamma"):
        n, left, wanted = len(a), int(np.floor(c["review_share"] * len(a))), []
        low = np.minimum(appr, decl)
        for i in range(n):
            if left > 0 and c["review_fee"] + pol["lambda"] * ((left / (n - i)) / c["review_share"]) ** -pol["gamma"] < low[i]:
                wanted.append("review")
                left -= 1
            else:
                wanted.append(str(fb[i]))
        return wanted, fb.tolist()
    return np.where(c["review_fee"] + pol["lambda"] < np.minimum(appr, decl), "review", fb).tolist(), fb.tolist()


def replay(path: Path) -> tuple[float, dict, int]:
    recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    head, tx = recs[0], [r for r in recs if r["type"] == "tx"]
    c = head["costs"]
    q = np.array([r["q"] for r in tx])
    a = np.array([r["tx"]["amount"] for r in tx])
    y = [r["truth"] for r in tx]
    if head.get("policy") and head["policy"].get("kind") == "amount-aware":
        wanted, fallback = decide_rule(q, a, head["policy"], c)
    elif head.get("policy"):
        return float(recs[-1]["summary"]["cost"]), {"skipped": 1}, 0  # a rule this script doesn't know: the phase's own check covers it
    elif head.get("thresholds") and (head["thresholds"]["review"] is not None or head["thresholds"]["decline"] is not None):
        tr, td = head["thresholds"]["review"], head["thresholds"]["decline"]
        wanted = ["decline" if td is not None and v >= td else "review" if tr is not None and v >= tr else "approve" for v in q]
        fallback = ["approve"] * len(q)
    else:
        wanted, fallback = ["approve"] * len(q), ["approve"] * len(q)
    left, total, n = int(np.floor(c["review_share"] * len(q))), 0.0, {"good_declined": 0, "fraud_stopped": 0}
    bad = 0
    for w, f, yy, aa, r in zip(wanted, fallback, y, a, tx):
        act = w
        if act == "review":
            if left > 0:
                left -= 1
            else:
                act = f
        if act != r["action"]:
            bad += 1
        if act == "review":
            total += c["review_fee"]
        elif act == "approve":
            total += aa if yy else 0.0
        elif not yy:
            total += max(c["friction_min"], c["friction_rate"] * aa)
        n["good_declined"] += act == "decline" and not yy
        n["fraud_stopped"] += act != "approve" and bool(yy)
    return total, n, bad


def main() -> None:
    res = json.load(open(sys.argv[1], encoding="utf-8"))
    eval_id = sys.argv[2] if len(sys.argv) > 2 else res["eval"]
    windows = list(res["windows"])
    per, rows_bad, gap, wrong_count = {}, 0, 0.0, []
    for setup, row in res["setups"].items():
        got = [replay(ROOT / "runs" / f"{eval_id}_eval_{setup}_{w}.jsonl") for w in windows]
        per[setup] = np.array([g[0] for g in got])
        rows_bad += sum(g[2] for g in got)
        gap = max(gap, float(np.abs(per[setup] - np.array(row["windows"])).max()))
        if "skipped" not in got[0][1]:
            good, stopped = sum(g[1]["good_declined"] for g in got), sum(g[1]["fraud_stopped"] for g in got)
            if good != row["good_customers_declined"] or stopped != row["frauds_stopped"]:
                wrong_count.append((setup, good, row["good_customers_declined"], stopped, row["frauds_stopped"]))
    print(f"{len(res['setups'])} setups, {len(windows)} windows: {rows_bad} rows decided differently, largest cost gap {gap:.2f} USD per window, {len(wrong_count)} customer counts differ")
    for w in wrong_count:
        print(" ", w)
    comps = []
    for k, v in res["verdict"].items():
        comps += v if isinstance(v, list) else [v]
    rng = np.random.default_rng(99)
    for p in comps:
        d = per[p["better"] or p["a"]] - per[p["other"]]
        m = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
        print(f"  {p['better'] or p['a']} - {p['other']}: mean {d.mean():.1f} (results {p['mean_diff']:.1f}), CI {np.percentile(m, 2.5):.0f} to {np.percentile(m, 97.5):.0f} (results {p['mean_diff_ci95'][0]:.0f} to {p['mean_diff_ci95'][1]:.0f})")


if __name__ == "__main__":
    main()
