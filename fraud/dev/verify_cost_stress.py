"""Checks Phase 16's results from the raw traces, apart from the report code. Imports no cost, rule or report code from the fraud app.

    python -m fraud.dev.verify_cost_stress <run id>

Reads the traces' amounts, truths and recorded decisions, re-prices them with its own arithmetic and its own budget replay, and compares with the results file (the "as deployed" view).
Then re-derives Phase 14's rule from the header's recorded fit for the "re-picked" view at the recorded lambda. Prints what differs, and a second bootstrap with another seed.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = "20261103-164830"


def load(setup: str) -> list[list[dict]]:
    files = [ROOT / "runs" / f"{EVAL}_eval_{setup}_amt{k}.jsonl" for k in range(20)]
    return [[json.loads(x) for x in open(p, encoding="utf-8")] for p in files]


def price(wanted, fallback, truth, amount, c) -> tuple[float, int]:
    """Total cost of one window and the good customers declined."""
    left, total, good = int(np.floor(c["review_share"] * len(wanted))), 0.0, 0
    for w, f, y, a in zip(wanted, fallback, truth, amount):
        if w == "review":
            if left > 0:
                left -= 1
                total += c["review_fee"]
                continue
            w = f
        if w == "approve":
            total += a if y else 0.0
        elif not y:
            total += max(c["friction_min"], c["friction_rate"] * a) + c.get("decline_flat", 0.0)
            good += 1
    return total, good


def columns(recs: list[dict]):
    tx = [r for r in recs if r["type"] == "tx"]
    return [r["wanted"] for r in tx], [r["fallback"] for r in tx], [r["truth"] for r in tx], [r["tx"]["amount"] for r in tx], [r["q"] for r in tx]


def rule(q, amount, head, c):
    """Phase 14's rule from the header's own fit, written out again."""
    pol = head["policy"]
    z = (np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))) - pol["fit_mean"][0]) / pol["fit_sd"][0]
    p = 1 / (1 + np.exp(-(pol["fit_weights"][0] + pol["fit_weights"][1] * z)))
    appr = p * amount
    decl = (1 - p) * (np.maximum(c["friction_min"], c["friction_rate"] * amount) + c.get("decline_flat", 0.0))
    fb = np.where(appr <= decl, "approve", "decline")
    return np.where(c["review_fee"] + pol["lambda"] < np.minimum(appr, decl), "review", fb).tolist(), fb.tolist()


def main() -> None:
    res = json.load(open(ROOT / "results" / "eval_cost_stress" / f"{sys.argv[1]}.json", encoding="utf-8"))
    bad, checked = [], 0
    for name, thr_setup, amt_setup in (("logreg", "ieee-logreg-all", "ieee-logreg-all-amt"), ("adapter", "ieee-ft-ieee-full", "ieee-ft-full-amt")):
        thr, amt = load(thr_setup), load(amt_setup)
        cols_t, cols_a = [columns(r) for r in thr], [columns(r) for r in amt]
        for family, rows in res["families"].items():
            for row in rows:
                c = row["costs"]
                got_r = [price(*x[:4], c) for x in cols_a]
                got_t = [price(*x[:4], c) for x in cols_t]
                dep = row[f"{name}_deployed"]
                mine = np.mean([g[0] for g in got_r]), np.mean([g[0] for g in got_t]), sum(g[1] for g in got_r), sum(g[1] for g in got_t)
                want = dep["rule_cost"], dep["thresholds_cost"], dep["rule_good_declined"], dep["thresholds_good_declined"]
                checked += 1
                if max(abs(a - b) for a, b in zip(mine, want)) > 0.51:  # trace amounts are rounded to cents
                    bad.append((name, family, row["label"], "deployed", mine, want))
                # Re-picked: re-derive the rule at the recorded lambda; thresholds at the recorded lines.
                rp = row[f"{name}_repicked"]
                pairs = [rule(np.array(x[4]), np.array(x[3]), {"policy": {**r[0]["policy"], "lambda": rp["lambda"]}}, c) for x, r in zip(cols_a if name == "logreg" else cols_a, amt)]
                got = [price(w, f, x[2], x[3], c) for (w, f), x in zip(pairs, cols_a)]
                tr, td = rp["thresholds"]
                act = lambda v: "decline" if td is not None and v >= td else "review" if tr is not None and v >= tr else "approve"
                got2 = [price([act(v) for v in x[4]], ["approve"] * len(x[4]), x[2], x[3], c) for x in cols_t]
                mine = np.mean([g[0] for g in got]), np.mean([g[0] for g in got2])
                checked += 1
                if max(abs(mine[0] - rp["rule_cost"]), abs(mine[1] - rp["thresholds_cost"])) > 0.51:
                    bad.append((name, family, row["label"], "repicked", mine, (rp["rule_cost"], rp["thresholds_cost"])))
    print(f"checked {checked} cost-set views; {len(bad)} differ by more than rounding")
    for b in bad[:20]:
        print(b)
    # A second bootstrap, another seed, on Phase 14's costs: should land near the recorded interval.
    thr, amt = [columns(r) for r in load("ieee-logreg-all")], [columns(r) for r in load("ieee-logreg-all-amt")]
    base = res["base_costs"]
    d = np.array([price(*a[:4], base)[0] - price(*t[:4], base)[0] for a, t in zip(amt, thr)])
    m = d[np.random.default_rng(99).integers(0, 20, (20000, 20))].mean(1)
    print(f"Phase 14's costs, logreg rule minus thresholds: mean {d.mean():.1f}, CI {np.percentile(m, 2.5):.0f} to {np.percentile(m, 97.5):.0f} (seed 99)")
    assert glob.glob(str(ROOT / "results" / "eval_cost_stress" / "*.json"))


if __name__ == "__main__":
    main()
