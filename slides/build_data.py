"""Build slides/data.js from the committed result files. Standard library only; no model runs.

    python slides/build_data.py

Slides bind to data.js, so a number on a slide can't drift from its source. Prose-only numbers are in
manual_data.json, each with its source. tests/test_slides.py rebuilds this and compares.

Differences here always mean first minus second, so a plus sign means the first setup costs more. The
JSON's mean_diff is signed from the cheaper setup's side, so `diff` turns it round when needed.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "fraud" / "results"
TRACE = HERE.parent / "fraud" / "runs" / "20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl"
SOURCES = {
    "final": "eval_final/20261104-090649.json",
    "readout": "eval_final/rich_cust_readout.json",
    "extras": "eval_final/extras.json",
    "nuc": "eval_lm_nuc/20261104-090217.json",
    "cost": "eval_lm_cost/20261104-140349.json",
    "amount": "eval_ieee_amount/20261103-164830.json",
    "rule": "eval_rule/20261103-211736.json",
}
REPLAY_ROWS = 40  # the first rows of the pinned PaySim window


def load(name: str) -> dict:
    return json.loads((RESULTS / SOURCES[name]).read_text(encoding="utf-8"))


def num(x, digits: int = 6):
    return None if x is None else round(float(x), digits)


def diff(entry: dict, first: str) -> dict:
    """A verdict entry as first minus second, whichever side the JSON calls better."""
    sign = 1 if entry["better"] == first else -1
    low, high = entry["mean_diff_ci95"]
    ci = [low, high] if sign == 1 else [-high, -low]
    wins = entry["cheaper"] if sign == 1 else entry["dearer"]
    return {"mean": num(sign * entry["mean_diff"]), "ci": [num(ci[0]), num(ci[1])],
            "first_cheaper": wins, "windows": entry["cheaper"] + entry["dearer"] + entry["ties"],
            "distinguishable": bool(entry["distinguishable"])}


def find(entries: list, a: str, b: str) -> dict:
    for e in entries:
        if {e["a"], e["b"]} == {a, b}:
            return diff(e, a)
    raise KeyError((a, b))


def setup(s: dict, windows: bool = False) -> dict:
    out = {"mean": num(s["mean_cost"]), "ci": [num(v) for v in s["cost_ci95"]], "auc": num(s["auc"]),
           "auc_ci": [num(v) for v in s["auc_ci95"]] if s.get("auc_ci95") else None,
           "recall": num(s["recall"]), "precision": num(s["precision"]), "frauds_stopped": s["frauds_stopped"],
           "fraud_usd_stopped": num(s["fraud_usd_stopped"]), "fraud_let_through": num(s["fraud_let_through"]),
           "good_declined": s["good_customers_declined"], "good_declined_usd": num(s["good_declined"]),
           "good_reviewed": s["good_reviewed"], "good_touched": s["good_customers_declined"] + s["good_reviewed"],
           "reviews": s["reviews"], "review_fees": num(s["review_fees"])}
    if windows:
        out["windows"] = [num(v) for v in s["windows"]]
    return out


def final_block() -> dict:
    d, r, x = load("final"), load("readout"), load("extras")
    s = d["setups"]
    v = d["verdict"]
    all_v = [v["primary"], v["second"], v["third"], *v["reported"], *r["against"]]
    t = r["table"]["final-rich-cust"]
    rich_cust = {"mean": num(t["mean_cost"]), "ci": [num(c) for c in t["cost_ci95"]], "auc": num(s["final-rich-amt"]["auc"]),
                 "frauds_stopped": t["frauds_stopped"], "fraud_usd_stopped": num(t["fraud_usd_stopped"]),
                 "fraud_let_through": num(t["fraud_usd_let_through"]), "good_declined": t["good_customers_declined"],
                 "good_touched": t["good_declined_or_reviewed"], "reviews": t["reviews"], "windows": [num(w) for w in r["windows"]]}
    latency = s["final-rich-amt"]["latency_ms"]
    return {
        "eval": d["eval"], "windows_n": len(d["windows"]), "rows": d["test_rows"], "frauds": d["test_fraud"],
        "fraud_usd": num(s["final-logreg-amt"]["fraud_usd"]),
        "window_rows": {k: v_ for k, v_ in d["windows"].items()},
        "latency": {"p50": num(latency["p50"], 1), "p90": num(latency["p90"], 1), "bar": d["latency_bar_ms"]},
        "setups": {"signal_amt": setup(s["final-signal-amt"], True), "signal_cust": setup(s["final-signal-cust"], True),
                   "baseline": setup(s["final-logreg-amt"], True), "rich_amt": setup(s["final-rich-amt"], True),
                   "approve_all": setup(s["ieee-approve-all"], True), "rich_cust": rich_cust},
        "diffs": {"rich_amt_baseline": find(all_v, "final-rich-amt", "final-logreg-amt"),
                  "signal_amt_baseline": find(all_v, "final-signal-amt", "final-logreg-amt"),
                  "signal_cust_baseline": find(all_v, "final-signal-cust", "final-logreg-amt"),
                  "signal_amt_rich_amt": find(all_v, "final-signal-amt", "final-rich-amt"),
                  "signal_amt_approve": find(all_v, "final-signal-amt", "ieee-approve-all"),
                  "rich_cust_baseline": find(all_v, "final-rich-cust", "final-logreg-amt"),
                  "rich_cust_rich_amt": find(all_v, "final-rich-cust", "final-rich-amt"),
                  "rich_cust_approve": find(all_v, "final-rich-cust", "ieee-approve-all"),
                  "rich_amt_approve": find(all_v, "final-rich-amt", "ieee-approve-all")},
        "rank_corr": num(x["signal_rank_corr"]),
        "told": {"rich_cust": num(r["told_10_usd"]["mean"]["final-rich-cust"]), "baseline": num(r["told_10_usd"]["mean"]["final-logreg-amt"]),
                 "diff": num(r["told_10_usd"]["against_baseline"]), "ci": [num(c) for c in r["told_10_usd"]["ci"]]},
        "dearest": {"window": x["dearest_window"]["window"], "cost": num(x["dearest_window"]["cost"]), "baseline": num(x["dearest_window"]["baseline"])},
    }


def dev_block() -> dict:
    n = load("nuc")
    s = n["setups"]
    names = {"frozen": "lm-frozen-amt", "phase12": "lm-phase12-amt", "seed1": "lm-seed1-amt", "rich": "lm-rich-amt",
             "logreg": "tab-logreg-amt", "signal": "lm-signal-amt", "signal_alone": "signal-alone-amt", "trees": "tab-all-amt"}
    v = n["verdict"]
    all_v = [*v["against_trees"], *v["against_logreg"], *v["against_signal"], *v["among_lm"], v["trees_vs_logreg"]]
    return {
        "eval": n["eval"], "windows_n": len(n["windows"]), "rows": n["rows"], "frauds": n["frauds"],
        "setups": {k: setup(s[name]) for k, name in names.items()},
        "diffs": {"phase12_frozen": find(all_v, "lm-phase12-amt", "lm-frozen-amt"),
                  "rich_phase12": find(all_v, "lm-rich-amt", "lm-phase12-amt"),
                  "seed1_phase12": find(all_v, "lm-seed1-amt", "lm-phase12-amt"),
                  "rich_logreg": find(all_v, "lm-rich-amt", "tab-logreg-amt"),
                  "signal_signal_alone": find(all_v, "lm-signal-amt", "signal-alone-amt"),
                  "rich_trees": find(all_v, "lm-rich-amt", "tab-all-amt"),
                  "signal_rich": find(all_v, "lm-signal-amt", "lm-rich-amt")},
        "echo": {"rank_corr": num(n["echo"]["rank_corr"]), "differ": n["echo"]["decisions_differ"], "rows": n["echo"]["rows"]},
    }


def cost_block() -> dict:
    c = load("cost")
    s, v = c["setups"], c["verdict"]
    rich, weighted = s["lm-rich-amt"], s["lm-weighted-amt"]
    n = len(c["windows"])
    per = lambda a, key: a[key] / n
    return {
        "eval": c["eval"], "windows_n": n, "frauds": rich["frauds"], "big_frauds": rich["big_frauds"], "lambda": c["fits"]["lm-rich"]["lambda"],
        "setups": {"rich": {**setup(rich), "big_stopped": rich["big_frauds_stopped"]}, "weighted": {**setup(weighted), "big_stopped": weighted["big_frauds_stopped"]},
                   "rich_cust": setup(s["lm-rich-cust"]), "weighted_cust": setup(s["lm-weighted-cust"])},
        "diffs": {"weighted_rich": diff(v["primary"], "lm-weighted-amt"), "weighted_logreg": diff(v["second"], "lm-weighted-amt"),
                  "weighted_trees": diff(v["third"], "lm-weighted-amt")},
        "waterfall": {"start": num(rich["mean_cost"]), "end": num(weighted["mean_cost"]),
                      "fraud": num(per(weighted, "fraud_let_through") - per(rich, "fraud_let_through")),
                      "declines": num(per(weighted, "good_declined") - per(rich, "good_declined")),
                      "reviews": num(per(weighted, "review_fees") - per(rich, "review_fees"))},
    }


def rule_block() -> dict:
    a, r = load("amount"), load("rule")
    s = a["setups"]
    return {
        "phase14": {"lines": num(s["ieee-logreg-all"]["mean_cost"]), "rule": num(s["ieee-logreg-all-amt"]["mean_cost"]),
                    "diff": diff(a["verdict"]["primary"], "ieee-logreg-all-amt"), "latency": {"p50": num(s["ieee-ft-ieee-full"]["latency_ms"]["p50"]), "p90": num(s["ieee-ft-ieee-full"]["latency_ms"]["p90"])}},
        "phase18": {"base": r["setups"]["rule-base"]["good_customers_declined"], "customers": r["setups"]["rule-customers"]["good_customers_declined"],
                    "diff": diff(r["verdict"]["fourth"], "rule-customers")},
    }


def paysim_block() -> dict:
    """The committed PaySim trace only: no IEEE-CIS row reaches the deck (the owner's answer, 2026-11-04)."""
    rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines()]
    tx = [r for r in rows if r.get("type") == "tx"]
    model = [r for r in tx if r["by"] == "model"]
    first = model[0]
    return {
        "model": rows[0]["backend"]["model"], "window": rows[0]["window"]["id"], "n": len(tx), "model_n": len(model),
        "model_declined": sum(r["action"] == "decline" for r in model), "replay_n": REPLAY_ROWS,
        "decision": {"type": first["tx"]["type"], "amount": first["tx"]["amount"], "to": first["tx"]["to"], "hour": first["tx"]["hour"],
                     "scores": {k: num(v) for k, v in first["scores"].items()}, "action": first["action"], "truth": first["truth"], "ms": num(first["ms"], 0)},
        "replay": [{"i": r["i"], "type": r["tx"]["type"], "amount": r["tx"]["amount"], "by": r["by"], "action": r["action"]} for r in tx[:REPLAY_ROWS]],
    }


def build() -> dict:
    manual = json.loads((HERE / "manual_data.json").read_text(encoding="utf-8"))
    costs = load("final")["costs"]
    return {
        "_sources": {k: f"fraud/results/{v}" for k, v in SOURCES.items()} | {"paysim_trace": f"fraud/runs/{TRACE.name}", "manual": "slides/manual_data.json"},
        "costs": costs, "final": final_block(), "dev": dev_block(), "cost": cost_block(), "rule": rule_block(), "paysim": paysim_block(), "manual": manual,
    }


def render() -> str:
    body = json.dumps(build(), indent=1, ensure_ascii=False, sort_keys=True)
    return f"// Generated by slides/build_data.py from committed results. Don't edit by hand.\nwindow.DATA = {body};\n"


if __name__ == "__main__":
    (HERE / "data.js").write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {HERE / 'data.js'}")
