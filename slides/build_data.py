"""Build slides/data.js from the committed result files. Standard library only; no model runs.

    python slides/build_data.py

Slides bind to data.js, so a number on a slide can't drift from its source. Prose-only numbers are in
manual_data.json, each with its source. tests/test_slides.py rebuilds this and compares.

Differences here always mean first minus second, so a plus sign means the first setup costs more. The
JSON's mean_diff is signed from the cheaper setup's side, so `diff` turns it round when needed.
"""

from __future__ import annotations

import json
import math
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
    "audit": "eval_column_audit/20261103-210600.json",
    "tabular": "eval_tabular/20261103-235132.json",
    "timing": "time_logreg/20261112-095810.json",
}
REPLAY_ROWS = 40  # the first rows of the pinned PaySim window


def load(name: str) -> dict:
    return json.loads((RESULTS / SOURCES[name]).read_text(encoding="utf-8"))


def num(x, digits: int = 6):
    return None if x is None else round(float(x), digits)


# Each setup's short name, the same as the console's shortName() in fraud/viewer/console.js (tests/test_slides.py checks).
NAMES = {
    "final-logreg-amt": "Baseline: logreg, rule", "final-signal-amt": "d1 reads trees, rule", "final-signal-cust": "d1 reads trees, customer line",
    "final-rich-amt": "d1, no trees, rule", "final-rich-cust": "d1, no trees, customer line", "ieee-approve-all": "Approve everything",
    "lm-frozen-amt": "d1, frozen, rule", "lm-phase12-amt": "d1, Phase 12 adapter, rule", "lm-seed1-amt": "d1, seed 1, rule", "lm-rich-amt": "d1, no trees, rule",
    "tab-logreg-amt": "logreg, rule", "lm-signal-amt": "d1 reads trees, rule", "signal-alone-amt": "trees at 400 alone, rule", "tab-all-amt": "trees, rule",
    "lm-weighted-amt": "d1, no trees, trained on cost, rule", "ieee-logreg-all-amt": "logreg, rule", "ieee-logreg-all": "logreg, two lines",
    "audit-noCDV-amt": "logreg, no C/D/V, rule", "audit-all-amt": "logreg, all columns, rule", "rule-customers": "rule with a customer line", "rule-base": "logreg, Phase 14's rule",
}


def js_round(x: float) -> int:
    """Math.round, so a sentence here rounds as the console's does (Python's round() goes to even)."""
    return math.floor(x + 0.5)


def sentence(first: str, second: str, mean: float, low: float, high: float, distinguishable: bool, unit: str) -> str:
    """A comparison as one plain sentence, word for word the console's sentence() (fraud/viewer/console.js)."""
    who = lambda s: f"“{NAMES[s]}”"
    bound = lambda v: f"{-js_round(v):,} less" if js_round(v) < 0 else f"{js_round(v):,} more" if js_round(v) > 0 else "0"
    m = js_round(mean)
    head = (f"{who(first)} cost the same per {unit} as {who(second)} to the dollar" if m == 0
            else f"{who(first)} cost {abs(m):,} USD {'less' if m < 0 else 'more'} per {unit} than {who(second)}")
    return (f"{head}, {'and' if distinguishable or m == 0 else 'but'} the 95% interval of the difference runs from {bound(low)} to {bound(high)}, "
            f"so the two are {'' if distinguishable else 'not '}distinguishable.")


def diff(entry: dict, first: str, unit: str) -> dict:
    """A verdict entry as first minus second, whichever side the JSON calls better, with its sentence.
    The first is always the registered a, as on the console: a comparison the other way round is refused."""
    if first != entry["a"]:
        raise ValueError(f"{first} vs {entry['a'] if first == entry['b'] else '?'}: the deck states comparisons in registered order, "
                         f"{entry['a']} minus {entry['b']}")
    sign = 1 if entry["better"] == first else -1
    second = entry["b"]
    low, high = entry["mean_diff_ci95"]
    ci = [low, high] if sign == 1 else [-high, -low]
    wins = entry["cheaper"] if sign == 1 else entry["dearer"]
    mean, dist = sign * entry["mean_diff"], bool(entry["distinguishable"])
    return {"first": first, "second": second, "mean": num(mean), "ci": [num(ci[0]), num(ci[1])],
            "first_cheaper": wins, "windows": entry["cheaper"] + entry["dearer"] + entry["ties"],
            "distinguishable": dist, "said": sentence(first, second, mean, ci[0], ci[1], dist, unit)}


def find(entries: list, a: str, b: str, unit: str) -> dict:
    for e in entries:
        if {e["a"], e["b"]} == {a, b}:
            return diff(e, a, unit)
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


def ms_text(x: float) -> str:
    """The console's ms() (fraud/viewer/console.js)."""
    return f"{x:.0f} ms" if x >= 100 else f"{x:.1f} ms" if x >= 1 else f"{x:.3f} ms" if x >= 0.001 else "<0.001 ms"


def speed_block(d: dict, all_v: list) -> dict:
    """Cost against speed on FINAL, as the console's Cost and speed panel draws it. The model's p90s are from the final test on
    an RTX 3060 Ti; logreg's from fraud/time_logreg.py on a laptop CPU, on dev rows. Ideal = under the registered latency bar and
    not distinguishably dearer than the cheapest timed setup, by the paired comparisons. Approve-all has no time."""
    t, s, bar = load("timing"), d["setups"], d["latency_bar_ms"]
    keys = {"baseline": "final-logreg-amt", "signal_amt": "final-signal-amt", "signal_cust": "final-signal-cust", "rich_amt": "final-rich-amt"}
    pts = [{"key": k, "id": sid, "name": NAMES[sid], "cost": num(s[sid]["mean_cost"]), "cpu": sid == t["setup"],
            "ms": num(t["latency_ms"]["p90"] if sid == t["setup"] else s[sid]["latency_ms"]["p90"])} for k, sid in keys.items()]
    cheapest = min(pts, key=lambda p: p["cost"])
    for p in pts:
        e = next((e for e in all_v if {e["a"], e["b"]} == {p["id"], cheapest["id"]}), None) if p is not cheapest else None
        p["dist_from_cheapest"] = None if p is cheapest else (e["distinguishable"] if e else "never compared")
        p["ideal"] = p["ms"] <= bar and (p is cheapest or (e is not None and not e["distinguishable"]))
    inn, out = [p["cost"] for p in pts if p["ideal"]], [p["cost"] for p in pts if not p["ideal"] and p["ms"] <= bar]
    lr = next(p for p in pts if p["cpu"])
    fastest = min(p["ms"] for p in pts if not p["cpu"])
    times = float(f"{fastest / lr['ms']:.2g}")
    dist = lr["dist_from_cheapest"]
    tail = ("and it is the cheapest setup." if dist is None else "and it was not compared with the cheapest setup." if dist == "never compared"
            else f"and its cost is {'distinguishably dearer than' if dist else 'not distinguishable from'} the cheapest setup's, “{cheapest['name']}”.")
    return {"bar": bar, "points": pts, "cheapest": cheapest["key"], "edge": num((max(inn) + min(out)) / 2) if out else None,
            "times": times, "cpu": t["machine"]["cpu"], "source": f"fraud/results/{SOURCES['timing']}",
            "said": f"“{lr['name']}” decides one payment in {ms_text(lr['ms'])} at p90, about {times:,.0f} times faster than the language model's {ms_text(fastest)}, {tail}"}


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
        "approve_all_accuracy": 1 - d["test_fraud"] / d["test_rows"],
        "fraud_usd": num(s["final-logreg-amt"]["fraud_usd"]),
        "window_rows": {k: v_ for k, v_ in d["windows"].items()},
        "latency": {"p50": num(latency["p50"], 1), "p90": num(latency["p90"], 1), "bar": d["latency_bar_ms"]},
        "setups": {"signal_amt": setup(s["final-signal-amt"], True), "signal_cust": setup(s["final-signal-cust"], True),
                   "baseline": setup(s["final-logreg-amt"], True), "rich_amt": setup(s["final-rich-amt"], True),
                   "approve_all": setup(s["ieee-approve-all"], True), "rich_cust": rich_cust},
        "diffs": {"rich_amt_baseline": find(all_v, "final-rich-amt", "final-logreg-amt", "window"),
                  "signal_amt_baseline": find(all_v, "final-signal-amt", "final-logreg-amt", "window"),
                  "signal_cust_baseline": find(all_v, "final-signal-cust", "final-logreg-amt", "window"),
                  "signal_cust_signal_amt": find(all_v, "final-signal-cust", "final-signal-amt", "window"),
                  "rich_amt_signal_amt": find(all_v, "final-rich-amt", "final-signal-amt", "window"),
                  "signal_amt_approve": find(all_v, "final-signal-amt", "ieee-approve-all", "window"),
                  "rich_cust_baseline": find(all_v, "final-rich-cust", "final-logreg-amt", "window"),
                  "rich_cust_rich_amt": find(all_v, "final-rich-cust", "final-rich-amt", "window"),
                  "rich_cust_approve": find(all_v, "final-rich-cust", "ieee-approve-all", "window"),
                  "rich_amt_approve": find(all_v, "final-rich-amt", "ieee-approve-all", "window")},
        # for the notes: the sign test the JSON records beside each registered question, and the window that pulls the primary's mean back
        "sign": {"signal_amt_baseline": {"p": num(v["primary"]["sign_p"]), "first_cheaper": find(all_v, "final-signal-amt", "final-logreg-amt", "window")["first_cheaper"]},
                 "rich_amt_baseline": {"p": num(v["third"]["sign_p"]), "first_cheaper": find(all_v, "final-rich-amt", "final-logreg-amt", "window")["first_cheaper"]},
                 "final12_gap": num(s["final-signal-amt"]["windows"][list(d["windows"]).index(x["dearest_window"]["window"])] - x["dearest_window"]["baseline"])},
        "prompt_tokens": num(s["final-rich-amt"]["prompt_tokens_mean"], 0),
        "rank_corr": num(x["signal_rank_corr"]),
        "told": {"rich_cust": num(r["told_10_usd"]["mean"]["final-rich-cust"]), "baseline": num(r["told_10_usd"]["mean"]["final-logreg-amt"]),
                 "diff": num(r["told_10_usd"]["against_baseline"]), "ci": [num(c) for c in r["told_10_usd"]["ci"]],
                 # against_baseline is already rich-cust minus the baseline (fraud/dev/final_rich_cust.py)
                 "said": sentence("final-rich-cust", "final-logreg-amt", r["told_10_usd"]["against_baseline"], *r["told_10_usd"]["ci"],
                                  r["told_10_usd"]["ci"][0] > 0 or r["told_10_usd"]["ci"][1] < 0, "window")},
        "dearest": {"window": x["dearest_window"]["window"], "cost": num(x["dearest_window"]["cost"]), "baseline": num(x["dearest_window"]["baseline"])},
        "speed": speed_block(d, all_v),
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
        "diffs": {"frozen_phase12": find(all_v, "lm-frozen-amt", "lm-phase12-amt", "chain"),
                  "rich_phase12": find(all_v, "lm-rich-amt", "lm-phase12-amt", "chain"),
                  "seed1_phase12": find(all_v, "lm-seed1-amt", "lm-phase12-amt", "chain"),
                  "rich_logreg": find(all_v, "lm-rich-amt", "tab-logreg-amt", "chain"),
                  "signal_signal_alone": find(all_v, "lm-signal-amt", "signal-alone-amt", "chain"),
                  "rich_trees": find(all_v, "lm-rich-amt", "tab-all-amt", "chain"),
                  "signal_rich": find(all_v, "lm-signal-amt", "lm-rich-amt", "chain")},
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
        "diffs": {"weighted_rich": diff(v["primary"], "lm-weighted-amt", "chain"), "weighted_logreg": diff(v["second"], "lm-weighted-amt", "chain"),
                  "weighted_trees": diff(v["third"], "lm-weighted-amt", "chain")},
        "waterfall": {"start": num(rich["mean_cost"]), "end": num(weighted["mean_cost"]),
                      "fraud": num(per(weighted, "fraud_let_through") - per(rich, "fraud_let_through")),
                      "declines": num(per(weighted, "good_declined") - per(rich, "good_declined")),
                      "reviews": num(per(weighted, "review_fees") - per(rich, "review_fees"))},
    }


def rule_block() -> dict:
    """Earlier phases, each on its own test windows: never paired with FINAL, and the window count goes on the slide."""
    a, r, c, t = load("amount"), load("rule"), load("audit"), load("tabular")
    s = a["setups"]
    return {
        "phase14": {"lines": num(s["ieee-logreg-all"]["mean_cost"]), "rule": num(s["ieee-logreg-all-amt"]["mean_cost"]), "windows_n": len(a["windows"]),
                    "diff": diff(a["verdict"]["primary"], "ieee-logreg-all-amt", "window"), "latency": {"p50": num(s["ieee-ft-ieee-full"]["latency_ms"]["p50"]), "p90": num(s["ieee-ft-ieee-full"]["latency_ms"]["p90"])}},
        "phase17": {"windows_n": len(c["windows"]), "diff": diff(c["verdict"]["primary"], "audit-noCDV-amt", "window"),
                    "dev": {"mean": num(c["heldout"]["noCDV"]["diff_to_all"]), "ci": [num(v) for v in c["heldout"]["noCDV"]["diff_ci"]], "windows": c["heldout"]["noCDV"]["windows"]}},
        "phase18": {"base": r["setups"]["rule-base"]["good_customers_declined"], "customers": r["setups"]["rule-customers"]["good_customers_declined"],
                    "windows_n": len(r["windows"]), "diff": diff(r["verdict"]["fourth"], "rule-customers", "window")},
        "phase20": {"windows_n": len(t["windows"]), "diff": diff(t["verdict"]["primary"], "tab-all-amt", "window"),
                    "dev": {"mean": num(t["heldout"]["tab-all"]["diff"]), "ci": [num(v) for v in t["heldout"]["tab-all"]["diff_ci"]],
                                    # the dev read's verdict, by the same test as every other: its CI leaves out 0
                                    "distinguishable": t["heldout"]["tab-all"]["diff_ci"][0] > 0 or t["heldout"]["tab-all"]["diff_ci"][1] < 0}},
    }


def paysim_block() -> dict:
    """The committed PaySim trace only: no IEEE-CIS row reaches the deck (the owner's answer, 2026-11-04)."""
    rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines()]
    tx = [r for r in rows if r.get("type") == "tx"]
    model = [r for r in tx if r["by"] == "model"]
    first = model[0]
    tops = [[max(range(3), key=o.__getitem__) for o in r["orders"]] for r in model]
    return {
        "model": rows[0]["backend"]["model"], "window": rows[0]["window"]["id"], "n": len(tx), "model_n": len(model),
        "model_declined": sum(r["action"] == "decline" for r in model), "replay_n": REPLAY_ROWS,
        # option-order bias: model rows whose two orders give different top answers, and those that took the last letter both times
        "order_flips": sum(a != b for a, b in tops), "last_letter_both": sum(a == 2 and b == 0 for a, b in tops),
        "decision": {"type": first["tx"]["type"], "amount": first["tx"]["amount"], "to": first["tx"]["to"], "hour": first["tx"]["hour"],
                     "scores": {k: num(v) for k, v in first["scores"].items()}, "action": first["action"], "truth": first["truth"], "ms": num(first["ms"], 0)},
        "replay": [{"i": r["i"], "type": r["tx"]["type"], "amount": r["tx"]["amount"], "by": r["by"], "action": r["action"]} for r in tx[:REPLAY_ROWS]],
    }


def worked_block() -> dict:
    """The first PaySim TRANSFER the model read in both orders and answered differently by order (reads TRACE at call time)."""
    names = ["approve", "review", "decline"]
    lines = TRACE.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    for line in lines:
        r = json.loads(line)
        if r.get("type") != "tx" or r.get("by") != "model" or r["tx"]["type"] != "TRANSFER":
            continue
        listed, reversed_ = r["orders"]
        top = [max(range(3), key=o.__getitem__) for o in (listed, reversed_)]
        if top[0] == top[1]:
            continue

        def logits(p):
            ls = [math.log(x) for x in p]
            return [x - max(ls) for x in ls]

        scores = r["scores"]
        return {
            "i": r["i"], "type": r["tx"]["type"], "amount": r["tx"]["amount"], "to": r["tx"]["to"], "hour": r["tx"]["hour"],
            "truth": r["truth"], "orders": {"listed": listed, "reversed": reversed_},
            "scores": [scores[n] for n in names],
            "logits": {"listed": logits(listed), "reversed": logits(reversed_)},
            "top": {"listed": names[top[0]], "reversed": names[top[1]], "averaged": max(names, key=scores.get)},
            "action": r["action"],
            # the facts the model read (fraud/brain.py describe) and the question it was asked, for slide 08
            "facts": {"receiver_before": r["signals"]["dest_in_before"], "same_amount_before": r["signals"]["same_amount_step_before"],
                      "round": r["signals"]["round_amount"]},
            "prompt": {"question": head.get("question"), "options": head.get("options")},
            # softmax worked by hand: e to each logit, and their sum
            "exp": {"listed": [math.exp(x) for x in logits(listed)], "sum_listed": sum(math.exp(x) for x in logits(listed))},
        }
    raise LookupError("no two-order TRANSFER that flips in the trace")


def illustration_block(ins: dict, worked: dict, costs: dict) -> dict:
    """The mechanism slides' numbers, derived from the made-up inputs in manual_data.json (illustration), the worked
    PaySim payment and the cost model, so they can't drift apart. The rule prices the payment at the p that
    logistic regression gives it. Features as fraud/baselines.py: night is hours 1-6, round is a whole multiple of 1,000."""
    sig = lambda s: 1 / (1 + math.exp(-s))
    amount, hour = worked["amount"], worked["hour"]
    x = {"transfer": 1.0 if worked["type"] == "TRANSFER" else 0.0, "log_amount": math.log(amount),
         "night": 1.0 if 1 <= hour <= 6 else 0.0, "round": 1.0 if amount % 1000 == 0 else 0.0}
    lr_in = ins["logreg"]
    rows = {k: {"x": x[k], "w": lr_in["w"][k], "adds": lr_in["w"][k] * x[k]} for k in x}
    score = lr_in["start"] + sum(r["adds"] for r in rows.values())
    p = sig(score)
    logreg = {"start": lr_in["start"], "rows": rows, "score": score, "p": p, "odds_times": math.exp(lr_in["w"]["transfer"])}

    def priced(a: float) -> dict:
        return {"amount": a, "approve": p * a, "decline": (1 - p) * max(costs["friction_rate"] * a, costs["friction_min"])}
    rule = {"p": p, "lam": ins["rule"]["lam"], "review": costs["review_fee"] + ins["rule"]["lam"],
            "big": priced(amount), "small": priced(ins["rule"]["small_amount"])}

    t_in = ins["trees"]
    asks = {"t1": x["transfer"] == 1, "t2": amount > t_in["t2"]["over"], "t3": x["night"] == 1}
    trees = {"start": t_in["start"]}
    for t, yes in asks.items():
        trees[t] = t_in[t] | {"took": "yes" if yes else "no"}
    trees["score"] = t_in["start"] + sum(t_in[t][trees[t]["took"]] for t in asks)
    trees["p"] = sig(trees["score"])
    return {"logreg": logreg, "rule": rule, "trees": trees}


def build() -> dict:
    manual = json.loads((HERE / "manual_data.json").read_text(encoding="utf-8"))
    costs = load("final")["costs"]
    worked = worked_block()
    return {
        "_sources": {k: f"fraud/results/{v}" for k, v in SOURCES.items()} | {"paysim_trace": f"fraud/runs/{TRACE.name}", "manual": "slides/manual_data.json"},
        "costs": costs, "final": final_block(), "dev": dev_block(), "cost": cost_block(), "rule": rule_block(), "paysim": paysim_block(), "worked": worked, "manual": manual,
        "illustration": illustration_block(manual["illustration"], worked, costs),
    }


def render() -> str:
    body = json.dumps(build(), indent=1, ensure_ascii=False, sort_keys=True)
    return f"// Generated by slides/build_data.py from committed results. Don't edit by hand.\nwindow.DATA = {body};\n"


if __name__ == "__main__":
    (HERE / "data.js").write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {HERE / 'data.js'}")
