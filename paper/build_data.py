"""The paper's numbers: the deck's build, formatted as the deck's slides/js/core.js `formats` format them.

paper/main.typ reads paper/data.json through v("<path>|<fmt>"). The path is into slides/build_data.py's build()
output; no "|fmt" means String(value), as the deck's bindings do. Run `python paper/build_data.py` to rewrite it."""

import importlib.util
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINUS = "\u2212"


def deck() -> dict:
    """The deck's build, loaded by path: slides/ is a folder of files, not a package."""
    spec = importlib.util.spec_from_file_location("deck_build_data", ROOT / "slides" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(), module.js_round


_DECK, js_round = None, None


def _load():
    global _DECK, js_round
    if _DECK is None:
        _DECK, js_round = deck()
        _DECK = {**_DECK, "rev": revision()}
    return _DECK


# ---- the revision's results (REVISION_LOG.md, Stages 0-3): read from their committed JSON, each number with its label ----

R = ROOT / "fraud" / "results"
REV_SOURCES = {
    "audit": "window_audit.json", "readout": "eval_gbdt_readout/20261119-231754.json", "readouts": "eval_readouts/20261120-075713.json",
    "parity": "eval_parity/20261119-232618.json", "noc": "eval_noc/20261120-074245.json", "order": "eval_order_bias/20261120-045604.json",
    "custline": "eval_customer_line/20261119-232409.json", "lm_nuc": "eval_lm_nuc/20261104-090217.json", "prompt": "prompt_example.json",
    "amount": "eval_ieee_amount/20261103-164830.json", "lfm": "eval_lfm/20261030-122539.json", "lam118": "eval_lr_lambda118/20261121-202021.json",
}
SETUP_KEY = {"final-logreg-amt": "lr", "final-signal-amt": "lmtree", "final-signal-cust": "lmtree_cust", "final-rich-amt": "lm", "final-tab-all-amt": "gbdt", "ieee-approve-all": "approve_all"}
SETUP_NAME = {"lr": "LR", "lmtree": "LM + tree", "lmtree_cust": "LM + tree, customer line", "lm": "LM alone", "gbdt": "GBDT alone"}


def _j(name: str) -> dict:
    return json.loads((R / REV_SOURCES[name]).read_text(encoding="utf-8"))


def _diff(mean: float, ci: list, label: str) -> dict:
    lo, hi = ci
    return {"mean": mean, "ci": [lo, hi], "distinguishable": lo > 0 or hi < 0, "label": label}


def _pct1(x: float) -> str:
    return to_fixed(100 * x, 1)


def revision() -> dict:
    out: dict = {"_sources": {k: f"fraud/results/{v}" for k, v in REV_SOURCES.items()}}
    a = _j("audit")
    out["audit"] = {"test_rows": a["test"]["rows"], "windows_scored": a["test"]["windows_scored"], "untouched": a["test"]["untouched_windows"], "never_scored_rows": a["test"]["rows_never_scored"]}

    ro = _j("readouts")
    cmp = ro["comparisons"]
    out["cmp"] = {k: {**_diff(c["mean_diff_a_minus_b"], c["registered_percentile"], c["label"]), "block2": c["block2"], "block4": c["block4"], "bca": c["bca"],
                      "sd": c["paired_sd"], "mde": c["mde"]} for k, c in cmp.items()}
    out["mean"] = {SETUP_KEY[s]: v for s, v in ro["mean_cost"].items()}
    out["ranking"] = {SETUP_KEY[s]: x for s, x in ro["ranking"].items()}
    d = ro["diagnostic"]
    out["by_action"] = {SETUP_KEY[s]: d[s]["by_action_per_window"] for s in d if s in SETUP_KEY}
    out["drop_amount"] = {k: _diff(x["mean_diff"], x["ci"], x["label"]) | {"amounts": x["removed_amounts"]} for k, x in d["lm_alone_minus_lr_without_largest_frauds"].items()}
    out["drop_gap"] = {k: _diff(x["mean_diff"], x["ci"], x["label"]) | {"amounts": [e["amount"] for e in x["removed"]]} for k, x in d["lm_alone_minus_lr_without_largest_gap_frauds"].items()}
    out["reliability"] = {SETUP_KEY[s]: d[s]["reliability"] for s in d if s in SETUP_KEY}
    cells = ro["price_grid"]["cells"]
    out["grid"] = {"cells": [{"r": c["r"], "c_r": c["c_r"], "customer": c["customer"], "cheapest": SETUP_KEY[c["cheapest"]], "cheapest_name": SETUP_NAME[SETUP_KEY[c["cheapest"]]],
                              "vs_lr": c["cheapest_vs_lr"]["mean_diff"] if c["cheapest_vs_lr"] else 0.0, "dist": "yes" if c["distinguishable_from_lr"] else "no"} for c in cells],
                   "n": len(cells), "gbdt_cheapest": sum(c["cheapest"] == "final-tab-all-amt" for c in cells),
                   "lmtree_cheapest": sum(c["cheapest"] == "final-signal-amt" for c in cells), "dist_n": sum(c["distinguishable_from_lr"] for c in cells)}
    disp = {}
    for dim, groups in ro["disparity"].items():
        disp[dim] = {}
        for g0, per in groups.items():
            g = g0.replace(".", "_").replace(" ", "_")
            disp[dim][g] = {"name": g0, "n_good": per["final-logreg-amt"]["good_declined"]["n"], "n_fraud": per["final-logreg-amt"]["fraud_missed"]["n"]}
            for s, x in per.items():
                gd, fm = x["good_declined"], x["fraud_missed"]
                disp[dim][g][SETUP_KEY[s]] = {"gd": _pct1(gd["rate"]), "gd_ci": f"{_pct1(gd['ci'][0])}–{_pct1(gd['ci'][1])}",
                                             "fm": _pct1(fm["rate"]) if fm["rate"] is not None else "n/a", "fm_ci": f"{_pct1(fm['ci'][0])}–{_pct1(fm['ci'][1])}" if fm["ci"] else "n/a"}
    out["disparity"] = disp
    seed = ro["seed_noise"]["comparison"]
    out["seed"] = {"mean": seed["mean_diff"], "ci": seed["mean_diff_ci95"], "floor": ro["seed_noise"]["noise_floor_usd_per_window"], "label": "dev"}

    rd = _j("readout")
    gb = rd["setups"]["final-tab-all-amt"]
    out["gbdt"] = {"mean": gb["mean_cost"], "ci": gb["cost_ci95"], "auc": gb["auc"], "good_declined": gb["good_customers_declined"], "frauds_stopped": gb["frauds_stopped"],
                   "frauds": gb["frauds"], "label": rd["label"], "mde": {k: v["mde"] for k, v in rd["mde_realised"]["pairs"].items()}}

    par = _j("parity")
    out["parity"] = {"setups": {k: {"mean": v["mean_cost"], "auc": v["auc"]} for k, v in par["setups"].items()},
                     "cmp": {f"{c['a']}__{c['b']}": _diff(c["mean_diff"], c["mean_diff_ci95"], c["label"]) for c in par["comparisons"]}}
    noc = _j("noc")
    out["noc"] = {"setups": {k: {"mean": v["mean_cost"], "auc": v["auc"]} for k, v in noc["setups"].items()},
                  "cmp": {f"{c['a']}__{c['b']}": _diff(c["mean_diff"], c["mean_diff_ci95"], c["label"]) for c in noc["comparisons"]},
                  "changed": noc["ranking"]["changed"]}
    od = _j("order")
    out["order"] = {s: {"flip_pct": _pct1(x["flip_rate"]), "approve_top_pct": _round(100 * x["top_answer_share"]["rev"]["approve"]), "rank_corr": x["decline_score_rank_corr"],
                        "rev": x["orders"]["rev"]["mean_cost"], "fwd": x["orders"]["fwd"]["mean_cost"],
                        "fwd_rev": _diff(x["fwd_minus_rev"]["mean_diff"], x["fwd_minus_rev"]["mean_diff_ci95"], "dev"),
                        "avg_rev": _diff(x["avg_minus_rev"]["mean_diff"], x["avg_minus_rev"]["mean_diff_ci95"], "dev") if x.get("avg_minus_rev") else None}
                    for s, x in od["setups"].items()}
    cl = _j("custline")
    out["custline"] = {s: {"lambda": x["lambda"], "points": [{"price": c["price"], "cost": c["mean_cost"], "stopped": c["frauds_stopped"], "declined": c["good_customers_declined"],
                                                               "vs0": _diff(c["vs_price_0"]["mean_diff"], c["vs_price_0"]["mean_diff_ci95"], "dev") if c["vs_price_0"] else None}
                                                              for c in x["curve"].values()], "frauds": next(iter(x["curve"].values()))["frauds"]}
                       for s, x in cl["setups"].items()}
    manual = json.loads((ROOT / "paper" / "revision_manual.json").read_text(encoding="utf-8"))
    out["protocol"] = manual["protocol"]
    out["claims"] = manual["claims"]

    # Which test windows were used when (the window audit): every earlier evaluation before FINAL, contiguous or chained.
    src = _j("audit")
    earlier = [s for s in src["sources"] if s["name"] != "final"]
    out["windows"] = {"earlier": sum(s["windows_n"] for s in earlier), "earlier_contiguous": sum(s["windows_n"] for s in earlier if all(w["contiguous"] for w in s["windows"])),
                      "earlier_chains": sum(s["windows_n"] for s in earlier if not all(w["contiguous"] for w in s["windows"])),
                      "final": next(s["windows_n"] for s in src["sources"] if s["name"] == "final"), "rows_scored_twice": src["test"]["rows_scored_twice"]}

    # The earlier test windows of the rule's result: the rule against two fixed cut-offs, for LR and for the language model then in use.
    am = _j("amount")
    v = am["verdict"]
    rep = {(r["a"], r["b"]): r for r in v["reported"]}
    lr_cut30 = rep[("ieee-logreg-all-dev30", "ieee-logreg-all")]
    out["earlier"] = {"lm_rule_vs_cutoffs": _diff(v["second"]["mean_diff"], v["second"]["mean_diff_ci95"], "earlier test windows"),
                      "lr_cutoffs30_vs_cutoffs": _diff(lr_cut30["mean_diff"], lr_cut30["mean_diff_ci95"], "earlier test windows"),
                      "dev_windows": len(am["dev_windows"])}

    # The frozen check that chose d1-Omni-2B (PaySim dev rows).
    lf = _j("lfm")["dev"]
    out["frozen"] = {"d1_auc": lf["d1-omni-2b"]["auc"][lf["d1-omni-2b"]["variant"]], "qwen_auc": lf["frozen_1.5b"]["auc"]["both"], "logreg_auc": lf["logreg"]["auc"],
                     "rows": lf["d1-omni-2b"]["rows"], "d1_p90_ms": lf["d1-omni-2b"]["latency_single_dev0"]["p90"], "gate_pass": lf["d1-omni-2b"]["gate"]["pass"]}

    # LR's lambda on 30 dev windows (the baseline) and on all 118 (Appendices E-F); the tree score alone against LM + tree (dev).
    nuc = _j("lm_nuc")
    out["lam"] = {"lr_30": nuc["fits"]["tab-logreg"]["lambda_on_30"], "lr_118": par["setups"]["tab-logreg"]["lambda"]}
    # LR with lambda on all 118 dev windows, applied to FINAL (read-out, fraud/prereg/final_lr_lambda118_readout.md).
    l118 = _j("lam118")
    key = {"final-signal-amt": "lmtree", "final-signal-cust": "lmtree_cust", "final-rich-amt": "lm", "final-tab-all-amt": "gbdt"}
    out["lam118"] = {"mean": l118["mean_cost"]["final-logreg-amt-lam118"],
                     "primary": _diff(l118["primary"]["mean_diff_a_minus_b"], l118["primary"]["ci95"], "read-out"),
                     "vs": {key[s]: _diff(x["mean_diff_a_minus_b"], x["ci95"], "read-out") for s, x in l118["against_lr118"].items()}}
    out["signal_alone"] = {"mean": nuc["setups"]["signal-alone-amt"]["mean_cost"], "lm_signal": nuc["setups"]["lm-signal-amt"]["mean_cost"],
                           "distinguishable": nuc["verdict"]["against_signal"][0]["distinguishable"]}
    pe = _j("prompt")
    out["prompt"] = {k: pe[k] for k in ("context", "question", "state_plain", "state_rich", "signal_line", "full_rich", "is_fraud")}
    out["prompt"]["options"] = pe["options"]
    return out


def _round(x: float) -> int:
    _load()
    return js_round(x)


def to_fixed(x: float, digits: int) -> str:
    """JS Number.toFixed: the exact binary value, ties away from zero (Python's :.Nf rounds ties to even)."""
    return f"{Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP):.{digits}f}"


def group(x: float, digits: int = 0) -> str:
    """toLocaleString('en-US', fixed digits): Intl rounds half-expand on the shortest decimal, i.e. repr(x)."""
    d = Decimal(repr(float(abs(x)))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{d:,.{digits}f}"


def _int(x): return (MINUS if _round(x) < 0 else "") + group(_round(x))
def _signed(x): return ("+" if _round(x) > 0 else MINUS if _round(x) < 0 else "") + group(_round(x))
def _auc(x, d=3): return re.sub(r"^0", "", to_fixed(x, d))


def _str(x) -> str:
    """JS String(x) for what the data holds: numbers without a trailing .0, booleans lower-case."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


FORMATS: dict[str, Callable] = {
    "int": _int,
    "signed": _signed,
    "ci": lambda a: f"{_int(a[0])} to {_int(a[1])}",
    "range": lambda a: f"{_int(a[0])}\u2013{_int(a[1])}",
    "auc": _auc,
    "auc4": lambda x: _auc(x, 4),
    "aucrange": lambda a: f"{_auc(a[0])}\u2013{_auc(a[1])}",
    "num": _str,
    "abs": lambda x: group(_round(x)),
    "pct": lambda x: f"{_round(x * 100)}%",
    "d2": lambda x: (MINUS if to_fixed(x, 2).startswith("-") else "") + to_fixed(abs(x), 2),
    "usd2": lambda x: group(x, 2),
    "dist": lambda d: "distinguishable" if d["distinguishable"] else "not distinguishable",
}

KEY = re.compile(r'\bv\(\s*"([^"]+)"\s*\)')


def keys(typ: str) -> list[str]:
    """Every distinct v("path|fmt") key in the Typst source, in order of first use."""
    return list(dict.fromkeys(KEY.findall(typ)))


def resolve(data: dict, key: str) -> str:
    path, _, fmt = key.partition("|")
    node = data
    try:
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (KeyError, IndexError, ValueError, TypeError):
        raise KeyError(key) from None
    if node is None:
        raise KeyError(f"{key} (value is None)")
    if fmt and fmt not in FORMATS:
        raise KeyError(f"{key} (no format {fmt})")
    return FORMATS[fmt](node) if fmt else _str(node)


def render() -> str:
    typ = (ROOT / "paper" / "main.typ").read_text(encoding="utf-8")
    data = _load()
    out = {k: resolve(data, k) for k in sorted(keys(typ))}
    return json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    (ROOT / "paper" / "data.json").write_text(render(), encoding="utf-8", newline="\n")
