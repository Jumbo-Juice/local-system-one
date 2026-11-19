"""Feature parity: LR and boosted trees on exactly what the rich prompt shows, on dev only → fraud/results/eval_parity/ (revision Stage 2.2).

    python -m fraud.history              # the history cache, if it isn't there yet
    python -m fraud.eval_parity          # fit on fit rows, score dev, compare; the trees take most of the time (CPU)

Models: fraud/models/parity_<name>.pkl (git-ignored). Traces: fraud/runs/<eval id>_eval_<setup>_hdev<k>.jsonl. Results: fraud/results/eval_parity/<eval id>.json + .md.

PRE-REGISTRATION (revision Stage 2.2; committed before any model here was fitted).
- The question: did the language model lose to LR and trees on modelling, or on information? LR and trees on all 583 columns saw
  far more than the prompt, and no history (REVISION_LOG.md, Stage 0). Here they see exactly the prompt's fields.
- Features (features_parity below), the fields describe_rich() writes into the LM-alone prompt, nothing else:
  log amount; hour (24 one-hots); one-hots of ProductCD, card4, card6, P_emaildomain, R_emaildomain, DeviceType, M4, M6;
  signed log and a missing flag of C1, C4, C8, C13, D1, C2, C5, C6, C9, C11, C14, D2, D3, D4, D10, D15, dist1; and the card and
  address history the prompt states: card_log_before, card_log_since, card_log_1d, card_log_7d, card_amt_vs_mean,
  addr_log_distinct_card (fraud/history.py, already log scale, no gaps). Not included: the hand-written context block (a
  human summary of fit-row patterns, which a fitted model learns for itself) and the tree score (that is LM + tree's extra line).
- Setups: par-logreg (fraud/baselines.LogReg) and par-trees (fraud/gbdt.py at Phase 20's settings: 800 trees, depth 6,
  learning rate 0.05, row sample 0.5, column sample 0.3, seed 0, float32), both fitted on fit rows only.
- Policy: p fitted on dev30, lambda on all 118 dev windows, for both, the same procedure as the LM setups they are compared
  with (so this comparison has no lambda asymmetry). Phase 14's rule. Read on the 88 held-out chains.
- Comparisons, paired window-bootstrap 95% percentile interval (10,000 resamples, seed 20261102), distinguishable only if it
  excludes zero: LM alone (lm-rich) vs par-logreg and vs par-trees; LM + tree (lm-signal) vs par-trees; and, for how much the
  prompt leaves out, par-logreg vs tab-logreg and par-trees vs tab-all (the all-column versions). AUC for each.
- Reading the answer (Implementation choice): if LM alone is within the interval of par-logreg or par-trees, the gap to the
  all-column baselines was information; if it stays distinguishably dearer than par-trees, it was (at least partly) modelling.
- Label: dev. Every held-out read with lambda on 118 windows is flattered, as the paper already says.
- Stated before running (Inferred): par-trees is cheaper than LM alone, but by less than tab-all is; par-logreg about ties LM
  alone; both parity models are dearer than their all-column versions.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_ieee as ev, eval_lm_nuc as nuc, eval_tabular as et, history, ieee_brain, ieee_data, tabular as tb
from .baselines import LogReg
from .gbdt import GBDT
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_parity"
LABEL = "dev"
TEXT = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4", "M6")
NUMERIC = (*ieee_brain.COUNTERS, "D1", *ieee_brain.MORE_COUNTERS, *ieee_brain.DAYS, "dist1")
HISTORY = ("card_log_before", "card_log_since", "card_log_1d", "card_log_7d", "card_amt_vs_mean", "addr_log_distinct_card")


class Parity:
    """features(t, idx): the rich prompt's fields as numbers."""

    def __init__(self, hist: dict, numeric: tuple[str, ...] = NUMERIC):
        self.h, self.numeric = hist["history"], numeric

    def __call__(self, t, idx: np.ndarray) -> np.ndarray:
        num = np.stack([t.col(k)[idx] for k in self.numeric], 1).astype(np.float64)
        return np.c_[np.log1p(t.amount[idx]), ev._hours(t, idx), *[ev._one_hot(t, k, idx) for k in TEXT], ev._signed_log(num), np.isnan(num),
                     np.stack([self.h[k][idx] for k in HISTORY], 1)]


def fit(name: str, f, t, log=print):
    a = ieee_data.split_bounds(len(t))[0]
    idx = np.arange(a)
    path = et.MODELS / f"parity_{name}.pkl"
    t0 = time.perf_counter()
    if name.endswith("logreg"):
        m = LogReg().fit(f(t, idx), t.is_fraud[:a].astype(float))
    elif path.exists():
        m = pickle.loads(path.read_bytes())
    else:
        m = GBDT(et.PARAMS).fit(f(t, idx).astype(np.float32), t.is_fraud[:a].astype(np.float64))
        path.write_bytes(pickle.dumps(m))
    log(f"{name} fitted in {time.perf_counter() - t0:.0f} s")
    return m


def compare(names: dict, by: dict, pairs) -> list[dict]:
    return [{**ev.paired(by[a], by[b]), "a": a, "b": b, "label": LABEL} for a, b in pairs]


def run(setups: dict, pairs, title: str, script: str, results: Path, log=print) -> dict:
    """Fit what isn't saved, refit each policy on dev, write held-out traces, compare. setups: name → (kind, q or features)."""
    t = ieee_data.load()
    n = len(t)
    dev, held = tb.dev30(n), tb.heldout_dev(n)
    rows = np.concatenate([w.rows for w in held])
    y = t.is_fraud[rows].astype(int)
    eval_id, by, rows_out = stamp(), {}, {}
    for name, (kind, x) in setups.items():
        q = et.score(fit(name, x, t, log), x, t, dev + held) if kind == "fit" else np.clip(x, et.TREE_SCORE_FLOOR, 1 - et.TREE_SCORE_FLOOR)
        pol = tb.fit_policy(q, t, dev)
        lam30 = pol.lam
        pol.lam = et.lambda_on(pol, q, t, dev + held)
        tb.write_setup(eval_id, t, held, name, q, pol, "rule", "lm" if name.startswith("lm-") else "logreg")
        tr = tb.read_setup(eval_id, name, held)
        by[name] = tb.window_costs(tr)
        rows_out[name] = {"mean_cost": float(by[name].mean()), "auc": ev.auc(q[rows], y), "lambda": pol.lam, "lambda_on_30": lam30, **tb.customers(tr), "label": LABEL}
        log(f"{name}: {rows_out[name]['mean_cost']:,.0f} USD/window, AUC {rows_out[name]['auc']:.4f}, lambda {pol.lam}")
    out = {"eval": eval_id, "script": script, "label": LABEL, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": ev.COSTS.as_dict(), "windows": len(held),
           "lambda": "p on dev30, lambda on all 118 dev windows, for every setup here", "setups": rows_out, "comparisons": compare(setups, by, pairs)}
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = [f"# {title}, dev {eval_id}", "", "88 held-out dev chains. Label: dev. p on dev30, lambda on all 118 dev windows for every setup. Model scores are uncalibrated.", "",
         "| setup | mean cost | AUC | good declined | frauds stopped |", "|---|---:|---:|---:|---:|"]
    L += [f"| {k} | {v['mean_cost']:,.0f} | {v['auc']:.4f} | {v['good_customers_declined']} | {v['frauds_stopped']} of {v['frauds']} |" for k, v in sorted(rows_out.items(), key=lambda kv: kv[1]["mean_cost"])]
    L += ["", "| a − b | mean | 95% CI | distinguishable |", "|---|---:|---|---|"]
    L += [f"| {c['a']} − {c['b']} | {c['mean_diff']:+,.0f} | {c['mean_diff_ci95'][0]:,.0f} to {c['mean_diff_ci95'][1]:,.0f} | {'yes' if c['mean_diff_ci95'][0] > 0 or c['mean_diff_ci95'][1] < 0 else 'no'} |" for c in out["comparisons"]]
    (results / f"{eval_id}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return out


def references(t, log=print) -> dict:
    n = len(t)
    ref = nuc.references(t, log)
    return {"lm-rich": ("score", nuc.saved("lm-rich", n)[0]), "lm-signal": ("score", nuc.saved("lm-signal", n)[0]),
            "tab-logreg": ("score", ref["tab-logreg"]), "tab-all": ("score", ref["tab-all"])}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = lambda m: print(m, flush=True)
    t = ieee_data.load()
    f = Parity(history.load())
    setups = {"par-logreg": ("fit", f), "par-trees": ("fit", f), **references(t, log)}
    pairs = (("lm-rich", "par-logreg"), ("lm-rich", "par-trees"), ("lm-signal", "par-trees"), ("par-logreg", "tab-logreg"), ("par-trees", "tab-all"))
    run(setups, pairs, "Feature parity", "fraud/eval_parity.py", RESULTS, log)


if __name__ == "__main__":
    main()
