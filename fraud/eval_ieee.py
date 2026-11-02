"""One random frozen and one random PaySim fine-tuned model on IEEE-CIS → fraud/results/eval_ieee/.

    python -m fraud.eval_ieee --draw                  # the seeded draw and the windows; no model, no labels
    python -m fraud.eval_ieee --smoke 30              # latency and prompt size on fit rows; no scores shown
    python -m fraud.eval_ieee --ieee-adapter <dir>    # the eval: dev, thresholds, test windows, report
    python -m fraud.eval_ieee --ieee-adapter <dir> --resume <eval id>  # finish an interrupted eval (finished windows are kept)
    python -m fraud.eval_ieee --reprice <eval id>     # re-decide its traces at COSTS from the recorded scores
    python -m fraud.eval_ieee --report <eval id>      # rebuild the results files from the traces

Traces: fraud/runs/<eval id>_dev_<setup>_dev<k>.jsonl, <eval id>_eval_<setup>_test<k>.jsonl.
Results: fraud/results/eval_ieee/<eval id>.json + .md.

PRE-REGISTRATION (committed before any model scored a row; the smoke read latency and prompt size only).
Owner, 2026-11-02: test one random frozen and one random fine-tuned model on IEEE-CIS, to see which performs
better and whether retraining is necessary. The owner did not answer the three open questions of the handoff,
so these are Implementation choices, made to be re-read from the traces if the owner decides otherwise:
- Question 1: the fine-tune runs AS IS, a PaySim adapter on IEEE rows. This tests transfer. It is not a fine-tune on
  real data. Thresholds and the option order are picked on IEEE dev rows; no weight is changed.
- Draw: numpy default_rng(20261102), one integer per pool, in pool order: frozen from (d1-omni-2b, d3-nano,
  qwen2.5-1.5b), fine-tuned from (20261005-013044_lora, 20261021-141001_lora-d3, 20261102-152745_lora-d1). Result:
  frozen Qwen2.5-1.5B, fine-tuned d1-Omni-2B (adapter 20261102-152745_lora-d1, trained on PaySim for 56 min). LFM3-1.7B
  is out (owner: too slow), d3-mini's weights are gone, and the smoke and 0-byte folders are not adapters. Not substituted.
  These are DIFFERENT base models, so the comparison changes two things at once. It cannot say what fine-tuning alone did.
- Data: train_transaction + train_identity only (the test files have no labels). Split by row order: first 60% fit, next
  20% dev, last 20% test (fraud/ieee_data.py). Test labels are read only by this eval. Natural fraud rate, no enrichment
  and no weights, so every number is a natural-rate number.
- Windows: 10 dev and 20 test windows of 1,000 contiguous rows: the split cut into equal blocks, one start per block from
  default_rng([20261102, 0 dev / 1 test]). Each test window runs once per setup. Nothing is tuned on them. 1,000 rows
  because the smoke gives about 2.5 h in all (Observed: 144 ms per row for the 1.5B, 143 / 87 ms both / one order for d1).
- Prompt: fraud/ieee_brain.py, one prompt with prompt v4's question and option labels and a new context and description.
  Its context states patterns Observed on fit rows only. No prompt development: no model scored a row before this
  commit, and a poor result is reported, not iterated on here.
- No type filter and no rules: every product has fraud in the fit rows, so there is nothing to filter, and PaySim's rules
  do not apply. Every row goes to the setup.
- Balance-column ban: PaySim's rule, not applicable here. IEEE equivalent: no prior isFraud label is ever an input, because
  labels arrive later. Card ids and addresses are not shown to the models or used as numbers. Whether Vesta's C, D and V columns
  are all known at decision time cannot be checked (anonymised); they are used as given and that is a stated limit.
- Question 2, costs (COSTS below; fraud.costs.DEFAULT is untouched): approved fraud loses its amount; a wrongly declined
  payment costs 10% of its amount, at least 1 USD (10% kept from PaySim); a review costs 5 USD, about 5 minutes of an analyst
  at 60 USD an hour (assumption); at most 5% of a window's rows reviewed, in arrival order. PaySim's 50 is not kept: it was
  tiny against PaySim's amounts (median about 75,000) but above most of what a review could save here (fit median 70 USD).
  Amounts taken as USD (from memory of the competition's data page, not verified; the page was not reachable by search).
- When the review budget is spent, a review becomes an approve for every setup: PaySim's "larger score" fallback assumes
  scores that mean something on this data. Implementation choice.
- Thresholds: review and decline lines from dev score quantiles, the alert shares in ALERT_SHARES including "never", minimising
  total dev cost per setup. Quantiles because an untrained scorer may put all its scores in one corner; a fixed grid would
  then decide by accident. Approve-all is on this grid, so every setup can choose to do nothing.
- Setups: ieee-frozen-qwen1.5b (engine, both option orders, as in Phases 3–5); ieee-ft-d1omni (fraud.d1torch with the adapter
  merged, lone prompts; dev reads both orders, Phase 9's debias rule picks the variant for test); ieee-logreg-same (numpy
  logistic regression fitted on fit rows, on the fields the prompt shows); ieee-logreg-all (every column except card and address ids:
  how much the prompt leaves out; no bar); ieee-approve-all.
- No gate: the owner's question needs both models on test whatever dev shows.
- Which performs better: lower total cost over the 20 test windows. Distinguishable only if the 95% bootstrap interval (windows
  resampled, seed 20261102) of the mean paired difference excludes 0. Pooled test AUC, sign test reported.
- Question 3, retraining is NOT necessary only if the better model passes both: (a) cost <= ieee-logreg-same in >= 10 of the 20
  windows, the same count as PaySim's Bar A, and (b) total cost < approve-all's. Otherwise retraining is necessary, by this bar.
  The logistic regression is itself trained on IEEE fit rows, which is the point: it is the price of training on this data.
- Latency bar: p90 per decision <= 250 ms on test, per model (as Phase 5).
- Reported, no bar: cost with 95% CI, AUC with CI, recall, precision, alert rate, reviews, declines, thresholds, dev AUCs, argmax
  shares and argmax cost, and cost sensitivity with thresholds re-picked on dev from the same scores: review 1 USD, review 20 USD,
  and PaySim's numbers as they are (friction minimum 10, review 50).
- Stated before running (Inferred): both models likely lose to logistic regression. A prompt of a dozen fields can't show what the V
  columns carry, the frozen 1.5B never saw this data, and the adapter learned PaySim's type-and-new-account pattern, which these rows
  do not contain. Fraud amount here is about 5–7 USD per row in fit and dev, so the room to save is small against any review fee.

AMENDMENT, owner, 2026-11-02, made while the eval above was running and before any of its output was read. It replaces the
three Implementation choices above with the owner's answers:
- Retraining is treated as guaranteed. The bar for "retraining is necessary" is withdrawn; nothing is decided by it.
- A review costs 10 USD, the default for every review from now on (fraud.costs.DEFAULT). Friction stays 10%, at least 1 USD.
  The running process keeps 5 USD in its traces. The fee only prices decisions, so --reprice re-picks thresholds on dev and rewrites
  every trace at 10 USD from the recorded scores; no model is re-run. Sensitivity: review 1, 5 and 20 USD, and Phases 1–10's costs.
- Train on IEEE-CIS and compare it with the PaySim fine-tune. Primary question now: on the same 20 test windows, which of
  ieee-ft-ieee-d1omni and ieee-ft-d1omni has the lower total cost? Distinguishable only if the 95% bootstrap interval of the mean
  paired difference excludes 0. Same base model, prompt, windows and training size, so unlike the frozen comparison this one changes one thing.
- The adapter: one run of fraud/finetune_ieee.py, configured there: d1-Omni-2B from its frozen weights, the same LoRA, schedule and
  1,510 batches as the PaySim adapter, on 3,021 fraud + 9,063 legit rows drawn from fit rows only, seed 0. Its setup follows Phase 9's rule like the
  PaySim adapter: dev reads both orders, the debias rule picks the variant, thresholds come from dev.
- The frozen-vs-PaySim answer stays reported. Both fine-tunes are reported against ieee-logreg-same, ieee-logreg-all and approve-all, with no bar.
- Stated before running (Inferred): the IEEE fine-tune beats the PaySim one on AUC and probably cost, but may still not beat logistic
  regression on the same fields: 12,084 rows is small, and text is a poor way to read counters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import costs, ieee_brain, ieee_data
from .baselines import LogReg
from .brain import ACTIONS, OPTIONS
from .eval import sign_test
from .eval_d3 import choose_variant
from .eval_ft import BAR_P90_MS, release
from .ieee_data import Ieee
from .runs import RUNS, finished, read_run, stamp, to_jsonl

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_ieee"
MODELS = ROOT / "models"

SEED = 20261102
FROZEN_POOL = ("d1-omni-2b", "d3-nano", "qwen2.5-1.5b")
FINETUNED_POOL = ("20261005-013044_lora", "20261021-141001_lora-d3", "20261102-152745_lora-d1")
FROZEN, FINETUNED, FT_IEEE = "ieee-frozen-qwen1.5b", "ieee-ft-d1omni", "ieee-ft-ieee-d1omni"  # FINETUNED: the PaySim adapter
LOGREG_SAME, LOGREG_ALL, APPROVE_ALL = "ieee-logreg-same", "ieee-logreg-all", "ieee-approve-all"
MODEL_SETUPS = (FROZEN, FINETUNED, FT_IEEE)
SETUPS = (FROZEN, FINETUNED, FT_IEEE, LOGREG_SAME, LOGREG_ALL, APPROVE_ALL)

# USD (Documented: TransactionAmt, the competition's data page). Implementation choice, see the pre-registration.
COSTS = costs.Costs(friction_rate=0.10, friction_min=1.0, review_fee=costs.DEFAULT.review_fee, review_share=0.05)
SENSITIVITY = {"review $1": costs.Costs(0.10, 1.0, 1.0, 0.05), "review $5": costs.Costs(0.10, 1.0, 5.0, 0.05),
               "review $20": costs.Costs(0.10, 1.0, 20.0, 0.05), "Phases 1-10 (review 50, friction min 10)": costs.PHASES_1_10}

WINDOW_SIZE = 1000
DEV_WINDOWS, TEST_WINDOWS = 10, 20
ALERT_SHARES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
ID_LIKE = ("card1", "card2", "addr1", "addr2")  # identifiers, not quantities


def draw(seed: int = SEED) -> tuple[str, str]:
    """The owner's "one random frozen, one random fine-tuned": two independent draws from the pools."""
    rng = np.random.default_rng(seed)
    return FROZEN_POOL[int(rng.integers(len(FROZEN_POOL)))], FINETUNED_POOL[int(rng.integers(len(FINETUNED_POOL)))]


# ---------------------------------------------------------------- windows

@dataclass
class Win:
    id: str
    split: str
    rows: np.ndarray

    def header(self) -> dict:
        return {"id": self.id, "split": self.split, "seed": SEED, "steps": [int(self.rows[0]), int(self.rows[-1])],
                "size": int(len(self.rows)), "dataset": "ieee-cis", "enriched": False}


def make_windows(n: int, split: str, count: int, size: int = WINDOW_SIZE, seed: int = SEED) -> list[Win]:
    """Natural-rate runs of contiguous rows: the split cut into ``count`` equal blocks, one seeded start in each."""
    lo, hi = ieee_data.split_rows(n, split)
    edges = np.linspace(lo, hi, count + 1).astype(np.int64)
    rng = np.random.default_rng([seed, 0 if split == "dev" else 1])
    out = []
    for k in range(count):
        start = int(rng.integers(edges[k], edges[k + 1] - size + 1))
        out.append(Win(f"{split}{k}", split, np.arange(start, start + size)))
    return out


# ---------------------------------------------------------------- baselines

def _one_hot(t: Ieee, name: str, idx: np.ndarray) -> np.ndarray:
    return np.eye(len(t.vocab[name]))[t.cat[idx, t.cat_names.index(name)]][:, 1:]


def _hours(t: Ieee, idx: np.ndarray) -> np.ndarray:
    return np.eye(24)[(t.dt[idx] % 86400) // 3600]


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.sign(x) * np.log1p(np.abs(x)))


def features_same(t: Ieee, idx: np.ndarray) -> np.ndarray:
    """What the prompt shows, as numbers."""
    names = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4", "M6")
    counts = [t.col(k)[idx] for k in (*ieee_brain.COUNTERS, "D1")]
    return np.c_[np.log1p(t.amount[idx]), _hours(t, idx), *[_one_hot(t, k, idx) for k in names],
                 *[_signed_log(c) for c in counts], np.isnan(counts[-1])]


def features_all(t: Ieee, idx: np.ndarray) -> np.ndarray:
    """Every column of the train files except identifiers: how much the prompt leaves out."""
    keep = [j for j, k in enumerate(t.num_names) if k not in ID_LIKE]
    flags = [j for j in keep if not t.num_names[j].startswith("V")]
    x = t.num[idx][:, keep].astype(np.float64)
    return np.c_[np.log1p(t.amount[idx]), _hours(t, idx), *[_one_hot(t, k, idx) for k in t.cat_names],
                 _signed_log(x), np.isnan(t.num[idx][:, flags])]


def fit_logreg(t: Ieee, features) -> LogReg:
    a, _ = ieee_data.split_bounds(len(t))
    idx = np.arange(a)
    return LogReg().fit(features(t, idx), t.is_fraud[idx].astype(float))


# ---------------------------------------------------------------- deciding

def act(q: float, tr: float | None, td: float | None) -> str:
    """Thresholds on the score; None means the action is never taken."""
    return "decline" if td is not None and q >= td else "review" if tr is not None and q >= tr else "approve"


def pick_thresholds(scores, truth, amount, c: costs.Costs) -> tuple[float | None, float | None, list]:
    """(review, decline) from score quantiles of the dev rows, minimising total dev cost. Quantiles, not a fixed
    grid: a scorer that has not seen this dataset may put every score in one corner. Ties: first in grid order."""
    pooled = np.concatenate(scores)
    cut = {s: (None if s == 0 else float(np.quantile(pooled, 1 - s))) for s in ALERT_SHARES}
    grid = []
    for sr in ALERT_SHARES:  # the review line alerts at least as many rows as the decline line
        for sd in ALERT_SHARES:
            if sd > sr:
                continue
            tr, td = cut[sr], cut[sd]
            total = 0.0
            for q, y, a in zip(scores, truth, amount):
                wanted = [act(float(v), tr, td) for v in q]
                total += costs.score(c, wanted, ["approve"] * len(q), y, a).summary["cost"]
            grid.append((tr, td, total))
    best = min(grid, key=lambda g: g[2])
    return best[0], best[1], grid


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC, ties count half."""
    u, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    rank = (np.cumsum(counts) - (counts - 1) / 2)[inv]
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    return float((rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def trace(t: Ieee, w: Win, setup: str, outs: list[dict], thresholds, backend: dict, eval_id: str) -> list[dict]:
    """A fraud-console trace of one window. Before thresholds exist (dev) a model's action is its argmax."""
    wanted = [(act(o["q"], *thresholds) if thresholds else o.get("argmax", "approve")) for o in outs]
    fraud, amount = t.is_fraud[w.rows].astype(bool), t.amount[w.rows]
    scored = costs.score(COSTS, wanted, ["approve"] * len(outs), fraud, amount)
    ticks = []
    for k, (i, o) in enumerate(zip(w.rows, outs)):
        tx = {"type": t.text("ProductCD", int(i)), "amount": round(float(t.amount[i]), 2), "to": "merchant",
              "hour": ieee_brain.hour(t, int(i))}
        rec = {"type": "tx", "i": k, "row": int(i), "step": int((t.dt[i] - t.dt[0]) // 3600), "tx": tx, "signals": {},
               "by": o["by"], "wanted": wanted[k], "action": scored.final[k], "fallback": "approve", "q": o["q"],
               "ms": round(o["ms"], 3), "truth": int(fraud[k]), "cost": round(float(scored.cost[k]), 2),
               "budget_left": scored.budget_left[k], "weight": 1.0}
        for key in ("scores", "orders", "outside_mass", "prompt_tokens"):
            if key in o:
                rec[key] = o[key]
        ticks.append(rec)
    ms = [r["ms"] for r in ticks if r["by"] == "model"]
    summary = {**scored.summary, "setup": setup, "model_decisions": len(ms), "total_ms": round(float(sum(ms)), 1),
               "latency_ms": {"p50": float(np.percentile(ms, 50)) if ms else 0.0,
                              "p90": float(np.percentile(ms, 90)) if ms else 0.0}}
    header = {"type": "header", "app": "fraud", "dataset": "IEEE-CIS", "eval": eval_id, "setup": setup, "window": w.header(),
              "backend": backend, "costs": COSTS.as_dict(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "signals": [],
              "actions": list(costs.ACTIONS), "rules": [], "fraud_free_types": [],
              "thresholds": {"review": thresholds[0], "decline": thresholds[1]} if thresholds else None,
              "options": list(OPTIONS), "question": ieee_brain.QUESTION, "context": ieee_brain.CONTEXT,
              "example_prompt": ieee_brain.describe(t, int(w.rows[0]))}
    return [header] + ticks + [{"type": "end", "summary": summary}]


def trace_file(eval_id: str, stage: str, setup: str, wid: str) -> Path:
    return RUNS / f"{eval_id}_{stage}_{setup}_{wid}.jsonl"


class Qwen15:
    """The frozen 1.5B through the engine, as in Phases 3-5: both option orders, LM head in float32."""

    def __init__(self):
        from .eval_ft import make_model_engine

        self.engine, self.info = make_model_engine("hybrid-1.5b", None)

    def score(self, t: Ieee, i: int, variant: str) -> dict:
        d = ieee_brain.decision(t, i)
        t0 = time.perf_counter()
        r = self.engine.decide_batch([d])[0]
        ms = (time.perf_counter() - t0) * 1000
        p = [float(x) for x in r.probs]
        return {"by": "model", "q": p[2], "argmax": ACTIONS[r.index], "scores": dict(zip(ACTIONS, p)), "orders": r.orders,
                "outside_mass": float(r.outside_mass), "prompt_tokens": int(r.prompt_tokens), "ms": ms}


class D1Fine:
    """d1-Omni-2B with its PaySim adapter merged, as in Phase 9: each order its own lone prompt."""

    def __init__(self, adapter: Path):
        from .d1torch import D1Torch
        from .eval_d3ft import warmed

        man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
        self.model = warmed(D1Torch.load().merge_adapter(adapter / "adapter"))
        self.info = {**self.model.info(), "adapter": {"dir": adapter.name, "sha256": man["adapter_sha256"],
                                                      "rows_sha256": man["rows_sha256"], "steps": man["steps_done"]}}

    def score(self, t: Ieee, i: int, variant: str) -> dict:
        body = ieee_brain.request(t, i, variant)
        t0 = time.perf_counter()
        r = self.model.post(body)
        ms = (time.perf_counter() - t0) * 1000
        orders = [[float(r["answers"][q]["probabilities"][o]) for o in OPTIONS] for q in body["questions"]]
        p = [sum(col) / len(orders) for col in zip(*orders)]
        return {"by": "model", "q": p[2], "argmax": ACTIONS[max(range(3), key=lambda k: p[k])],
                "scores": dict(zip(ACTIONS, p)), "orders": [dict(zip(ACTIONS, o)) for o in orders],
                "prompt_tokens": int(r["usage"]["input_tokens"]), "ms": ms}


def load_model(setup: str, ieee_adapter: Path | None = None):
    return Qwen15() if setup == FROZEN else D1Fine(ieee_adapter if setup == FT_IEEE else MODELS / FINETUNED_POOL[2])


def run_setup(eval_id, stage, setup, ws, t, outs_of, thresholds, backend, log, on_start=None) -> list[list[dict]]:
    out = []
    for w in ws:
        path = trace_file(eval_id, stage, setup, w.id)
        if not finished(path):
            t0 = time.perf_counter()
            recs = trace(t, w, setup, outs_of(w), thresholds, backend() if callable(backend) else backend, eval_id)
            RUNS.mkdir(parents=True, exist_ok=True)
            path.write_text(to_jsonl(recs), encoding="utf-8")
            sm = recs[-1]["summary"]
            log(f"{stage:4} {setup:22} {w.id:7} cost {sm['cost']:>10,.0f}  p90 {sm['latency_ms']['p90']:6.1f} ms  "
                f"({time.perf_counter() - t0:6.1f} s)")
        out.append(read_run(path))
    return out


def view(traces: list[list[dict]], variant: str = "both") -> dict:
    """Per window: scores (of one option order if asked), truth, amount."""
    v = {"scores": [], "truth": [], "amount": []}
    for recs in traces:
        tx = [r for r in recs if r.get("type") == "tx"]
        at = None if variant == "both" else ("fwd", "rev").index(variant)
        v["scores"].append(np.array([r["q"] if at is None else r["orders"][at]["decline"] for r in tx], float))
        v["truth"].append(np.array([r["truth"] for r in tx]))
        v["amount"].append(np.array([r["tx"]["amount"] for r in tx]))
    return v


def pooled_auc(v: dict) -> float:
    return auc(np.concatenate(v["scores"]), np.concatenate(v["truth"]))


def variant_of(setup: str, traces: list[list[dict]]) -> tuple[str, float | None, dict | None]:
    """The 1.5B reads both orders as in Phases 3-5; a d1-Omni fine-tune gets Phase 9's debias rule from its dev traces."""
    if setup == FROZEN:
        return "both", None, None
    aucs = {k: pooled_auc(view(traces, k)) for k in ("both", "fwd", "rev")}
    variant, gain = choose_variant(aucs)
    return variant, gain, aucs


def run(eval_id: str, ieee_adapter: Path, log=print) -> dict:
    t = ieee_data.load()
    dev, test = make_windows(len(t), "dev", DEV_WINDOWS), make_windows(len(t), "test", TEST_WINDOWS)

    # Baselines first, without a model in memory: fitted on fit rows, scored on the same windows.
    lr_outs = {}
    for setup, features in ((LOGREG_SAME, features_same), (LOGREG_ALL, features_all)):
        need = not all(finished(trace_file(eval_id, "dev", setup, w.id)) for w in dev) or \
            not all(finished(trace_file(eval_id, "eval", setup, w.id)) for w in test)
        if need:
            t0 = time.perf_counter()
            lr = fit_logreg(t, features)
            log(f"{setup} fitted in {time.perf_counter() - t0:.0f} s")
            lr_outs[setup] = {w.id: [{"by": "logreg", "q": float(p), "ms": 0.0} for p in lr.predict(features(t, w.rows))]
                              for w in dev + test}
            del lr
    dev_traces = {}
    for setup in (LOGREG_SAME, LOGREG_ALL):
        dev_traces[setup] = run_setup(eval_id, "dev", setup, dev, t, lambda w, s=setup: lr_outs[s][w.id], None, {}, log)
    dev_traces[APPROVE_ALL] = run_setup(eval_id, "dev", APPROVE_ALL, dev, t,
                                        lambda w: [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows], None, {}, log)

    for setup in MODEL_SETUPS:
        model = None

        def ensure():
            nonlocal model
            if model is None:
                model = load_model(setup, ieee_adapter)
            return model

        dev_traces[setup] = run_setup(eval_id, "dev", setup, dev, t,
                                      lambda w: [ensure().score(t, int(i), "both") for i in w.rows], None,
                                      lambda: ensure().info, log)
        variant, _, _ = variant_of(setup, dev_traces[setup])
        v = view(dev_traces[setup], variant)
        tr, td, _ = pick_thresholds(v["scores"], v["truth"], v["amount"], COSTS)
        log(f"{setup}: variant {variant}, thresholds review {tr}, decline {td}")
        run_setup(eval_id, "eval", setup, test, t, lambda w: [ensure().score(t, int(i), variant) for i in w.rows],
                  (tr, td), lambda: ensure().info, log)
        model = None
        release()

    for setup in (LOGREG_SAME, LOGREG_ALL):
        v = view(dev_traces[setup])
        tr, td, _ = pick_thresholds(v["scores"], v["truth"], v["amount"], COSTS)
        run_setup(eval_id, "eval", setup, test, t, lambda w, s=setup: lr_outs[s][w.id], (tr, td), {}, log)
    run_setup(eval_id, "eval", APPROVE_ALL, test, t, lambda w: [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows],
              (None, None), {}, log)
    return report(eval_id, log)


def reprice(eval_id: str, log=print) -> None:
    """Rewrite an eval's traces at COSTS from their recorded scores: new thresholds from dev, no model run."""
    t = ieee_data.load()
    dev, test = make_windows(len(t), "dev", DEV_WINDOWS), make_windows(len(t), "test", TEST_WINDOWS)

    def again(stage, setup, w, thresholds):
        path = trace_file(eval_id, stage, setup, w.id)
        old = read_run(path)
        outs = []
        for r in old:
            if r.get("type") != "tx":
                continue
            o = {k: r[k] for k in ("by", "q", "ms", "scores", "orders", "outside_mass", "prompt_tokens") if k in r}
            if "scores" in r:
                o["argmax"] = max(ACTIONS, key=lambda a: r["scores"][a])
            outs.append(o)
        recs = trace(t, w, setup, outs, thresholds, old[0]["backend"], eval_id)
        recs[0].update(created=old[0]["created"], repriced=time.strftime("%Y-%m-%dT%H:%M:%S"))
        path.write_text(to_jsonl(recs), encoding="utf-8")
        return recs

    for setup in SETUPS:
        if not all(finished(trace_file(eval_id, "dev", setup, w.id)) for w in dev):
            log(f"{setup}: no complete dev traces, skipped")
            continue
        traces = [again("dev", setup, w, None) for w in dev]
        tr, td = None, None
        if setup != APPROVE_ALL:
            v = view(traces, variant_of(setup, traces)[0] if setup in MODEL_SETUPS else "both")
            tr, td, _ = pick_thresholds(v["scores"], v["truth"], v["amount"], COSTS)
        for w in test:
            again("eval", setup, w, (tr, td))
        log(f"{setup}: repriced at review {COSTS.review_fee:g}")


# ---------------------------------------------------------------- the report

def boot_ci(values: np.ndarray, seed: int = SEED, reps: int = 10000) -> tuple[float, float]:
    """Percentile 95% interval of the mean, resampling windows."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (reps, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_auc(v: dict, seed: int = SEED, reps: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n, out = len(v["scores"]), []
    for _ in range(reps):
        k = rng.integers(0, n, n)
        out.append(auc(np.concatenate([v["scores"][j] for j in k]), np.concatenate([v["truth"][j] for j in k])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    wins, losses = int((a < b).sum()), int((a > b).sum())
    return {"cheaper": wins, "dearer": losses, "ties": int((a == b).sum()), "sign_p": sign_test(wins, losses),
            "mean_diff": float((a - b).mean()), "mean_diff_ci95": boot_ci(a - b)}


def compare(costs_by: dict[str, np.ndarray], a: str, b: str) -> dict:
    """The lower total test cost of two setups, and whether the windows can tell them apart."""
    first, second = (a, b) if costs_by[a].sum() <= costs_by[b].sum() else (b, a)
    tied = costs_by[a].sum() == costs_by[b].sum()
    p = paired(costs_by[first], costs_by[second])
    return {"a": a, "b": b, "better": None if tied else first, "other": second, **p,
            "distinguishable": bool(p["mean_diff_ci95"][1] < 0)}


def verdict(costs_by: dict[str, np.ndarray]) -> dict:
    """The primary comparison, then what the owner's first question and the baselines need."""
    return {"ieee_vs_paysim": compare(costs_by, FT_IEEE, FINETUNED), "frozen_vs_paysim": compare(costs_by, FROZEN, FINETUNED),
            "reported": [compare(costs_by, s, b) for s in (FT_IEEE, FINETUNED) for b in (LOGREG_SAME, LOGREG_ALL, APPROVE_ALL)]}


def report(eval_id: str, log=print) -> dict:
    t = ieee_data.load()
    dev, test = make_windows(len(t), "dev", DEV_WINDOWS), make_windows(len(t), "test", TEST_WINDOWS)
    dev_tr = {s: [read_run(trace_file(eval_id, "dev", s, w.id)) for w in dev] for s in SETUPS}
    test_tr = {s: [read_run(trace_file(eval_id, "eval", s, w.id)) for w in test] for s in SETUPS}
    views = {s: view(test_tr[s]) for s in SETUPS}
    win_cost = {s: np.array([r[-1]["summary"]["cost"] for r in test_tr[s]]) for s in SETUPS}
    rows = {}
    for s in SETUPS:
        act_all = [x for r in test_tr[s] for x in r if x.get("type") == "tx"]
        n = len(act_all)
        fraud = [x for x in act_all if x["truth"]]
        alerts = [x for x in act_all if x["action"] != "approve"]
        stopped = [x for x in fraud if x["action"] != "approve"]
        lat = [x["ms"] for x in act_all if x["by"] == "model"]
        rows[s] = {"mean_cost": float(win_cost[s].mean()), "total_cost": float(win_cost[s].sum()),
                   "cost_ci95": boot_ci(win_cost[s]), "auc": pooled_auc(views[s]) if s != APPROVE_ALL else None,
                   "auc_ci95": boot_auc(views[s]) if s != APPROVE_ALL else None,
                   "recall": len(stopped) / max(1, len(fraud)), "precision": len(stopped) / max(1, len(alerts)),
                   "alert_rate": len(alerts) / n, "reviews": sum(x["action"] == "review" for x in act_all),
                   "declines": sum(x["action"] == "decline" for x in act_all),
                   "reviews_wanted": sum(x["wanted"] == "review" for x in act_all),
                   "fraud_amount_stopped": float(sum(x["tx"]["amount"] for x in stopped)),
                   "windows": [float(c) for c in win_cost[s]],
                   "latency_ms": {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90))} if lat else None,
                   "prompt_tokens_mean": float(np.mean([x["prompt_tokens"] for x in act_all if "prompt_tokens" in x])) if lat else None,
                   "thresholds": test_tr[s][0][0].get("thresholds")}
    dev_info = {}
    for s in SETUPS:
        if s == APPROVE_ALL:
            continue
        v = view(dev_tr[s])
        info = {"auc": pooled_auc(v)}
        if s in (FINETUNED, FT_IEEE):
            info["variant"], info["debias_gain"], info["auc_orders"] = variant_of(s, dev_tr[s])
        if s in MODEL_SETUPS:
            tx = [x for r in dev_tr[s] for x in r if x.get("type") == "tx"]
            info["argmax_share"] = {a: sum(x["wanted"] == a for x in tx) / len(tx) for a in ACTIONS}
            info["argmax_cost_mean"] = float(np.mean([r[-1]["summary"]["cost"] for r in dev_tr[s]]))
        dev_info[s] = info
    # Sensitivity, from the same scores: thresholds re-picked on dev per cost set, test re-scored. No bar.
    sens = {}
    for name, c in SENSITIVITY.items():
        sens[name] = {"costs": c.as_dict(), "mean_cost": {}}
        for s in SETUPS:
            if s == APPROVE_ALL:
                tr, td = None, None
            else:
                variant = dev_info[s].get("variant", "both")
                vd = view(dev_tr[s], variant)
                tr, td, _ = pick_thresholds(vd["scores"], vd["truth"], vd["amount"], c)
            v = views[s]
            sens[name]["mean_cost"][s] = float(np.mean([
                costs.score(c, [act(float(x), tr, td) for x in q], ["approve"] * len(q), y, a).summary["cost"]
                for q, y, a in zip(v["scores"], v["truth"], v["amount"])]))
    y_all = np.concatenate(views[FROZEN]["truth"])
    out = {"eval": eval_id, "script": "fraud/eval_ieee.py", "dataset": "IEEE-CIS train files, time split",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "costs": COSTS.as_dict(), "seed": SEED, "draw": draw(),
           "windows": {w.id: w.header()["steps"] for w in dev + test}, "test_rows": int(len(y_all)),
           "test_fraud": int(y_all.sum()), "setups": rows, "dev": dev_info, "sensitivity": sens,
           "verdict": verdict(win_cost),
           "bar_latency": {s: rows[s]["latency_ms"]["p90"] <= BAR_P90_MS for s in MODEL_SETUPS}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{eval_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = markdown(out)
    (RESULTS / f"{eval_id}.md").write_text(md, encoding="utf-8")
    log(md)
    return out


def _th(x) -> str:
    return "never" if x is None else f"{x:.4g}"


def markdown(r: dict) -> str:
    v = r["verdict"]
    L = [f"# IEEE-CIS eval {r['eval']}", "",
         f"Pre-registration: `fraud/eval_ieee.py`. {r['test_rows']:,} test rows, {r['test_fraud']} fraud, natural rate, "
         f"{TEST_WINDOWS} windows of {WINDOW_SIZE}. Costs in USD: {r['costs']}. All numbers Observed. Model scores are uncalibrated.", "",
         "| setup | mean cost | 95% CI | AUC | 95% CI | recall | precision | alert rate | reviews | declines | p90 ms | thresholds (review, decline) |",
         "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for s, x in sorted(r["setups"].items(), key=lambda kv: kv[1]["mean_cost"]):
        a = f"{x['auc']:.4f}" if x["auc"] is not None else "n/a"
        ac = f"{x['auc_ci95'][0]:.3f}–{x['auc_ci95'][1]:.3f}" if x["auc_ci95"] else "n/a"
        lat = f"{x['latency_ms']['p90']:.0f}" if x["latency_ms"] else "n/a"
        th = x["thresholds"] or {}
        L.append(f"| {s} | {x['mean_cost']:,.0f} | {x['cost_ci95'][0]:,.0f}–{x['cost_ci95'][1]:,.0f} | {a} | {ac} | {x['recall']:.3f} | "
                 f"{x['precision']:.3f} | {x['alert_rate']:.3f} | {x['reviews']} | {x['declines']} | {lat} | "
                 f"{_th(th.get('review'))}, {_th(th.get('decline'))} |")
    def line(p):
        return (f"**{p['better'] or 'tie'}** vs {p['other']}: {p['cheaper']} cheaper, {p['dearer']} dearer, {p['ties']} ties; sign test p "
                f"{p['sign_p']:.3g}; mean paired difference {p['mean_diff']:,.0f} USD per window, 95% CI {p['mean_diff_ci95'][0]:,.0f} to "
                f"{p['mean_diff_ci95'][1]:,.0f} → {'distinguishable' if p['distinguishable'] else 'not distinguishable'}")

    L += ["", "## Answers", "",
          f"- IEEE fine-tune vs PaySim fine-tune: {line(v['ieee_vs_paysim'])}.",
          f"- Frozen vs PaySim fine-tune (different base models): {line(v['frozen_vs_paysim'])}.",
          *[f"- {line(p)}." for p in v["reported"]],
          "- Latency bar, p90 <= 250 ms: " + ", ".join(f"{s} {'PASS' if ok else 'FAIL'}" for s, ok in r["bar_latency"].items()) + ".",
          "", "## Dev", "", "| setup | dev AUC | detail |", "|---|---:|---|"]
    for s, d in r["dev"].items():
        extra = []
        if "auc_orders" in d:
            extra.append(f"orders {', '.join(f'{k} {x:.4f}' for k, x in d['auc_orders'].items())} → {d['variant']}")
        if "argmax_share" in d:
            extra.append("argmax " + ", ".join(f"{k} {x:.3f}" for k, x in d["argmax_share"].items()) + f"; argmax mean cost {d['argmax_cost_mean']:,.0f}")
        L.append(f"| {s} | {d['auc']:.4f} | {'; '.join(extra)} |")
    L += ["", "## Cost sensitivity (no bar; thresholds re-picked on dev per row)", "",
          "| cost set | " + " | ".join(SETUPS) + " |", "|---|" + "---:|" * len(SETUPS)]
    for name, x in r["sensitivity"].items():
        L.append(f"| {name} | " + " | ".join(f"{x['mean_cost'][s]:,.0f}" for s in SETUPS) + " |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- smoke

def smoke(n: int, log=print) -> None:
    """Prompt size and per-decision latency on the first n fit rows. Scores and labels are not read or shown."""
    t = ieee_data.load()
    for setup in (FROZEN, FINETUNED):
        model = load_model(setup)
        for variant in ("both",) if setup == FROZEN else ("both", "fwd"):
            got = [model.score(t, i, variant) for i in range(n)]
            ms = [g["ms"] for g in got]
            log(f"{setup} {variant}: tokens {np.mean([g['prompt_tokens'] for g in got]):.0f}, p50 {np.percentile(ms, 50):.0f} ms, "
                f"p90 {np.percentile(ms, 90):.0f} ms")
        model = None
        release()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw", action="store_true", help="print the draw and the windows, then stop")
    ap.add_argument("--smoke", type=int, metavar="N", help="latency and prompt size on N fit rows, then stop")
    ap.add_argument("--resume", metavar="EVAL_ID", help="continue this eval: skip its finished windows")
    ap.add_argument("--report", metavar="EVAL_ID", help="only rebuild the results files from this eval's traces")
    ap.add_argument("--reprice", metavar="EVAL_ID", help="re-decide this eval's traces at COSTS from their scores")
    ap.add_argument("--ieee-adapter", type=Path, help="the folder written by python -m fraud.finetune_ieee")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if args.draw:
        t = ieee_data.load()
        log(f"seed {SEED}: frozen {draw()[0]}, fine-tuned {draw()[1]}")
        for w in make_windows(len(t), "dev", DEV_WINDOWS) + make_windows(len(t), "test", TEST_WINDOWS):
            log(f"{w.id}: rows {w.rows[0]}–{w.rows[-1]}")
    elif args.smoke:
        smoke(args.smoke, log)
    elif args.reprice:
        reprice(args.reprice, log)
    elif args.report:
        report(args.report, log)
    else:
        if args.ieee_adapter is None or not (args.ieee_adapter / "manifest.json").exists():
            ap.error("--ieee-adapter must be a folder with manifest.json and adapter/ (python -m fraud.finetune_ieee)")
        run(args.resume or stamp(), args.ieee_adapter.resolve(), log)


if __name__ == "__main__":
    main()
