"""The model's decision: one transaction and its signals → approve / review / decline.

One ``Decision`` per transaction, decided alone (a stream: the latency is per transaction), read
in both option orders and averaged (``order_debias``): with three options the 1.5B otherwise follows
the option position (docs/research.md → Goal selection is sensitive to option order).
"""

from __future__ import annotations

import time

import numpy as np

from system_one import Decision

from .data import TYPES, Transactions

ACTIONS = ("approve", "review", "decline")  # option k of OPTIONS is action k
# Prompt v4 of fraud/dev/promptdev.py, frozen on 2026-10-03 (fraud/PROGRESS.md): the best ranking of
# v0-v7 on dev windows (AUC 0.90-0.93). Like every variant, its top option is "yes" for nearly every
# transaction; the owner chose to keep argmax and report that.
OPTIONS = (
    "no: approve it",
    "unsure: send it to an analyst",
    "yes: decline it",
)
QUESTION = "Does this transaction match the fraud pattern?"
CONTEXT = (
    "Mobile-money service. Fraudsters take over an account, TRANSFER all its money to a fresh account "
    "that never received money before, then CASH_OUT exactly that amount within the same hour. "
    "Legitimate TRANSFERs usually go to known accounts; legitimate CASH_OUTs rarely repeat an amount "
    "moved earlier in the hour. Most transactions are legitimate."
)


def _times(n: int) -> str:
    return "never" if n == 0 else "once" if n == 1 else f"{n} times"


def describe(t: Transactions, s: dict, i: int) -> str:
    """The state the model reads: the signals as flags, no account ids, no balances."""
    din = int(s["dest_in_before"][i])
    same = int(s["same_amount_step_before"][i])
    recv = ("NEW account: it never received money before" if din == 0
            else f"known account: received money {_times(din)} before")
    amt = (f"the SAME amount was already moved {_times(same)} earlier this hour" if same
           else "no earlier transaction this hour had this amount")
    return "\n".join([
        f"{TYPES[t.type[i]]} of {t.amount[i]:,.2f}, hour {int(s['hour'][i])}:00.",
        f"Receiver: {recv}.",
        f"Amount: {amt}.",
        f"Round amount: {'yes' if s['round_amount'][i] else 'no'}.",
    ])


def decision(t: Transactions, s: dict, i: int) -> Decision:
    return Decision(QUESTION, OPTIONS, state=describe(t, s, i), context=CONTEXT)


class ModelDecider:
    by = "model"

    def __init__(self, engine):
        self.engine = engine

    def prompt(self, t: Transactions, s: dict, i: int) -> str:
        d = decision(t, s, i)
        return self.engine.render(d, self.engine.labels_for(len(d.options)))

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        d = decision(t, s, i)
        t0 = time.perf_counter()
        r = self.engine.decide_batch([d])[0]
        ms = (time.perf_counter() - t0) * 1000
        p = [float(x) for x in r.probs]
        return {
            "by": self.by,
            "action": ACTIONS[r.index],
            "fallback": "approve" if p[0] >= p[2] else "decline",
            "scores": dict(zip(ACTIONS, p)),
            "orders": r.orders,
            "outside_mass": float(r.outside_mass),
            "prompt_tokens": int(r.prompt_tokens),
            "ms": ms,
        }


class ThresholdDecider(ModelDecider):
    """The model's order-averaged "decline" score mapped to an action by two thresholds picked on
    dev windows (fraud/eval_ft.py), like the logistic regression baseline. Used for the fine-tuned
    model, whose scores rank but whose argmax is not the cost-optimal action."""

    def __init__(self, engine, t_review: float, t_decline: float):
        super().__init__(engine)
        self.t_review, self.t_decline = t_review, t_decline

    def apply(self, out: dict) -> dict:
        q = out["scores"]["decline"]
        out["action"] = "decline" if q >= self.t_decline else "review" if q >= self.t_review else "approve"
        out["thresholds"] = {"review": self.t_review, "decline": self.t_decline}
        return out

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        return self.apply(super().decide(t, s, i))


def load_adapter(engine, path) -> dict:
    """Merge a LoRA adapter (fraud/finetune.py) into the HF backend's model in place.

    Merged weights run at the base model's speed. The float32 LM head is untouched: LoRA only
    targets the attention projections. Returns the adapter's manifest for the trace header.
    """
    import json
    from pathlib import Path

    from peft import PeftModel

    path = Path(path)
    backend = engine.backend
    backend.model = PeftModel.from_pretrained(backend.model, path / "adapter").merge_and_unload().eval()
    if getattr(backend, "prefix_cache", None) is not None:
        backend.prefix_cache = type(backend.prefix_cache)(backend.prefix_cache.size)
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def warm_up(engine) -> None:
    """The first XPU passes compile kernels and allocate buffers; keep them out of the run."""
    filler = "TRANSFER of 1,000.00, hour 3:00. Receiver: NEW account: it never received money before." * 3
    for _ in range(3):
        engine.decide_batch([Decision(QUESTION, OPTIONS, state=filler, context=CONTEXT)])


def percentile(ms: list[float], q: float) -> float:
    return float(np.percentile(ms, q)) if ms else 0.0
