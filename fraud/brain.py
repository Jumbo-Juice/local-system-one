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

ACTIONS = ("approve", "review", "decline")
OPTIONS = (
    "approve: let it through",
    "review: hold it for a human analyst",
    "decline: block it",
)
QUESTION = "What should happen to this transaction?"
CONTEXT = (
    "Mobile-money service. Fraudsters take over an account, TRANSFER its money to another "
    "account, then CASH_OUT. Most transactions are legitimate. Approving fraud loses the amount; "
    "declining a real customer loses business; review is cheap but limited."
)


def _times(n: int) -> str:
    return "never" if n == 0 else "once" if n == 1 else f"{n} times"


def describe(t: Transactions, s: dict, i: int) -> str:
    """The state the model reads: about 60 words, no account ids, no balances."""
    kind = TYPES[t.type[i]]
    to = "a merchant" if t.dest_merchant[i] else "a customer"
    lines = [
        f"{kind} of {t.amount[i]:,.2f} to {to}, hour {int(s['hour'][i])}:00.",
        f"Sender: sent {_times(int(s['orig_out_before'][i]))} before, received {_times(int(s['orig_in_before'][i]))}.",
        f"Receiver: received {_times(int(s['dest_in_before'][i]))} before "
        f"({_times(int(s['dest_transfer_in_before'][i]))} by TRANSFER), sent {_times(int(s['dest_out_before'][i]))}.",
        f"Same amount moved earlier this hour: {_times(int(s['same_amount_step_before'][i]))}.",
        f"Round amount: {'yes' if s['round_amount'][i] else 'no'}.",
    ]
    return "\n".join(lines)


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


def warm_up(engine) -> None:
    """The first XPU passes compile kernels and allocate buffers; keep them out of the run."""
    filler = "TRANSFER of 1,000.00 to a customer, hour 3:00. Sender: sent never before." * 3
    for _ in range(3):
        engine.decide_batch([Decision(QUESTION, OPTIONS, state=filler, context=CONTEXT)])


def percentile(ms: list[float], q: float) -> float:
    return float(np.percentile(ms, q)) if ms else 0.0
