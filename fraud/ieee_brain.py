"""The model's decision on IEEE-CIS: one payment → approve / review / decline.

QUESTION and OPTIONS are prompt v4's (fraud/brain.py), so a PaySim fine-tune still reads the labels it
was trained on. The transaction text is new: CONTEXT and describe() read only what a row shows. CONTEXT's
patterns are Observed on the fit rows (fraud/DATA-IEEE.md). One prompt, written before any model scored
a row, never tuned on scores.
"""

from __future__ import annotations

import math

from system_one import Decision

from .brain import ACTIONS, OPTIONS, QUESTION
from .ieee_data import Ieee

CONTEXT = (
    "Online card payments in US dollars. About 4 in 100 are fraud. Fraud is more common on product C, on credit "
    "cards, when a device is recorded (mobile most of all), when a recipient email is given, between hours 4:00 "
    "and 10:00, with match code M4 = M2, with counters C4 or C8 above 0, and when D1 is 0. Most payments are legitimate."
)
COUNTERS = ("C1", "C4", "C8", "C13")


def hour(t: Ieee, i: int) -> int:
    """TransactionDT is seconds from an unstated point, so this is a repeating 24-hour clock, not local time."""
    return int(t.dt[i] % 86400) // 3600


def _value(v: float) -> str:
    return "none" if math.isnan(v) else f"{v:g}"


def describe(t: Ieee, i: int) -> str:
    """The state the model reads: named fields and four counters, no card ids, no V columns."""
    card = " ".join(k for k in (t.text("card4", i), t.text("card6", i)) if k) or "unknown"
    counters = ", ".join(f"{k} {_value(float(t.col_at(k, i)))}" for k in COUNTERS)
    return "\n".join([
        f"Payment of {t.amount[i]:,.2f} USD, product {t.text('ProductCD', i)}, hour {hour(t, i)}:00.",
        f"Card: {card}.",
        f"Purchaser email: {t.text('P_emaildomain', i) or 'none'}. Recipient email: {t.text('R_emaildomain', i) or 'none'}.",
        f"Device: {t.text('DeviceType', i) or 'none recorded'}.",
        f"Match codes: M4 {t.text('M4', i) or 'none'}, M6 {t.text('M6', i) or 'none'}.",
        f"Counters: {counters}. D1: {_value(float(t.col_at('D1', i)))}.",
    ])


def decision(t: Ieee, i: int) -> Decision:
    return Decision(QUESTION, OPTIONS, state=describe(t, i), context=CONTEXT)


def request(t: Ieee, i: int, orders: str) -> dict:
    """The /v1/systemone body for row i, one choice question per option order (as d3.request)."""
    d = decision(t, i)
    q = lambda opts: {"type": "choice", "instructions": d.instruction, "criteria": {o: None for o in opts}}
    questions = {}
    if orders in ("both", "fwd"):
        questions["fwd"] = q(OPTIONS)
    if orders in ("both", "rev"):
        questions["rev"] = q(OPTIONS[::-1])
    return {"state": f"{d.context}\n\n{d.state}", "questions": questions}


__all__ = ["ACTIONS", "CONTEXT", "COUNTERS", "OPTIONS", "QUESTION", "decision", "describe", "hour", "request"]
