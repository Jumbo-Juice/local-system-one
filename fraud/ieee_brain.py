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
MORE_COUNTERS = ("C2", "C5", "C6", "C9", "C11", "C14")
DAYS = ("D2", "D3", "D4", "D10", "D15")


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


def _ago(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.0f} hours" if seconds < 2 * 86400 else f"{seconds / 86400:.0f} days"


def describe_rich(t: Ieee, i: int, hist: dict) -> str:
    """describe() and more of the row, plus what earlier rows say about its card and address (fraud/history.py). Phase 21. No label is read."""
    h = hist["history"]
    count = lambda k: int(round(math.expm1(float(h[k][i]))))
    before = count("card_log_before")
    if before:
        ratio = math.exp(float(h["card_amt_vs_mean"][i]))
        card = f"{before} earlier payments, the last one {_ago(math.expm1(float(h['card_log_since'][i])))} ago, {count('card_log_1d')} in the last day, {count('card_log_7d')} in the last 7 days; this amount is {ratio:.1f} times its usual."
    else:
        card = "no earlier payments."
    more = ", ".join(f"{k} {_value(float(t.col_at(k, i)))}" for k in MORE_COUNTERS)
    days = ", ".join(f"{k} {_value(float(t.col_at(k, i)))}" for k in DAYS)
    lines = [describe(t, i), f"Card history: {card}", f"Address: {count('addr_log_distinct_card')} other cards seen there before. Distance: {_value(float(t.col_at('dist1', i)))}.",
             f"More counters: {more}. More days: {days}."]
    if "signal" in hist:
        risk = float(hist["signal"][i])
        if math.isnan(risk):
            raise ValueError(f"row {i} has no tabular score")
        lines.append(f"Risk score from a tabular model: {risk:.3f} (uncalibrated; higher is more suspicious).")
    return "\n".join(lines)


def decision(t: Ieee, i: int, hist: dict | None = None) -> Decision:
    return Decision(QUESTION, OPTIONS, state=describe_rich(t, i, hist) if hist else describe(t, i), context=CONTEXT)


def request(t: Ieee, i: int, orders: str, hist: dict | None = None) -> dict:
    """The /v1/systemone body for row i, one choice question per option order (as d3.request)."""
    d = decision(t, i, hist)
    q = lambda opts: {"type": "choice", "instructions": d.instruction, "criteria": {o: None for o in opts}}
    questions = {}
    if orders in ("both", "fwd"):
        questions["fwd"] = q(OPTIONS)
    if orders in ("both", "rev"):
        questions["rev"] = q(OPTIONS[::-1])
    return {"state": f"{d.context}\n\n{d.state}", "questions": questions}


__all__ = ["ACTIONS", "CONTEXT", "COUNTERS", "OPTIONS", "QUESTION", "decision", "describe", "describe_rich", "hour", "request"]
