"""The hand-written rules: the hybrid's type filter and the rules-only baseline.

Both were written from the training steps (1–300) only, before any test window ran. The numbers
behind each line are in fraud/docs/data.md → Rule cells on the training steps.
"""

from __future__ import annotations

import time

from .data import TYPE_ID, Transactions

# Types with no fraud in the training steps (0 of 2,293,106 rows). The hybrid approves them unseen.
FRAUD_FREE = frozenset({TYPE_ID["CASH_IN"], TYPE_ID["DEBIT"], TYPE_ID["PAYMENT"]})


class TypeFilter:
    """The rule half of the hybrid. ``decide`` returns None for rows the model must judge."""

    by = "filter"

    def decide(self, t: Transactions, s: dict, i: int) -> dict | None:
        t0 = time.perf_counter()
        if t.type[i] in FRAUD_FREE:
            return {"by": self.by, "action": "approve", "fallback": "approve", "rule": "fraud-free type",
                    "ms": (time.perf_counter() - t0) * 1000}
        return None


# The rules-only table: the first matching line decides. (action, fallback, why)
RULES = (
    ("fraud-free type", "approve", "approve", "no fraud in 2.29M training rows of CASH_IN/DEBIT/PAYMENT"),
    ("cash-out of an amount moved this hour", "decline", "decline", "69% fraud; catches 99% of fraud cash-outs"),
    ("round transfer", "decline", "decline", "37 of 39 round TRANSFERs were fraud"),
    ("transfer to a new receiver", "review", "decline", "4.2% fraud; holds 1,686 of 1,691 fraud transfers"),
    ("transfer of an amount moved this hour", "review", "decline", "24% fraud (67 rows)"),
    ("otherwise", "approve", "approve", "0.002% fraud in the remaining cells"),
)


def rule_for(t: Transactions, s: dict, i: int) -> int:
    typ = t.type[i]
    same = s["same_amount_step_before"][i] > 0
    if typ in FRAUD_FREE:
        return 0
    if typ == TYPE_ID["CASH_OUT"] and same:
        return 1
    if typ == TYPE_ID["TRANSFER"]:
        if s["round_amount"][i]:
            return 2
        if s["dest_in_before"][i] == 0:
            return 3
        if same:
            return 4
    return 5


class RulesOnly:
    by = "rules"

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        t0 = time.perf_counter()
        name, action, fallback, _ = RULES[rule_for(t, s, i)]
        return {"by": self.by, "action": action, "fallback": fallback, "rule": name,
                "ms": (time.perf_counter() - t0) * 1000}
