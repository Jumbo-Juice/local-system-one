"""The cost model and the review budget. Every setup is scored by ``score`` and nothing else.

Implementation choice (fraud/PROGRESS.md → Phase 1): approving fraud loses its amount; declining a
legit transaction costs ``friction_rate`` × amount (at least ``friction_min``); a review costs
``review_fee`` and finds the truth (fraud blocked, legit approved). A window may review at most
``review_share`` of its transactions, in arrival order; once the budget is used up, a review
becomes the decider's own fallback (approve or decline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ACTIONS = ("approve", "review", "decline")


@dataclass(frozen=True)
class Costs:
    friction_rate: float
    friction_min: float
    review_fee: float
    review_share: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scored:
    final: list[str]        # action after the budget (review may become its fallback)
    cost: np.ndarray        # per transaction
    budget_left: list[int]  # reviews left after each transaction
    summary: dict


def unit_cost(c: Costs, action: str, fraud: bool, amount: float) -> float:
    if action == "review":
        return c.review_fee
    if action == "approve":
        return amount if fraud else 0.0
    if action == "decline":
        return 0.0 if fraud else max(c.friction_min, c.friction_rate * amount)
    raise ValueError(action)


def score(c: Costs, wanted: list[str], fallback: list[str], fraud: np.ndarray, amount: np.ndarray,
          weight: np.ndarray | None = None) -> Scored:
    """``wanted[i]`` is the decider's action, ``fallback[i]`` its approve/decline choice if a review
    cannot be had. ``weight`` (optional) reweights each transaction to the natural fraud rate."""
    n = len(wanted)
    budget = int(np.floor(c.review_share * n))
    final, cost, left = [], np.zeros(n), []
    for i in range(n):
        a = wanted[i]
        if a == "review":
            if budget > 0:
                budget -= 1
            else:
                a = fallback[i]
        if a not in ACTIONS or (a == "review" and wanted[i] != "review"):
            raise ValueError(f"bad action {a!r} at {i}")
        final.append(a)
        cost[i] = unit_cost(c, a, bool(fraud[i]), float(amount[i]))
        left.append(budget)
    return Scored(final, cost, left, summarize(final, wanted, cost, fraud, amount, weight))


def summarize(final, wanted, cost, fraud, amount, weight=None) -> dict:
    fraud = np.asarray(fraud, bool)
    act = np.asarray(final)
    confusion = {truth: {a: int(((act == a) & (fraud == f)).sum()) for a in ACTIONS}
                 for truth, f in (("legit", False), ("fraud", True))}
    stopped = fraud & (act != "approve")  # fraud declined or caught in review
    alerts = act != "approve"
    out = {
        "n": int(len(act)),
        "fraud": int(fraud.sum()),
        "cost": float(cost.sum()),
        "fraud_amount": float(amount[fraud].sum()),
        "fraud_amount_stopped": float(amount[stopped].sum()),
        "recall": float(stopped.sum() / max(1, fraud.sum())),
        "precision": float(stopped.sum() / max(1, alerts.sum())),
        "alert_rate": float(alerts.mean()) if len(act) else 0.0,
        "reviews_wanted": int(sum(a == "review" for a in wanted)),
        "reviews_done": int((act == "review").sum()),
        "confusion": confusion,
    }
    if weight is not None:
        w = np.asarray(weight, float)
        out["cost_natural"] = float((w * cost).sum())
        out["precision_natural"] = float((w * stopped).sum() / max(1e-12, (w * alerts).sum()))
    return out
