"""Phase 18's additions to the amount-aware rule (fraud/eval_ieee_amount.py). All numpy, all on the current row and earlier rows only.

- recent_prior: a label-free estimate of how much fraud the recent rows hold (Saerens, Latinne and Decaestecker 2002, EM on predicted probabilities; Documented from memory, not checked against the paper).
- shift_p: moves p to that prior without changing the ranking.
- isotonic: pool-adjacent-violators, a monotone map from p to the fraud rate on the fit rows.
- decide_paced: the rule, with the price of a review following how many reviews remain, and an extra flat price on declining in the expected cost. It prices a decision only: scoring stays with fraud.costs.
"""

from __future__ import annotations

import numpy as np

from . import costs


def recent_prior(p: np.ndarray, train_prior: float, iters: int = 50) -> float:
    """The fraud share these rows' p imply if only the share moved since the fit. No label is read."""
    pi = train_prior
    for _ in range(iters):
        odds = (pi / train_prior) * p
        q = odds / (odds + ((1 - pi) / (1 - train_prior)) * (1 - p))
        new = float(q.mean())
        if abs(new - pi) < 1e-7:
            return new
        pi = new
    return pi


def shift_p(p: np.ndarray, train_prior: float, new_prior: float) -> np.ndarray:
    a = p * (new_prior / train_prior)
    return a / (a + (1 - p) * ((1 - new_prior) / (1 - train_prior)))


class Isotonic:
    """Pool-adjacent-violators on a sample, then a step function. Never below or above what its pools saw."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Isotonic":
        order = np.argsort(x, kind="stable")
        xs, ys = x[order], y[order].astype(float)
        means, weights, ends = [], [], []
        for k in range(len(xs)):
            m, w = ys[k], 1.0
            while means and means[-1] >= m:
                m = (means[-1] * weights[-1] + m * w) / (weights[-1] + w)
                w += weights[-1]
                means.pop(), weights.pop(), ends.pop()
            means.append(m), weights.append(w), ends.append(xs[k])
        self.ends, self.means = np.array(ends), np.array(means)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.means[np.minimum(np.searchsorted(self.ends, x, side="left"), len(self.ends) - 1)]


def pace(budget_left: int, rows_left: int, share: float) -> float:
    """1 when reviews remain at the window's rate, above 1 when there are more than the rows need, below 1 when fewer."""
    return (budget_left / max(1, rows_left)) / share


def decide_paced(p: np.ndarray, amount: np.ndarray, lam: float, c: costs.Costs, gamma: float = 0.0, flat: float = 0.0) -> tuple[list[str], list[str]]:
    """(wanted, fallback) per row, in arrival order. lambda_i = lam * pace ** -gamma; gamma 0 and flat 0 are exactly am.decide.
    flat adds to the priced cost of declining only here. The scorer is told nothing about it."""
    approve = p * amount
    decline = (1 - p) * (c.friction(amount) + flat)
    fallback = np.where(approve <= decline, "approve", "decline").tolist()
    if not gamma:
        wanted = np.where(c.review_fee + lam < np.minimum(approve, decline), "review", np.array(fallback)).tolist()
        return wanted, fallback
    n, budget, wanted = len(p), int(np.floor(c.review_share * len(p))), []
    low = np.minimum(approve, decline)
    for i in range(n):
        if budget > 0:
            price = lam * pace(budget, n - i, c.review_share) ** -gamma
            if c.review_fee + price < low[i]:
                wanted.append("review")
                budget -= 1
                continue
        wanted.append(fallback[i])
    return wanted, fallback
