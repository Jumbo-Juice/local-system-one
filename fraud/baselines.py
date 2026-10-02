"""Baselines next to the model: logistic regression (numpy), random, approve-all.

Logistic regression reads the same signals as the prompt (plus the transaction's type and amount).
It is fitted on training steps only. Its two thresholds (review, decline) are picked by cost on
the dev windows, a fixed grid, before any test window runs (fraud/eval.py).
"""

from __future__ import annotations

import time

import numpy as np

from .data import TYPE_ID, Transactions
from .signals import NAMES

COUNTS = [k for k in NAMES if k.endswith("_before")]


def features(t: Transactions, s: dict, idx: np.ndarray) -> np.ndarray:
    hour = s["hour"][idx]
    cols = [
        (t.type[idx] == TYPE_ID["TRANSFER"]).astype(float),
        (t.type[idx] == TYPE_ID["CASH_OUT"]).astype(float),
        np.log1p(t.amount[idx]),
        t.dest_merchant[idx].astype(float),
        ((hour >= 1) & (hour <= 6)).astype(float),  # night hours
        s["round_amount"][idx].astype(float),
        *[np.log1p(s[k][idx]) for k in COUNTS],
    ]
    return np.stack(cols, axis=1)


FEATURES = ("is_transfer", "is_cash_out", "log_amount", "dest_merchant", "night", "round_amount",
            *[f"log1p_{k}" for k in COUNTS])


class LogReg:
    """L2-regularised logistic regression fitted by Newton's method (IRLS)."""

    def __init__(self, l2: float = 1e-3, iters: int = 25):
        self.l2, self.iters = l2, iters
        self.mu = self.sd = self.w = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogReg":
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = np.c_[np.ones(len(x)), (x - self.mu) / self.sd]
        w = np.zeros(z.shape[1])
        reg = self.l2 * np.eye(len(w))
        reg[0, 0] = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-z @ w))
            g = z.T @ (p - y) / len(y) + reg @ w
            h = (z * (p * (1 - p))[:, None]).T @ z / len(y) + reg
            step = np.linalg.solve(h, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.c_[np.ones(len(x)), (x - self.mu) / self.sd]
        return 1 / (1 + np.exp(-z @ self.w))

    def to_dict(self) -> dict:
        return {"features": FEATURES, "mean": self.mu.tolist(), "sd": self.sd.tolist(), "weights": self.w.tolist()}


class LogRegDecider:
    by = "logreg"

    def __init__(self, model: LogReg, t_review: float, t_decline: float):
        self.model, self.t_review, self.t_decline = model, t_review, t_decline

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        t0 = time.perf_counter()
        p = float(self.model.predict(features(t, s, np.array([i])))[0])
        action = "decline" if p >= self.t_decline else "review" if p >= self.t_review else "approve"
        return {"by": self.by, "action": action, "fallback": "decline" if p >= 0.5 else "approve",
                "scores": {"fraud": p}, "ms": (time.perf_counter() - t0) * 1000}


class RandomDecider:
    by = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        a = ("approve", "review", "decline")[int(self.rng.integers(3))]
        return {"by": self.by, "action": a, "fallback": ("approve", "decline")[int(self.rng.integers(2))], "ms": 0.0}


class ApproveAll:
    by = "all"

    def decide(self, t: Transactions, s: dict, i: int) -> dict:
        return {"by": self.by, "action": "approve", "fallback": "approve", "ms": 0.0}
