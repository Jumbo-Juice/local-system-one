"""Train/test split by time and seeded, fraud-enriched windows.

A window is a contiguous span of hours from one split. It keeps ``fraud_target`` fraud rows (whole
TRANSFER→CASH_OUT pairs where possible) and fills up to ``size`` with legit rows sampled uniformly
from the same span, in arrival order. Natural fraud is ~0.1% in training, so a natural 500-row
window would hold 0–1 frauds. Each row gets a weight that maps the window back to the natural
rate of its own span (fraud/docs/data.md → Time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TYPE_ID, Transactions

TRAIN = (1, 300)  # Implementation choice (fraud/docs/data.md → Time)
TEST = (301, 743)
SPLITS = {"dev": TRAIN, "test": TEST}


@dataclass
class Window:
    id: str
    split: str
    seed: int
    steps: tuple[int, int]
    rows: np.ndarray     # row indices in arrival order
    weight: np.ndarray   # per row: natural-rate weight
    fraud: int           # fraud rows kept
    span_rows: int
    span_fraud: int

    def header(self) -> dict:
        return {"id": self.id, "split": self.split, "seed": self.seed, "steps": list(self.steps),
                "size": int(len(self.rows)), "fraud": self.fraud,
                "span_rows": self.span_rows, "span_fraud": self.span_fraud,
                "natural_fraud_rate": self.span_fraud / max(1, self.span_rows)}


def fraud_groups(t: Transactions, rows: np.ndarray) -> list[np.ndarray]:
    """Fraud rows (sorted) grouped so a TRANSFER and the CASH_OUT of the same amount right after
    it stay together."""
    groups, i = [], 0
    while i < len(rows):
        r = rows[i]
        if (i + 1 < len(rows) and rows[i + 1] == r + 1 and t.type[r] == TYPE_ID["TRANSFER"]
                and t.type[r + 1] == TYPE_ID["CASH_OUT"] and t.amount[r] == t.amount[r + 1]):
            groups.append(rows[i:i + 2])
            i += 2
        else:
            groups.append(rows[i:i + 1])
            i += 1
    return groups


def make(t: Transactions, split: str, seed: int, size: int = 500, fraud_target: int = 25,
         wid: str | None = None) -> Window:
    lo, hi = SPLITS[split]
    if np.any(np.diff(t.step) < 0):
        raise ValueError("rows must be sorted by step")
    rng = np.random.default_rng([seed, 0 if split == "dev" else 1])
    first = np.searchsorted(t.step, np.arange(lo, hi + 2))  # row where each step starts
    fraud_cum = np.r_[0, np.cumsum(t.is_fraud)]

    def counts(a: int, b: int) -> tuple[int, int]:  # rows and fraud in steps a..b
        r0, r1 = first[a - lo], first[b + 1 - lo]
        return r1 - r0, int(fraud_cum[r1] - fraud_cum[r0])

    a = b = int(rng.integers(lo, hi + 1))
    while True:
        n, nf = counts(a, b)
        if nf >= fraud_target and n - nf >= size - fraud_target:
            break
        if b < hi:
            b += 1
        elif a > lo:
            a -= 1
        else:
            raise ValueError("split too small for this window")
    r0, r1 = first[a - lo], first[b + 1 - lo]
    span = np.arange(r0, r1)
    is_f = t.is_fraud[span].astype(bool)
    groups = fraud_groups(t, span[is_f])
    picked: list[int] = []
    for g in rng.permutation(len(groups)):
        if len(picked) >= fraud_target:
            break
        picked.extend(groups[g].tolist())
    legit = rng.choice(span[~is_f], size - len(picked), replace=False)
    rows = np.sort(np.r_[np.array(picked, np.int64), legit])
    span_rows, span_fraud = int(len(span)), int(is_f.sum())
    w_f = span_fraud / len(picked)
    w_l = (span_rows - span_fraud) / len(legit)
    weight = np.where(t.is_fraud[rows].astype(bool), w_f, w_l)
    return Window(wid or f"{split}{seed}", split, seed, (a, b), rows, weight, len(picked), span_rows, span_fraud)
