"""Named feature sets over ev.features_all and fraud/history.py, for Phases 19-20. A set is a list of parts, and a part reads only its own row and earlier rows (fraud/history.py).

Parts: history (per-card and per-address counts, velocity, time since last, amount against the past); steady (the same minus lifetime counts, which keep growing as a dataset ages); extra (cents and calendar shape);
identity (train_identity's numeric columns, missing flags, top categories); inter (product by amount, by velocity, by first-seen); uid (history keyed on D1's start day, so it leans on D1).
"""

from __future__ import annotations

import numpy as np

from . import eval_column_audit as au, eval_ieee as ev, history as h

GROWS = ("_log_before", "_first", "_log_distinct_", "_log_same_amt")  # lifetime counts: their level rises with the row number, whatever the card does


def steady_names(names) -> list[str]:
    return [k for k in names if not any(g in k for g in GROWS)]


class FeatureSet:
    """features(t, idx) for one set, reading the history cache once."""

    def __init__(self, parts: tuple[str, ...] = (), drop: str = "", hist: dict | None = None):
        self.parts, self.drop = tuple(parts), drop
        self.base = au.features_without(drop)
        self.hist = hist if hist is not None else (h.load() if parts else None)
        self.names_hist = list(self.hist["history"]) if self.hist else []
        self.names_steady = steady_names(self.names_hist) if self.hist else []

    def __call__(self, t, idx: np.ndarray) -> np.ndarray:
        cols = [self.base(t, idx)]
        hist = self.hist
        for part in self.parts:
            if part == "history":
                cols.append(np.stack([hist["history"][k][idx] for k in self.names_hist], 1))
            elif part == "steady":
                cols.append(np.stack([hist["history"][k][idx] for k in self.names_steady], 1))
            elif part == "extra":
                cols.append(np.stack([v[idx] for v in hist["extra"].values()], 1))
            elif part == "identity":
                cols.append(np.stack([v[idx] for v in hist["numeric"].values()], 1))
                for name, c in hist["one_hot"].items():
                    width = int(c.max()) + 1
                    cols.append(np.eye(width)[c[idx]][:, 1:])
            elif part == "uid":
                cols.append(np.stack([v[idx] for v in hist["uid"].values()], 1))
            elif part == "inter":
                cols.append(self.interactions(t, idx))
            else:
                raise KeyError(part)
        return np.concatenate([c.astype(np.float64) for c in cols], 1)

    def interactions(self, t, idx: np.ndarray) -> np.ndarray:
        hist = self.hist["history"]
        product = ev._one_hot(t, "ProductCD", idx)
        keys = (np.log1p(t.amount[idx]), hist["card_log_1d"][idx], hist["card_log_since"][idx], hist["card_amt_vs_mean"][idx])
        return np.concatenate([product * k[:, None] for k in keys], 1)
