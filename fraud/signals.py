"""Per-transaction signals, each from earlier rows only (file order = arrival order).

The prompt, the rules and the logistic-regression baseline all read these same signals, so the
setups differ in how they decide, not in what they see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import DATA, TYPE_ID, Transactions

CACHE = DATA / "signals.npz"

NAMES = (
    "hour",                     # step % 24
    "orig_out_before",          # times the sender sent money before
    "orig_in_before",           # times the sender received money before
    "dest_in_before",           # times the receiver received money before
    "dest_out_before",          # times the receiver sent money before
    "dest_transfer_in_before",  # TRANSFERs the receiver received before
    "same_amount_step_before",  # earlier rows in the same hour with exactly this amount
    "round_amount",             # amount is a whole multiple of 1,000
)


def prior_sum(keys: np.ndarray, weights: np.ndarray | None = None, order: np.ndarray | None = None) -> np.ndarray:
    """For each element: the sum of ``weights`` over elements with the same key and a strictly
    smaller ``order`` (default: position). Ties in ``order`` do not count each other."""
    n = len(keys)
    order = np.arange(n) if order is None else np.asarray(order)
    w = np.ones(n, np.int64) if weights is None else np.asarray(weights, np.int64)
    idx = np.lexsort((order, keys))
    k, o, ws = keys[idx], order[idx], w[idx]
    csum = np.concatenate(([0], np.cumsum(ws)))  # csum[j] = sum of ws[:j]
    pos = np.arange(n)
    new_key = np.r_[True, k[1:] != k[:-1]]
    new_block = new_key | np.r_[True, o[1:] != o[:-1]]
    key_start = np.maximum.accumulate(np.where(new_key, pos, 0))
    block_start = np.maximum.accumulate(np.where(new_block, pos, 0))
    out = np.empty(n, np.int64)
    out[idx] = csum[block_start] - csum[key_start]
    return out


def compute(t: Transactions) -> dict[str, np.ndarray]:
    n = len(t)
    rows = np.arange(n)
    transfer = (t.type == TYPE_ID["TRANSFER"]).astype(np.int64)
    # One event per side of every row: the sender's "out" and the receiver's "in".
    acct = np.concatenate([t.orig, t.dest])
    order = np.concatenate([rows, rows])
    is_out = np.r_[np.ones(n, np.int64), np.zeros(n, np.int64)]
    out_before = prior_sum(acct, is_out, order)
    in_before = prior_sum(acct, 1 - is_out, order)
    transfer_in_before = prior_sum(acct, np.r_[np.zeros(n, np.int64), transfer], order)
    cents = np.round(t.amount * 100).astype(np.int64)
    _, pair = np.unique(np.stack([t.step.astype(np.int64), cents]), axis=1, return_inverse=True)
    return {
        "hour": (t.step % 24).astype(np.int64),
        "orig_out_before": out_before[:n],
        "orig_in_before": in_before[:n],
        "dest_in_before": in_before[n:],
        "dest_out_before": out_before[n:],
        "dest_transfer_in_before": transfer_in_before[n:],
        "same_amount_step_before": prior_sum(pair.ravel()),
        "round_amount": ((cents % 100_000) == 0).astype(np.int64),
    }


def load(t: Transactions, path: Path = CACHE) -> dict[str, np.ndarray]:
    """Signals for the full dataset, cached next to the data cache."""
    if Path(path).exists():
        with np.load(path) as z:
            if len(z["hour"]) == len(t) and set(z.files) == set(NAMES):
                return {k: z[k] for k in NAMES}
    s = compute(t)
    np.savez(path, **s)
    return s
