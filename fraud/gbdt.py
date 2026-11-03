"""Gradient-boosted trees in numpy only, for the logistic loss (Phase 20). No new package.

Histogram boosting, written from the standard method (Friedman 2001; Ke et al. 2017, LightGBM: Documented from memory, not checked against the papers):
features are cut into at most BINS quantile bins from the fit rows, and each tree splits on the bins' summed gradients and hessians, level by level. Missing values get their own bin (0), so a split can send them either way by where bin 0 falls.
Rows and columns are subsampled per tree with a seeded generator. Deterministic for a given seed and thread count of 1 or more, as sums don't depend on the order threads finish in.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

BINS = 64


def quantile_edges(x: np.ndarray, bins: int = BINS) -> list[np.ndarray]:
    """One sorted edge list per column, from the non-missing values. Always in float32, as to_bins reads them."""
    x = np.asarray(x, np.float32)
    out = []
    grid = np.linspace(0, 1, bins)[1:-1]
    for j in range(x.shape[1]):
        v = x[:, j]
        v = v[~np.isnan(v)]
        out.append(np.unique(np.quantile(v, grid)) if len(v) else np.array([]))
    return out


def to_bins(x: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """uint8 bins: 0 is missing, 1 and up are by value. Column-major, as trees read a column at a time.
    Always in float32: a value on an edge must land in the same bin whichever float type the caller holds. Without this, scoring float64 against edges from float32 moved every row (Phase 20: AUC 0.93 fell to 0.89)."""
    x = np.asarray(x, np.float32)
    out = np.zeros(x.shape, np.uint8, order="F")
    for j, e in enumerate(edges):
        v = x[:, j]
        b = np.searchsorted(e, v, side="right") + 1
        b[np.isnan(v)] = 0
        out[:, j] = b
    return out


@dataclass
class Params:
    trees: int = 400
    depth: int = 6
    lr: float = 0.05
    min_hessian: float = 20.0
    l2: float = 10.0
    row_sample: float = 0.5
    col_sample: float = 0.3
    seed: int = 0
    threads: int = 16


class Tree:
    """Arrays in level order: feature, bin threshold (go left when bin <= it), left, right; a leaf has feature -1 and its value in value."""

    def __init__(self, feature, threshold, left, right, value):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value

    def predict(self, b: np.ndarray) -> np.ndarray:
        node = np.zeros(b.shape[0], np.int32)
        while True:
            inner = self.feature[node] >= 0
            if not inner.any():
                return self.value[node]
            at = np.flatnonzero(inner)
            n = node[at]
            go_left = b[at, self.feature[n]] <= self.threshold[n]
            node[at] = np.where(go_left, self.left[n], self.right[n])


def _column_hist(args) -> tuple[np.ndarray, np.ndarray]:
    col, nodes, width, g, h = args
    key = nodes.astype(np.int64) * BINS + col
    return np.bincount(key, g, width), np.bincount(key, h, width)


def fit_tree(b: np.ndarray, g: np.ndarray, h: np.ndarray, rows: np.ndarray, cols: np.ndarray, p: Params, pool: ThreadPoolExecutor) -> Tree:
    """One tree, grown level by level on these rows and columns. Each level costs one histogram per column over all of the level's rows."""
    feature, threshold, left, right, value = [-1], [0], [-1], [-1], [0.0]
    leaves = np.zeros(len(rows), np.int32)  # the current level's node position per row
    level = [0]  # node ids at this level, by position
    gs, hs = g[rows], h[rows]
    for _ in range(p.depth):
        width = len(level) * BINS
        columns = [b[rows, j] for j in cols]
        hists = list(pool.map(_column_hist, [(c, leaves, width, gs, hs) for c in columns]))
        G = np.stack([x[0] for x in hists]).reshape(len(cols), len(level), BINS)
        H = np.stack([x[1] for x in hists]).reshape(len(cols), len(level), BINS)
        CG, CH = np.cumsum(G, 2), np.cumsum(H, 2)
        tg, th = CG[0, :, -1], CH[0, :, -1]
        rg, rh = tg[None, :, None] - CG, th[None, :, None] - CH
        parent = tg ** 2 / (th + p.l2)
        gain = CG ** 2 / (CH + p.l2) + rg ** 2 / (rh + p.l2) - parent[None, :, None]
        gain[(CH < p.min_hessian) | (rh < p.min_hessian)] = -np.inf
        gain[:, :, -1] = -np.inf
        flat = gain.transpose(1, 0, 2).reshape(len(level), -1)
        pick = flat.argmax(1)
        ok = flat[np.arange(len(level)), pick] > 1e-9
        next_level, map_left, map_right = [], np.full(len(level), -1, np.int32), np.full(len(level), -1, np.int32)
        for k, node in enumerate(level):
            value[node] = float(-tg[k] / (th[k] + p.l2))
            if not ok[k]:
                continue
            c, t = np.unravel_index(pick[k], (len(cols), BINS))
            feature[node], threshold[node] = int(cols[c]), int(t)
            lv, rv = len(feature), len(feature) + 1
            for _n in range(2):
                feature.append(-1), threshold.append(0), left.append(-1), right.append(-1), value.append(0.0)
            left[node], right[node] = lv, rv
            map_left[k], map_right[k] = len(next_level), len(next_level) + 1
            next_level += [lv, rv]
        if not next_level:
            break
        feat_k = np.array([feature[n] for n in level])
        thr_k = np.array([threshold[n] for n in level])
        split = feat_k[leaves] >= 0
        go_left = np.zeros(len(rows), bool)
        for k in np.flatnonzero(feat_k >= 0):
            here = leaves == k
            go_left[here] = b[rows[here], feat_k[k]] <= thr_k[k]
        new = np.where(go_left, map_left[leaves], map_right[leaves])
        keep = split
        rows, gs, hs, leaves = rows[keep], gs[keep], hs[keep], new[keep]
        level = next_level
    if rows.size:  # the last level's values
        G = np.bincount(leaves, gs, len(level))
        H = np.bincount(leaves, hs, len(level))
        for k, node in enumerate(level):
            value[node] = float(-G[k] / (H[k] + p.l2))
    return Tree(*(np.array(x, t) for x, t in zip((feature, threshold, left, right, value), (np.int32, np.int32, np.int32, np.int32, np.float64))))


class GBDT:
    def __init__(self, params: Params = Params()):
        self.p = params
        self.trees: list[Tree] = []
        self.base = 0.0
        self.edges: list[np.ndarray] = []
        self.history: list[dict] = []

    def fit(self, x: np.ndarray, y: np.ndarray, valid: tuple[np.ndarray, np.ndarray] | None = None, every: int = 50, log=None) -> "GBDT":
        p = self.p
        rng = np.random.default_rng(p.seed)
        t0 = time.perf_counter()
        self.edges = quantile_edges(x)
        b = to_bins(x, self.edges)
        if log:
            log(f"binned {x.shape[1]} columns in {time.perf_counter() - t0:.0f} s")
        rate = float(y.mean())
        self.base = float(np.log(rate / (1 - rate)))
        f = np.full(len(y), self.base)
        fv = None
        if valid is not None:
            bv = to_bins(valid[0], self.edges)
            fv = np.full(len(valid[1]), self.base)
        with ThreadPoolExecutor(p.threads) as pool:
            for k in range(p.trees):
                prob = 1 / (1 + np.exp(-f))
                g, h = prob - y, prob * (1 - prob)
                rows = np.sort(rng.choice(len(y), int(p.row_sample * len(y)), replace=False))
                cols = np.sort(rng.choice(x.shape[1], max(1, int(p.col_sample * x.shape[1])), replace=False))
                tree = fit_tree(b, g, h, rows, cols, p, pool)
                self.trees.append(tree)
                f += p.lr * tree.predict(b)
                if fv is not None:
                    fv += p.lr * tree.predict(bv)
                if (k + 1) % every == 0 or k == p.trees - 1:
                    row = {"trees": k + 1, "s": round(time.perf_counter() - t0, 1), "train_logloss": float(np.mean(np.logaddexp(0, -f * (2 * y - 1))))}
                    if fv is not None:
                        row["valid_logloss"] = float(np.mean(np.logaddexp(0, -fv * (2 * valid[1] - 1))))
                    self.history.append(row)
                    if log:
                        log(json.dumps(row))
        return self

    def raw(self, x: np.ndarray, trees: int | None = None) -> np.ndarray:
        b = to_bins(x, self.edges)
        f = np.full(x.shape[0], self.base)
        for tree in self.trees[:trees]:
            f += self.p.lr * tree.predict(b)
        return f

    def predict(self, x: np.ndarray, trees: int | None = None) -> np.ndarray:
        return 1 / (1 + np.exp(-self.raw(x, trees)))
