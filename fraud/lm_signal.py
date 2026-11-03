"""The tabular risk score the language model reads as one signal (Phase 21, owner: "yes, as a signal") → fraud/data/ieee-cis/signal.npz (git-ignored).

    python -m fraud.lm_signal --build    # fit rows out of fold, dev rows from Phase 20's tab-all model. No test row.
    python -m fraud.lm_signal --final    # adds the FINAL rows. Only the last phase runs this.

The score is fraud/gbdt.py's trees on all columns at 400 trees. On a fit row it comes from a model that did not see that row: the fit rows are cut into FOLDS blocks in time order, and each block is scored by a model fitted on the others, so the language model doesn't train on a score that already knows its row's label.
On dev and test rows it comes from Phase 20's tab-all, the model fitted on all fit rows, at the same 400 trees, so the two are close in strength. tab-all has a little more data. A fold's model does see rows later than the block it scores; that happens only in training. No model ever sees a dev or test label here.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from . import eval_tabular as et, feature_sets as fs, final_test as ft, ieee_data
from .gbdt import GBDT, Params

CACHE = ieee_data.RAW / "signal.npz"
FOLDS, TREES = 4, 400
PARAMS = Params(trees=TREES, depth=6, lr=0.05)


def folds(n_fit: int, k: int = FOLDS) -> list[np.ndarray]:
    return np.array_split(np.arange(n_fit), k)


def out_of_fold(x: np.ndarray, y: np.ndarray, blocks: list[np.ndarray], params: Params, log=print) -> np.ndarray:
    """A score per row, each from a model fitted without its block."""
    q = np.full(len(y), np.nan)
    for k, block in enumerate(blocks):
        t0 = time.perf_counter()
        rest = np.setdiff1d(np.arange(len(y)), block)
        q[block] = GBDT(params).fit(x[rest], y[rest]).predict(x[block])
        log(f"fold {k + 1} of {len(blocks)} in {time.perf_counter() - t0:.0f} s")
    return q


def load_model() -> GBDT:
    path = et.MODELS / "tabular_tab-all.pkl"
    if not path.exists():
        raise SystemExit(f"no {path}; run fraud.eval_tabular first")
    return pickle.loads(path.read_bytes())


def build(log=print) -> None:
    t = ieee_data.load()
    a, b = ieee_data.split_bounds(len(t))
    f = fs.FeatureSet((), "", None)
    q = np.full(len(t), np.nan)
    q[:a] = out_of_fold(f(t, np.arange(a)).astype(np.float32), t.is_fraud[:a].astype(np.float64), folds(a), PARAMS, log)
    q[a:b] = load_model().predict(f(t, np.arange(a, b)).astype(np.float32), TREES)
    np.savez(CACHE, q=q.astype(np.float32))
    log(f"wrote {CACHE}: fit and dev rows")


def add_final(log=print) -> None:
    """The last phase only: the FINAL rows' scores."""
    t = ieee_data.load()
    with np.load(CACHE) as z:
        q = z["q"]
    rows = np.concatenate([w.rows for w in ft.final_windows(len(t))])
    q[rows] = load_model().predict(fs.FeatureSet((), "", None)(t, rows).astype(np.float32), TREES)
    np.savez(CACHE, q=q)
    log(f"added {len(rows):,} FINAL rows to {CACHE}")


def load(path: Path = CACHE) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"no cache at {path}; run: python -m fraud.lm_signal --build")
    with np.load(path) as z:
        return z["q"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.build:
        build(log)
    else:
        add_final(log)


if __name__ == "__main__":
    main()
