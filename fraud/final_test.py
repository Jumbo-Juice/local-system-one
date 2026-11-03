"""The rows of the IEEE-CIS test split nobody has used yet: 20 FINAL windows, and a pool for the phases before them.

    python -m fraud.final_test                  # the windows and what is left; no model, no labels

Reserved before any new work (owner, 2026-11-03). Phases 11, 12 and 14 used 60 windows of 1,000 rows, spread evenly over the test split, so 58,108
rows are left, in pieces. Many phases on one split invite overfitting by selection, and FINAL is the guard: no setup scores a FINAL row until the last phase, once.

Implementation choices, fixed in the first commit of this work:
- A slot is 1,000 contiguous untouched rows, the same shape as every earlier window. Each untouched run of L rows holds floor(L / 1000) slots, placed
  with one seeded shift of the unused rows. Observed: 30 slots in all, so contiguous windows alone would leave only 10 after FINAL.
- FINAL: the slots in time order, cut into 20 equal blocks, one seeded slot per block, so they cover the whole split as the earlier windows do.
- Later phases: every untouched row that is not FINAL, in arrival order, cut into chains of 1,000 rows. A chain skips the rows others used, so it is not contiguous: it keeps arrival
  order and the natural rate, and spans more time than 1,000 rows would. That is the price of having 38 windows and not 10. Chains are handed out in one seeded order by `phase_windows`, in the order of PHASE_WINDOWS;
  an earlier phase's windows don't change when a later one is added. A later phase states its count in its pre-registration, and the pool can run out; `phase_windows` then raises, and nothing is reused.
- Reading feature columns of a FINAL row to build history for later rows is not scoring it. No label of a FINAL row is read before the last phase.
"""

from __future__ import annotations

import numpy as np

from . import eval_ieee as ev, eval_ieee_amount as am, eval_ieee_full as fu, ieee_data

SEED = 20261105
FINAL_COUNT = 20
PHASE_WINDOWS: dict[str, int] = {"audit": 8, "rule": 10, "features": 10, "tabular": 10}  # phase name → windows, in the order they were pre-registered


def used_windows(n: int) -> list[ev.Win]:
    """Every test window an eval has scored: Phases 11, 12 and 14."""
    return ev.make_windows(n, "test", ev.TEST_WINDOWS) + fu.new_windows(n) + am.test_windows(n)


def untouched_runs(n: int, used: list[ev.Win] | None = None) -> list[tuple[int, int]]:
    """(first row, one past the last) of each run of test rows no window covers, in time order."""
    lo, hi = ieee_data.split_rows(n, "test")
    free = np.ones(hi - lo, bool)
    for w in used_windows(n) if used is None else used:
        free[w.rows[0] - lo:w.rows[-1] + 1 - lo] = False
    edges = np.flatnonzero(np.diff(np.r_[False, free, False].astype(np.int8)))
    return [(int(a) + lo, int(b) + lo) for a, b in zip(edges[::2], edges[1::2])]


def slots(n: int, size: int = ev.WINDOW_SIZE, seed: int = SEED) -> list[int]:
    """The start of every slot, in time order."""
    rng = np.random.default_rng([seed, 0])
    out = []
    for a, b in untouched_runs(n):
        k = (b - a) // size
        if k:
            shift = int(rng.integers(0, b - a - k * size + 1))
            out += [a + shift + j * size for j in range(k)]
    return out


def _final_starts(n: int, seed: int = SEED) -> list[int]:
    s = slots(n, seed=seed)
    if len(s) < FINAL_COUNT:
        raise ValueError(f"only {len(s)} slots for {FINAL_COUNT} final windows")
    rng = np.random.default_rng([seed, 1])
    edges = np.linspace(0, len(s), FINAL_COUNT + 1).astype(np.int64)
    return [s[int(rng.integers(edges[k], edges[k + 1]))] for k in range(FINAL_COUNT)]


def final_windows(n: int) -> list[ev.Win]:
    """The 20 FINAL windows. Only the last phase may score them."""
    return [ev.Win(f"final{k}", "test", np.arange(start, start + ev.WINDOW_SIZE), SEED) for k, start in enumerate(_final_starts(n))]


def pool(n: int, seed: int = SEED, size: int = ev.WINDOW_SIZE) -> list[np.ndarray]:
    """Chains of untouched rows that are not FINAL, in the seeded order later phases take them."""
    rows = np.concatenate([np.arange(a, b) for a, b in untouched_runs(n)])
    final = np.concatenate([np.arange(s, s + size) for s in _final_starts(n, seed)])
    rest = rows[~np.isin(rows, final)]
    chains = [rest[k * size:(k + 1) * size] for k in range(len(rest) // size)]
    return [chains[int(k)] for k in np.random.default_rng([seed, 2]).permutation(len(chains))]


def phase_windows(n: int, phase: str, table: dict[str, int] | None = None) -> list[ev.Win]:
    """This phase's own new windows, in time order, sharing no row with FINAL, an earlier phase or Phases 11-14."""
    table = PHASE_WINDOWS if table is None else table
    if phase not in table:
        raise KeyError(f"{phase} has no windows in PHASE_WINDOWS")
    names = list(table)
    first = sum(table[k] for k in names[:names.index(phase)])
    free = pool(n)
    if first + table[phase] > len(free):
        raise ValueError(f"{phase} wants {table[phase]} windows, {max(0, len(free) - first)} are left")
    chains = sorted(free[first:first + table[phase]], key=lambda c: int(c[0]))
    return [ev.Win(f"{phase}{k}", "test", c, SEED) for k, c in enumerate(chains)]


def guard(n: int, rows: np.ndarray) -> None:
    """Call before scoring test rows in a phase that is not the last. Raises if any row is FINAL."""
    final = np.concatenate([w.rows for w in final_windows(n)])
    if np.isin(rows, final).any():
        raise SystemExit("these rows include FINAL rows, which only the last phase may score")


def main() -> None:
    n = len(ieee_data.load())
    lo, hi = ieee_data.split_rows(n, "test")
    runs = untouched_runs(n)
    print(f"test rows {hi - lo:,}; used {sum(len(w.rows) for w in used_windows(n)):,}; untouched {sum(b - a for a, b in runs):,} in {len(runs)} runs; {len(slots(n))} slots")
    for w in final_windows(n):
        print(f"{w.id}: rows {w.rows[0]}–{w.rows[-1]}")
    print(f"{len(pool(n))} chains of {ev.WINDOW_SIZE:,} rows left for later phases; reserved: {PHASE_WINDOWS or 'none'}")


if __name__ == "__main__":
    main()
