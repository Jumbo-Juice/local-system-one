import json

import numpy as np
import pytest

from fraud import eval_ieee as ev, final_test as ft, ieee_data
from fraud.runs import RUNS

N = 590540


def rows_of(ws):
    return np.concatenate([w.rows for w in ws])


def test_used_windows_are_the_60_earlier_ones():
    used = ft.used_windows(N)
    assert len(used) == 60 and len(np.unique(rows_of(used))) == 60000


def test_untouched_is_every_test_row_no_window_covers():
    lo, hi = ieee_data.split_rows(N, "test")
    runs = ft.untouched_runs(N)
    assert hi - lo == 118108 and sum(b - a for a, b in runs) == 58108
    free = np.concatenate([np.arange(a, b) for a, b in runs])
    assert not np.isin(free, rows_of(ft.used_windows(N))).any()


def test_final_is_20_windows_of_1000_untouched_contiguous_test_rows():
    final = ft.final_windows(N)
    lo, hi = ieee_data.split_rows(N, "test")
    assert len(final) == 20 and all(len(w.rows) == 1000 and np.all(np.diff(w.rows) == 1) and lo <= w.rows[0] and w.rows[-1] < hi for w in final)
    rows = rows_of(final)
    assert len(np.unique(rows)) == 20000
    assert not np.isin(rows, rows_of(ft.used_windows(N))).any()


def test_final_is_fixed_by_the_seed():
    a = [int(w.rows[0]) for w in ft.final_windows(N)]
    assert a == [int(w.rows[0]) for w in ft.final_windows(N)] and a == sorted(a)
    assert a != ft._final_starts(N, seed=ft.SEED + 1)


def test_final_covers_the_whole_split():
    starts = np.array([w.rows[0] for w in ft.final_windows(N)])
    lo, hi = ieee_data.split_rows(N, "test")
    assert starts[:5].max() < lo + (hi - lo) / 2 < starts[-5:].min()


def test_enough_rows_stay_for_later_phases():
    assert len(ft.pool(N)) == 38 and all(len(c) == 1000 for c in ft.pool(N))


def test_pool_shares_no_row_with_final_or_earlier_windows_or_itself_and_keeps_arrival_order():
    rows = np.concatenate(ft.pool(N))
    assert len(np.unique(rows)) == len(rows)
    assert not np.isin(rows, rows_of(ft.final_windows(N) + ft.used_windows(N))).any()
    assert all(np.all(np.diff(c) > 0) for c in ft.pool(N))


def test_a_phase_keeps_its_windows_when_a_later_phase_is_added():
    a = ft.phase_windows(N, "one", {"one": 3})
    b = ft.phase_windows(N, "one", {"one": 3, "two": 4})
    two = ft.phase_windows(N, "two", {"one": 3, "two": 4})
    assert [w.rows[0] for w in a] == [w.rows[0] for w in b]
    assert not np.isin(rows_of(two), rows_of(a)).any()
    assert [w.id for w in a] == ["one0", "one1", "one2"]


def test_the_pool_can_run_out_and_nothing_is_reused():
    with pytest.raises(ValueError):
        ft.phase_windows(N, "big", {"big": len(ft.pool(N)) + 1})
    with pytest.raises(KeyError):
        ft.phase_windows(N, "unknown", {})


def test_guard_stops_a_phase_from_scoring_final_rows():
    ft.guard(N, np.concatenate(ft.pool(N)))
    with pytest.raises(SystemExit):
        ft.guard(N, ft.final_windows(N)[3].rows[:2])


def test_no_recorded_trace_has_read_a_final_row():
    """Where the traces are on disk: the windows above really are all the test rows any eval has written."""
    traces = [p for k in ("20261102-183431", "20261103-142622", "20261103-164830") for p in RUNS.glob(f"{k}_eval_ieee-*.jsonl")]
    if not traces:
        pytest.skip("no IEEE traces on this machine")
    final = set(rows_of(ft.final_windows(N)).tolist())
    used = set(rows_of(ft.used_windows(N)).tolist())
    for p in traces:
        with p.open(encoding="utf-8") as f:
            header = json.loads(f.readline())
        assert header["window"]["split"] == "test"
        a, b = header["window"]["steps"]
        assert not final.intersection(range(a, b + 1))
        assert used.issuperset(range(a, b + 1))
    assert ev.WINDOW_SIZE == 1000
