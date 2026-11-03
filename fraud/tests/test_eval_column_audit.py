import numpy as np

from fraud import eval_column_audit as au, eval_ieee as ev, final_test as ft, ieee_data, tabular as tb
from fraud.ieee_data import TEXT, Ieee

N = 590540


def table(n=400):
    rng = np.random.default_rng(0)
    names = ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "C1", "C2", "D1", "V1", "V2")
    vocab = {k: ["", "x", "y"] for k in TEXT}
    num = rng.normal(size=(n, len(names))).astype(np.float32)
    num[::7, names.index("D1")] = np.nan
    return Ieee(ids=np.arange(n), dt=86400 + np.arange(n) * 3600, amount=np.linspace(10, 100, n), is_fraud=(np.arange(n) % 5 == 0).astype(np.int8), num=num, num_names=names,
                cat=rng.integers(0, 3, (n, len(TEXT))).astype(np.int16), cat_names=TEXT, vocab=vocab, has_identity=np.zeros(n, bool))


def test_dropping_nothing_is_exactly_features_all():
    t = table()
    idx = np.arange(50)
    assert np.array_equal(au.features_without("")(t, idx), ev.features_all(t, idx))


def test_dropping_a_group_removes_its_columns_and_flags_and_nothing_else():
    t = table()
    idx = np.arange(50)
    width = lambda d: au.features_without(d)(t, idx).shape[1]
    full = width("")
    assert width("V") == full - 2  # two V columns, which have no flags
    assert width("C") == full - 4  # two C columns and their flags
    assert width("D") == full - 2  # one D column and its flag
    assert width("CDV") == full - 8
    assert width("M") == full - 9 * 2  # nine M categories, two one-hot columns each


def test_the_audit_windows_are_its_own_and_not_final():
    t = ft.phase_windows(N, "audit")
    assert len(t) == 8 and all(len(w.rows) == 1000 for w in t)
    rows = np.concatenate([w.rows for w in t])
    ft.guard(N, rows)
    assert not np.isin(rows, np.concatenate([w.rows for w in ft.used_windows(N)])).any()


def test_heldout_dev_stays_in_dev_outside_the_30_windows():
    held = tb.heldout_dev(N)
    lo, hi = ieee_data.split_rows(N, "dev")
    rows = np.concatenate([w.rows for w in held])
    assert len(held) == 88 and lo <= rows.min() and rows.max() < hi and len(np.unique(rows)) == len(rows)
    assert not np.isin(rows, np.concatenate([w.rows for w in tb.dev30(N)])).any()


def test_card_counts_are_earlier_and_later_rows_of_the_same_key():
    key = np.array([[1.0], [2.0], [1.0], [1.0], [2.0]])
    earlier, later = au.card_counts(key)
    assert earlier.tolist() == [0, 0, 1, 2, 1] and later.tolist() == [2, 1, 1, 0, 0]
