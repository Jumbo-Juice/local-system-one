import numpy as np

from fraud import eval_ieee as ev, feature_sets as fs, final_test as ft, history as h

from .test_history import identity, rows

N = 590540


def make():
    t, ident = rows(), identity()
    return t, h.build(t, ident)


def test_no_parts_is_the_column_audits_features():
    t, cache = make()
    idx = np.arange(40)
    assert np.array_equal(fs.FeatureSet((), "", cache)(t, idx), ev.features_all(t, idx))


def test_steady_leaves_out_lifetime_counts():
    names = ["card_log_before", "card_first", "card_log_1d", "card_log_since", "card_amt_z", "card_log_distinct_addr", "card_log_same_amt"]
    assert fs.steady_names(names) == ["card_log_1d", "card_log_since", "card_amt_z"]


def test_parts_add_columns_and_keep_rows():
    t, cache = make()
    idx = np.arange(40)
    base = fs.FeatureSet((), "", cache)(t, idx).shape[1]
    steady = fs.FeatureSet(("steady",), "", cache)(t, idx)
    both = fs.FeatureSet(("steady", "extra", "identity", "inter", "uid"), "", cache)(t, idx)
    assert steady.shape == (40, base + len(fs.steady_names(cache["history"])))
    assert both.shape[0] == 40 and both.shape[1] > steady.shape[1] and np.isfinite(both).all()


def test_a_scored_row_gets_the_same_features_alone_or_in_a_block():
    t, cache = make()
    f = fs.FeatureSet(("steady", "extra", "identity", "inter"), "", cache)
    assert np.array_equal(f(t, np.array([7])), f(t, np.arange(20))[[7]])


def test_the_feature_windows_are_new():
    new = ft.phase_windows(N, "features")
    assert len(new) == 10
    rows_new = np.concatenate([w.rows for w in new])
    ft.guard(N, rows_new)
    other = ft.phase_windows(N, "audit") + ft.phase_windows(N, "rule") + ft.used_windows(N)
    assert not np.isin(rows_new, np.concatenate([w.rows for w in other])).any()
