import numpy as np

from fraud import history as h
from fraud.ieee_data import TEXT, Ieee


def rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    names = ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "C1", "D1", "V1")
    num = np.c_[rng.integers(0, 12, n), rng.integers(0, 3, n), np.ones(n), np.ones(n), rng.integers(0, 6, n), np.ones(n), rng.normal(size=n), rng.normal(size=n), rng.integers(0, 50, n), rng.normal(size=n)].astype(np.float32)
    vocab = {k: ["", "x", "y"] for k in TEXT}
    return Ieee(ids=np.arange(n), dt=86400 + np.cumsum(rng.integers(1, 7200, n)), amount=np.round(rng.lognormal(4, 1, n), 2), is_fraud=rng.integers(0, 2, n).astype(np.int8), num=num, num_names=names,
                cat=rng.integers(0, 3, (n, len(TEXT))).astype(np.int16), cat_names=TEXT, vocab=vocab, has_identity=rng.integers(0, 2, n).astype(bool))


def identity(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = {k: np.array(["" for _ in range(n)], dtype=object) for k in (*h.ID_NUMERIC, *h.ID_TEXT, "DeviceInfo", "id_30", "id_31")}
    cols["DeviceInfo"] = np.array([f"d{k}" for k in rng.integers(0, 4, n)], dtype=object)
    cols["id_31"] = np.array(["chrome 60" if k else "Safari" for k in rng.integers(0, 2, n)], dtype=object)
    cols["id_30"] = np.array(["Windows 10" if k else "" for k in rng.integers(0, 2, n)], dtype=object)
    cols["id_01"] = np.array(["-5.0" if k else "" for k in rng.integers(0, 2, n)], dtype=object)
    return cols


def flat(b):
    return {f"{g}/{k}": v for g in ("history", "uid", "extra", "numeric", "one_hot") for k, v in b[g].items()}


def test_a_later_row_never_changes_an_earlier_rows_features():
    t, ident = rows(), identity()
    full = flat(h.build(t, ident))
    cut = 200
    late = rows()
    late.amount = late.amount.copy()
    late.amount[cut:] = late.amount[cut:] * 3 + 11
    late.num = late.num.copy()
    late.num[cut:, late.num_names.index("card1")] = 99
    late.dt = late.dt.copy()
    late.dt[cut:] = late.dt[cut:] + 777
    changed = flat(h.build(late, ident))
    for k in full:
        if k.startswith("one_hot/") or k.startswith("numeric/") or k.startswith("extra/"):
            continue  # one-hot vocabularies come from fit rows; extra and numeric read the row itself
        assert np.allclose(full[k][:cut], changed[k][:cut], equal_nan=True), k


def test_features_do_not_read_labels():
    t, ident = rows(), identity()
    a = flat(h.build(t, ident))
    t.is_fraud = 1 - t.is_fraud
    b = flat(h.build(t, ident))
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def test_counts_on_hand_made_rows():
    key = np.array([5, 7, 5, 5, 7])
    dt = np.array([100, 200, 300, 5000, 90000])
    amount = np.array([10.0, 20.0, 10.0, 40.0, 20.0])
    f = h.key_features("k", key, dt, amount)
    assert np.expm1(f["k_log_before"]).round().tolist() == [0, 0, 1, 2, 1]
    assert f["k_first"].tolist() == [1, 1, 0, 0, 0]
    assert np.expm1(f["k_log_since"]).round().tolist() == [0, 0, 200, 4700, 89800]
    assert np.expm1(f["k_log_1h"]).round().tolist() == [0, 0, 1, 0, 0]  # row 3: both earlier rows are over an hour ago
    assert np.expm1(f["k_log_1d"]).round().tolist() == [0, 0, 1, 2, 0]
    assert np.expm1(f["k_log_same_amt"]).round().tolist() == [0, 0, 1, 0, 1]
    assert f["k_amt_vs_mean"][0] == 0.0 and abs(f["k_amt_vs_mean"][3] - (np.log1p(40) - np.log1p(10))) < 1e-9


def test_distinct_counts_only_what_came_before():
    key = np.array([1, 1, 1, 1, 2])
    other = np.array([3, 3, 4, 5, 3])
    f = h.key_features("k", key, np.arange(5), np.ones(5), {"o": other})
    assert np.expm1(f["k_log_distinct_o"]).round().tolist() == [0, 1, 1, 2, 0]


def test_a_saved_cache_reads_back(tmp_path):
    t, ident = rows(), identity()
    b = h.build(t, ident)
    h.save(b, tmp_path / "h.npz")
    got = h.load(tmp_path / "h.npz")
    assert set(got["history"]) == set(b["history"]) and got["categories"]["browser"] == list(h.BROWSERS)
    k = next(iter(b["history"]))
    assert np.allclose(got["history"][k], b["history"][k], atol=1e-4)
