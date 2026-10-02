import numpy as np
import pytest

from fraud import windows
from fraud.data import TYPE_ID, Transactions


def synthetic(steps=300, per_step=40, seed=0):
    """Legit rows plus one TRANSFER→CASH_OUT fraud pair (same amount) every hour."""
    rng = np.random.default_rng(seed)
    step, typ, amount, fraud = [], [], [], []
    for s in range(1, steps + 1):
        for _ in range(per_step):
            step.append(s); typ.append(TYPE_ID["PAYMENT"]); amount.append(float(rng.integers(1, 999))); fraud.append(0)
        a = float(rng.integers(1000, 9999))
        step += [s, s]; typ += [TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]]; amount += [a, a]; fraud += [1, 1]
    n = len(step)
    return Transactions(np.array(step, np.int16), np.array(typ, np.int8), np.array(amount), np.arange(n, dtype=np.int32),
                        np.arange(n, dtype=np.int32) + n, np.zeros(n, bool), np.array(fraud, np.int8), np.zeros(n, np.int8))


@pytest.fixture(scope="module")
def t():
    return synthetic()


def test_window_size_share_order_and_split(t):
    w = windows.make(t, "dev", seed=3, size=200, fraud_target=10)
    assert len(w.rows) == 200 and np.all(np.diff(w.rows) > 0)
    assert w.fraud == int(t.is_fraud[w.rows].sum()) == 10
    lo, hi = windows.TRAIN
    assert lo <= t.step[w.rows].min() and t.step[w.rows].max() <= hi


def test_pairs_stay_together(t):
    w = windows.make(t, "dev", seed=5, size=200, fraud_target=10)
    fr = w.rows[t.is_fraud[w.rows] == 1]
    for r in fr:
        if t.type[r] == TYPE_ID["TRANSFER"]:
            assert r + 1 in fr
        else:
            assert r - 1 in fr


def test_seeded_and_distinct(t):
    a = windows.make(t, "dev", seed=1, size=200, fraud_target=10)
    b = windows.make(t, "dev", seed=1, size=200, fraud_target=10)
    c = windows.make(t, "dev", seed=2, size=200, fraud_target=10)
    assert a.rows.tolist() == b.rows.tolist()
    assert a.rows.tolist() != c.rows.tolist()


def test_weights_restore_span_counts(t):
    w = windows.make(t, "dev", seed=4, size=200, fraud_target=10)
    f = t.is_fraud[w.rows].astype(bool)
    assert w.weight[f].sum() == pytest.approx(w.span_fraud)
    assert w.weight[~f].sum() == pytest.approx(w.span_rows - w.span_fraud)
    assert w.header()["natural_fraud_rate"] == pytest.approx(w.span_fraud / w.span_rows)
