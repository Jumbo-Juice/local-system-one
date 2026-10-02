import numpy as np

from fraud import signals
from fraud.data import TYPE_ID, Transactions


def tx(rows):
    """rows: (step, type, amount, orig, dest)"""
    s, t, a, o, d = zip(*rows)
    n = len(rows)
    return Transactions(step=np.array(s, np.int16), type=np.array([TYPE_ID[x] for x in t], np.int8),
                        amount=np.array(a, float), orig=np.array(o, np.int32), dest=np.array(d, np.int32),
                        dest_merchant=np.zeros(n, bool), is_fraud=np.zeros(n, np.int8), flagged=np.zeros(n, np.int8))


def test_prior_sum_counts_only_earlier_rows():
    keys = np.array([5, 7, 5, 5, 7])
    assert signals.prior_sum(keys).tolist() == [0, 0, 1, 2, 1]
    w = np.array([1, 1, 0, 1, 1])
    assert signals.prior_sum(keys, w).tolist() == [0, 0, 1, 1, 1]


def test_account_history_uses_earlier_rows_only():
    t = tx([
        (1, "TRANSFER", 100.0, 1, 2),   # 0: 1 -> 2
        (1, "CASH_OUT", 100.0, 2, 9),   # 1: 2 -> 9 (2 received earlier, same amount same step)
        (1, "PAYMENT", 5.0, 1, 3),      # 2: 1 -> 3
        (2, "TRANSFER", 50.0, 4, 2),    # 3: 4 -> 2
        (2, "CASH_IN", 50.0, 2, 1),     # 4: 2 -> 1
    ])
    s = signals.compute(t)
    assert s["orig_out_before"].tolist() == [0, 0, 1, 0, 1]
    assert s["orig_in_before"].tolist() == [0, 1, 0, 0, 2]
    assert s["dest_in_before"].tolist() == [0, 0, 0, 1, 0]
    assert s["dest_out_before"].tolist() == [0, 0, 0, 1, 2]
    assert s["dest_transfer_in_before"].tolist() == [0, 0, 0, 1, 0]
    assert s["same_amount_step_before"].tolist() == [0, 1, 0, 0, 1]
    assert s["hour"].tolist() == [1, 1, 1, 2, 2]


def test_no_look_ahead():
    """Changing or dropping later rows never changes an earlier row's signals."""
    rng = np.random.default_rng(0)
    rows = [(int(1 + i // 7), ["TRANSFER", "CASH_OUT", "PAYMENT"][rng.integers(3)], float(rng.integers(1, 5) * 100),
             int(rng.integers(0, 6)), int(rng.integers(0, 6))) for i in range(60)]
    full = signals.compute(tx(rows))
    cut = signals.compute(tx(rows[:30]))
    for k in full:
        assert full[k][:30].tolist() == cut[k].tolist(), k


def test_round_amount():
    t = tx([(1, "TRANSFER", 2000.0, 1, 2), (1, "TRANSFER", 2000.5, 1, 2), (1, "TRANSFER", 1999.0, 1, 2)])
    assert signals.compute(t)["round_amount"].tolist() == [1, 0, 0]
