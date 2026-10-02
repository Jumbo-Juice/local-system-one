import numpy as np

from fraud import rules, signals
from fraud.data import TYPE_ID, Transactions


def tx(rows):
    s, t, a, o, d = zip(*rows)
    n = len(rows)
    return Transactions(np.array(s, np.int16), np.array([TYPE_ID[x] for x in t], np.int8), np.array(a, float),
                        np.array(o, np.int32), np.array(d, np.int32), np.zeros(n, bool), np.zeros(n, np.int8),
                        np.zeros(n, np.int8))


T = tx([
    (1, "PAYMENT", 10.0, 1, 2),        # 0 fraud-free
    (1, "TRANSFER", 1234.5, 3, 4),     # 1 new receiver -> review
    (1, "CASH_OUT", 1234.5, 5, 6),     # 2 same amount this hour -> decline
    (1, "TRANSFER", 77.0, 7, 4),       # 3 known receiver -> approve
    (1, "TRANSFER", 5000.0, 8, 9),     # 4 round -> decline
    (1, "CASH_OUT", 50.0, 10, 4),      # 5 otherwise -> approve
])
S = signals.compute(T)


def test_rules_only_table():
    got = [rules.RulesOnly().decide(T, S, i)["action"] for i in range(len(T))]
    assert got == ["approve", "review", "decline", "approve", "decline", "approve"]


def test_filter_only_takes_fraud_free_types():
    f = rules.TypeFilter()
    assert f.decide(T, S, 0)["action"] == "approve"
    assert all(f.decide(T, S, i) is None for i in range(1, len(T)))
