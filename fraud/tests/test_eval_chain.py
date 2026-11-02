import numpy as np

from fraud import eval_chain as ec, rules
from fraud.data import TYPE_ID, Transactions
from fraud.tests.test_brain_capture import S, T

TR, CO, PAY = TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"], TYPE_ID["PAYMENT"]


def table(rows):
    """(step, type, amount, fraud) rows."""
    a = np.array(rows, dtype=float)
    n = len(a)
    return Transactions(step=a[:, 0].astype(np.int16), type=a[:, 1].astype(np.int8), amount=a[:, 2],
                        orig=np.arange(n, dtype=np.int32), dest=np.arange(n, dtype=np.int32) + 100,
                        dest_merchant=np.zeros(n, bool), is_fraud=a[:, 3].astype(np.int8), flagged=np.zeros(n, np.int8))


def test_rule_lines_match_the_rule_table():
    assert ec.rule_lines(T, S).tolist() == [rules.rule_for(T, S, i) for i in range(len(T))]


def test_prior_transfer_is_earlier_same_hour_same_amount_only():
    t = table([(1, CO, 5, 0),   # the transfer comes later
               (1, TR, 5, 0),
               (1, CO, 5, 1),   # yes
               (1, CO, 6, 0),   # another amount
               (2, CO, 5, 0)])  # another hour
    assert ec.prior_transfer(t).tolist() == [False, False, True, False, False]


def test_chains_pair_a_transfer_with_its_later_cash_out_once():
    t = table([(1, CO, 9, 1),   # a cash-out before any transfer: its own chain
               (1, TR, 5, 1),
               (1, PAY, 5, 0),
               (1, CO, 5, 1),   # pairs with the transfer
               (1, CO, 5, 1),   # that transfer is taken: its own chain
               (2, CO, 7, 1)])
    ids = ec.chains(t, np.arange(len(t)))
    assert ids[1] == ids[3] and ids[2] == -1
    assert len({ids[0], ids[1], ids[4], ids[5]}) == 4


def test_a_chain_counts_once_and_a_review_never_makes_a_false_positive():
    y = np.array([True, True, False, False])
    chain = np.array([0, 0, -1, -1])
    r = ec.score(y, chain, np.array([False, True, True, False]), np.zeros(4, bool))
    assert r["chain"]["tp"] == 1 and r["chain"]["fn"] == 0 and r["chain"]["fp"] == 1
    assert r["row"]["tp"] == 1 and r["row"]["fn"] == 1
    h = ec.score(y, chain, np.zeros(4, bool), np.array([True, True, True, False]))
    assert h["row"]["tp"] == 2 and h["row"]["fp"] == 0 and h["reviews"] == 3


def test_setups_differ_only_where_the_rules_say():
    lines = np.array([1, 1, 2, 3, 4, 5])
    prior = np.array([True, False, False, False, False, False])
    d, r = ec.actions("human", lines, prior)
    assert d.tolist() == [True, True, True, False, False, False] and r.tolist() == [False, False, False, True, True, False]
    d, r = ec.actions("tight", lines, prior)
    assert d.tolist() == [True, False, True, False, False, False] and not r.any()
    assert ec.actions("decline", lines, prior)[0].tolist() == [True, True, True, True, True, False]


def test_bootstrap_over_all_hours_reproduces_the_chain_f1():
    step = np.array([1, 1, 2, 2])
    y = np.array([True, True, True, False])
    chain = np.array([0, 0, 1, -1])
    declined = np.array([False, True, False, True])
    reviewed = np.zeros(4, bool)
    whole = ec.chain_f1_by_hours(step, y, chain, declined, reviewed, [np.array([0, 1])])[0]
    assert whole == ec.score(y, chain, declined, reviewed)["chain"]["f1"]


def test_diagnose_splits_wrong_declines_by_rule_line(monkeypatch):
    t = table([(1, TR, 5, 0), (1, CO, 5, 0), (1, TR, 10_000_000, 0)])
    monkeypatch.setattr(ec, "rule_lines", lambda t, s: np.array([3, 1, 2]))
    d = ec.diagnose(t, {}, [1])
    assert d["wrong_declines"] == 2 and d["of_exactly_10M"] == 1 and d["top_hours"] == [[1, 2]]
    assert list(d["by_line"].values()) == [1, 1]

