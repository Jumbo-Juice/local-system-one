import numpy as np

from fraud import costs, eval_ieee as ev, eval_ieee_amount as am, eval_ieee_full as fu, ieee_data
from fraud.ieee_data import TEXT, Ieee

C = ev.COSTS
N = 590540


def table(n=2400):
    names = ("card1", "card2", "addr1", "addr2", "C1", "C4", "C8", "C13", "D1", "V1")
    vocab = {k: ["", "x", "y"] for k in TEXT}
    return Ieee(ids=np.arange(n), dt=86400 + np.arange(n) * 3600, amount=np.linspace(10, 100, n), is_fraud=(np.arange(n) % 5 == 0).astype(np.int8),
                num=np.ones((n, len(names)), np.float32), num_names=names, cat=np.ones((n, len(TEXT)), np.int16), cat_names=TEXT, vocab=vocab, has_identity=np.zeros(n, bool))


def test_new_test_windows_share_no_row_with_phase_11_or_12_and_stay_in_test():
    new = am.test_windows(N)
    old = ev.make_windows(N, "test", ev.TEST_WINDOWS) + fu.new_windows(N)
    lo, hi = ieee_data.split_rows(N, "test")
    assert len(new) == 20 and all(lo <= w.rows[0] and w.rows[-1] < hi and len(w.rows) == ev.WINDOW_SIZE for w in new)
    rows = np.concatenate([w.rows for w in new + old])
    assert len(np.unique(rows)) == len(rows)
    assert [w.rows[0] for w in new] == [w.rows[0] for w in am.test_windows(N)]
    assert [w.id for w in new[:2]] == ["amt0", "amt1"]


def test_extra_dev_windows_stay_in_dev_and_share_no_row_with_phase_11s():
    new, old = am.extra_dev(N), am.old_dev(N)
    lo, hi = ieee_data.split_rows(N, "dev")
    assert len(new) == 20 and all(lo <= w.rows[0] and w.rows[-1] < hi for w in new)
    rows = np.concatenate([w.rows for w in new + old])
    assert len(np.unique(rows)) == len(rows)


def test_a_small_payment_is_never_worth_a_review():
    wanted, _ = am.decide(np.array([0.5]), np.array([5.0]), 0, C)
    assert wanted == ["approve"] or wanted == ["decline"]
    assert "review" not in am.decide(np.linspace(0, 1, 101), np.full(101, 15.0), 0, C)[0]


def test_a_large_uncertain_payment_is_reviewed():
    assert am.decide(np.array([0.3]), np.array([5000.0]), 0, C) == (["review"], ["decline"])


def test_review_price_lambda_raises_the_bar():
    p, a = np.array([0.05]), np.array([400.0])  # approve costs 20, decline 38, review 10
    assert am.decide(p, a, 0, C)[0] == ["review"]
    assert am.decide(p, a, 20, C)[0] == ["approve"]


def test_expected_costs_choose_approve_or_decline_and_tie_to_approve():
    assert am.decide(np.array([0.001]), np.array([50.0]), 1000, C)[0] == ["approve"]
    assert am.decide(np.array([0.9]), np.array([50.0]), 1000, C)[0] == ["decline"]
    assert am.decide(np.array([0.5]), np.array([1.0]), 1000, C) == (["approve"], ["approve"])  # both cost 0.5


def test_friction_minimum_counts_for_small_payments():
    # 2 USD: friction is the 1 USD minimum, so declining needs p * 2 > (1 - p) * 1.
    assert am.decide(np.array([0.2]), np.array([2.0]), 0, C)[0] == ["approve"]
    assert am.decide(np.array([0.5]), np.array([2.0]), 0, C)[0] == ["decline"]


def test_pick_lambda_takes_the_lowest_and_first_on_a_tie():
    p, y, a = [np.array([0.5] * 3)], [np.array([1, 1, 0])], [np.array([5.0] * 3)]
    assert am.pick_lambda(p, y, a, C)[0] == am.LAMBDAS[0]  # no reviews at any lambda: all tie
    # Budget of 1 review in 20 rows. The first row would save 6 by a review, the second 240. A price of 10 on reviews saves the budget for the second.
    p = [np.array([0.2, 0.5] + [0.0] * 18)]
    a = [np.array([200.0, 5000.0] + [1.0] * 18)]
    y = [np.zeros(20, int)]
    lam, grid = am.pick_lambda(p, y, a, C)
    assert lam == 10 and dict(grid)[10] == 30.0 and dict(grid)[0] == 510.0


def test_trace_records_a_decided_rule_and_keeps_thresholds_otherwise():
    t = table()
    w = ev.Win("amt0", "test", np.arange(100, 120))
    outs = [{"by": "logreg", "q": 0.5, "ms": 0.0} for _ in w.rows]
    plain = ev.trace(t, w, "s", outs, (0.4, 0.9), {}, "e")
    assert plain[1]["wanted"] == "review" and plain[1]["fallback"] == "approve"
    wanted, fallback = ["review"] * 20, ["decline"] * 20
    recs = ev.trace(t, w, "s", outs, None, {}, "e", (wanted, fallback))
    assert recs[0]["thresholds"] is None and recs[1]["action"] == "review"
    assert recs[5]["action"] == "decline" and recs[5]["fallback"] == "decline"


def test_features_shape_and_logit_limits():
    d = {"lr": [np.array([0.0, 1.0])], "ft": [np.array([0.5, 0.5])]}
    assert np.isfinite(am.logit(d["lr"][0])).all()
    assert am.features(d, am.BLEND)[0].shape == (2, 2)
    assert am.features(d, am.LR_AMT)[0].shape == (2, 1) and am.features(d, am.FT_AMT)[0][0, 0] == 0.0


def test_costs_are_phase_11s_and_setups_are_named():
    assert C == costs.Costs(0.10, 1.0, 10.0, 0.05)
    assert len(set(am.SETUPS)) == 8 and am.RESULTS.name == "eval_ieee_amount"
    v = am.verdict({s: np.arange(3.0) + i for i, s in enumerate(am.SETUPS)})
    assert v["primary"]["a"] == am.LR_AMT and v["primary"]["b"] == am.LR and v["fourth"]["b"] == am.LR_AMT
