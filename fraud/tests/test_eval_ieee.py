import json
import math

import numpy as np

from fraud import costs, eval_ieee as ev, ieee_brain, ieee_data
from fraud.ieee_data import TEXT, Ieee


def table(n=40, fraud_every=5):
    names = ("card1", "card2", "addr1", "addr2", "C1", "C4", "C8", "C13", "D1", "V1")
    num = np.ones((n, len(names)), np.float32)
    num[0, names.index("D1")] = np.nan
    vocab = {k: ["", "x", "y"] for k in TEXT}
    return Ieee(ids=np.arange(n), dt=86400 + np.arange(n) * 3600, amount=np.linspace(10, 100, n),
                is_fraud=(np.arange(n) % fraud_every == 0).astype(np.int8), num=num, num_names=names,
                cat=np.ones((n, len(TEXT)), np.int16), cat_names=TEXT, vocab=vocab, has_identity=np.zeros(n, bool))


def test_the_draw_is_fixed_and_matches_the_setups():
    assert ev.draw() == ("qwen2.5-1.5b", "20261102-152745_lora-d1")
    assert ev.FINETUNED_POOL[2] == ev.draw()[1]


def test_split_is_60_20_20_in_order():
    assert ieee_data.split_bounds(100) == (60, 80)
    assert ieee_data.split_rows(100, "fit") == (0, 60)
    assert ieee_data.split_rows(100, "dev") == (60, 80)
    assert ieee_data.split_rows(100, "test") == (80, 100)


def test_windows_stay_in_their_split_and_do_not_overlap():
    n = 590540
    for split, count in (("dev", ev.DEV_WINDOWS), ("test", ev.TEST_WINDOWS)):
        ws = ev.make_windows(n, split, count)
        lo, hi = ieee_data.split_rows(n, split)
        assert len(ws) == count
        for a, b in zip(ws, ws[1:]):
            assert a.rows[-1] < b.rows[0]
        assert all(lo <= w.rows[0] and w.rows[-1] < hi and len(w.rows) == ev.WINDOW_SIZE for w in ws)
        assert [w.rows[0] for w in ws] == [w.rows[0] for w in ev.make_windows(n, split, count)]


def test_describe_shows_named_fields_and_never_the_label_or_ids():
    t = table()
    text = ieee_brain.describe(t, 0)
    assert "D1: none" in text and "product x" in text and "isFraud" not in text and "card1" not in text
    assert ieee_brain.OPTIONS == ("no: approve it", "unsure: send it to an analyst", "yes: decline it")
    assert set(ieee_brain.request(t, 0, "both")["questions"]) == {"fwd", "rev"}
    assert set(ieee_brain.request(t, 0, "fwd")["questions"]) == {"fwd"}


def test_hour_is_the_repeating_clock():
    t = table()
    assert ieee_brain.hour(t, 0) == 0 and ieee_brain.hour(t, 25) == 1


def test_features_have_no_nan_and_skip_identifiers():
    t = table()
    idx = np.arange(10)
    assert not np.isnan(ev.features_same(t, idx)).any()
    full = ev.features_all(t, idx)
    assert not np.isnan(full).any()
    assert full.shape[1] == 1 + 24 + 2 * len(TEXT) + 6 + 5  # amount, hours, one-hots, numeric minus 4 ids, non-V flags


def test_act_treats_none_as_never():
    assert ev.act(0.9, None, None) == "approve"
    assert ev.act(0.9, 0.5, None) == "review"
    assert ev.act(0.9, 0.5, 0.8) == "decline"
    assert ev.act(0.1, 0.5, 0.8) == "approve"


def test_thresholds_can_choose_approve_all_and_find_a_perfect_scorer():
    y = np.array([1] * 5 + [0] * 95)
    a = np.full(100, 100.0)
    tr, td, _ = ev.pick_thresholds([y.astype(float)], [y], [a], ev.COSTS)
    assert ev.act(1.0, tr, td) != "approve" and ev.act(0.0, tr, td) == "approve"
    tiny = np.full(100, 0.01)  # the amounts at stake are below a review fee and the friction minimum
    tr, td, _ = ev.pick_thresholds([np.linspace(0, 1, 100)], [y], [tiny], ev.COSTS)
    assert tr is None and td is None


def test_auc_counts_ties_as_half():
    y = np.array([1, 0, 1, 0])
    assert ev.auc(np.array([0.9, 0.1, 0.8, 0.2]), y) == 1.0
    assert ev.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert math.isnan(ev.auc(np.array([0.5, 0.5]), np.array([0, 0])))


def test_trace_scores_at_natural_rate_and_budget_falls_back_to_approve():
    t = table()
    w = ev.Win("dev0", "dev", np.arange(40))
    outs = [{"by": "model", "q": 1.0, "ms": 1.0, "argmax": "decline"} for _ in w.rows]
    recs = ev.trace(t, w, ev.FROZEN, outs, (0.5, None), {}, "x")
    tx = [r for r in recs if r.get("type") == "tx"]
    assert sum(r["action"] == "review" for r in tx) == 2  # 5% of 40
    assert all(r["action"] == "approve" and r["wanted"] == "review" for r in tx[2:])
    assert recs[0]["dataset"] == "IEEE-CIS" and recs[0]["window"]["enriched"] is False
    json.dumps(recs)


def test_verdict_needs_logreg_and_approve_all():
    base = {ev.FROZEN: np.full(20, 100.0), ev.FINETUNED: np.full(20, 90.0), ev.LOGREG_SAME: np.full(20, 95.0),
            ev.LOGREG_ALL: np.full(20, 80.0), ev.APPROVE_ALL: np.full(20, 500.0)}
    v = ev.verdict(base)
    assert v["better"] == ev.FINETUNED and v["bar_logreg"] and v["bar_approve_all"] and not v["retraining_necessary"]
    v = ev.verdict({**base, ev.LOGREG_SAME: np.full(20, 50.0)})
    assert not v["bar_logreg"] and v["retraining_necessary"]
    v = ev.verdict({**base, ev.APPROVE_ALL: np.full(20, 10.0)})
    assert not v["bar_approve_all"] and v["retraining_necessary"]


def test_costs_defaults_are_untouched():
    assert costs.DEFAULT == costs.Costs(0.10, 10.0, 50.0, 0.05)
