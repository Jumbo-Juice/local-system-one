import json
import math

import numpy as np

from fraud import costs, eval_ieee as ev, finetune_ieee as fi, ieee_brain, ieee_data
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


def test_verdict_compares_the_fine_tunes_first():
    base = {ev.FROZEN: np.full(20, 100.0), ev.FINETUNED: np.full(20, 90.0), ev.FT_IEEE: np.arange(60.0, 80.0),
            ev.LOGREG_SAME: np.full(20, 95.0), ev.LOGREG_ALL: np.full(20, 80.0), ev.APPROVE_ALL: np.full(20, 500.0)}
    v = ev.verdict(base)
    p = v["ieee_vs_paysim"]
    assert p["better"] == ev.FT_IEEE and p["other"] == ev.FINETUNED and p["cheaper"] == 20 and p["distinguishable"]
    assert v["frozen_vs_paysim"]["better"] == ev.FINETUNED
    assert len(v["reported"]) == 6


def test_compare_says_when_windows_cannot_tell():
    a, b = "a", "b"
    x = np.array([1.0, -1.0] * 10)
    p = ev.compare({a: 100 + x, b: 100 - x}, a, b)
    assert p["better"] is None and not p["distinguishable"]


def test_a_review_costs_10_from_now_and_old_phases_keep_50():
    assert costs.DEFAULT == costs.Costs(0.10, 10.0, 10.0, 0.05)
    assert costs.PHASES_1_10 == costs.Costs(0.10, 10.0, 50.0, 0.05)
    assert ev.COSTS == costs.Costs(0.10, 1.0, 10.0, 0.05)


def test_ieee_training_rows_are_fit_rows_only_at_the_paysim_size():
    t = table(n=60000, fraud_every=4)
    rows = fi.training_rows(t)
    a, _ = ieee_data.split_bounds(len(t))
    assert len(rows) == 12084 and rows.max() < a and len(set(rows.tolist())) == 12084
    assert int(t.is_fraud[rows].sum()) == 3021
    assert (rows == fi.training_rows(t)).all()
    d, want = fi.examples(t, rows[:4])[0]
    assert d.options[want] in ("no: approve it", "yes: decline it")


def test_reprice_changes_decisions_and_keeps_scores(tmp_path, monkeypatch):
    t = table(n=2400)
    make = lambda n, split, count, size=40, seed=ev.SEED, real=ev.make_windows: real(n, split, count, size, seed)
    monkeypatch.setattr(ev, "RUNS", tmp_path)
    monkeypatch.setattr(ev, "DEV_WINDOWS", 2)
    monkeypatch.setattr(ev, "TEST_WINDOWS", 2)
    monkeypatch.setattr(ev, "make_windows", make)
    monkeypatch.setattr(ev.ieee_data, "load", lambda: t)
    monkeypatch.setattr(ev, "SETUPS", (ev.LOGREG_SAME,))
    for stage, split in (("dev", "dev"), ("eval", "test")):
        for w in make(len(t), split, 2):
            outs = [{"by": "logreg", "q": float(t.is_fraud[i]), "ms": 0.0} for i in w.rows]
            recs = ev.trace(t, w, ev.LOGREG_SAME, outs, None, {}, "e")
            recs[0]["costs"] = {"review_fee": 5.0}
            ev.trace_file("e", stage, ev.LOGREG_SAME, w.id).write_text(ev.to_jsonl(recs), encoding="utf-8")
    ev.reprice("e", log=lambda m: None)
    w = make(len(t), "test", 2)[0]
    recs = ev.read_run(ev.trace_file("e", "eval", ev.LOGREG_SAME, w.id))
    tx = [r for r in recs if r.get("type") == "tx"]
    assert recs[0]["costs"]["review_fee"] == 10.0 and "repriced" in recs[0]
    assert [r["q"] for r in tx] == [float(t.is_fraud[i]) for i in w.rows]
    assert any(r["action"] != "approve" for r in tx)
