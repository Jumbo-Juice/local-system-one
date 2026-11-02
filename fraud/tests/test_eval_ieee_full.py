import json

import numpy as np

from fraud import eval_ieee as ev, eval_ieee_full as fu, finetune_ieee as fi, ieee_data
from fraud.ieee_data import TEXT, Ieee


def table(n=2400, fraud_every=5):
    names = ("card1", "card2", "addr1", "addr2", "C1", "C4", "C8", "C13", "D1", "V1")
    vocab = {k: ["", "x", "y"] for k in TEXT}
    return Ieee(ids=np.arange(n), dt=86400 + np.arange(n) * 3600, amount=np.linspace(10, 100, n),
                is_fraud=(np.arange(n) % fraud_every == 0).astype(np.int8), num=np.ones((n, len(names)), np.float32), num_names=names,
                cat=np.ones((n, len(TEXT)), np.int16), cat_names=TEXT, vocab=vocab, has_identity=np.zeros(n, bool))


def test_new_windows_stay_in_test_and_share_no_row_with_phase_11():
    n = 590540
    new = fu.new_windows(n)
    old = ev.make_windows(n, "test", ev.TEST_WINDOWS)
    lo, hi = ieee_data.split_rows(n, "test")
    assert len(new) == 20 and all(len(w.rows) == ev.WINDOW_SIZE and lo <= w.rows[0] and w.rows[-1] < hi for w in new)
    rows = np.concatenate([w.rows for w in new + old])
    assert len(np.unique(rows)) == len(rows)
    assert [w.rows[0] for w in new] == [w.rows[0] for w in fu.new_windows(n)]
    assert new[0].header()["seed"] == fu.SEED and old[0].header()["seed"] == ev.SEED


def test_full_training_rows_use_every_fit_fraud_and_nothing_after_fit():
    t = table(n=60000, fraud_every=20)
    a, _ = ieee_data.split_bounds(len(t))
    rows = fi.training_rows(t, full=True)
    fit_fraud = int(t.is_fraud[:a].sum())
    assert int(t.is_fraud[rows].sum()) == fit_fraud
    assert len(rows) == 4 * fit_fraud and rows.max() < a and len(set(rows.tolist())) == len(rows)


def test_verdict_puts_the_primary_question_first():
    base = {fu.FT_FULL: np.arange(60.0, 80.0), ev.FT_IEEE: np.full(20, 90.0), ev.LOGREG_SAME: np.full(20, 95.0),
            ev.LOGREG_ALL: np.full(20, 50.0), ev.APPROVE_ALL: np.full(20, 500.0)}
    v = fu.verdict(base)
    assert v["full_vs_old"]["better"] == fu.FT_FULL and v["full_vs_old"]["distinguishable"]
    assert v["full_vs_logreg_all"]["better"] == ev.LOGREG_ALL
    assert len(v["reported"]) == 5


def test_report_renders_from_traces_and_reads_old_settings(tmp_path, monkeypatch):
    t = table()
    real = ev.make_windows
    monkeypatch.setattr(ev, "RUNS", tmp_path)
    monkeypatch.setattr(ev, "DEV_WINDOWS", 2)
    monkeypatch.setattr(ev, "TEST_WINDOWS", 2)
    monkeypatch.setattr(ev, "WINDOW_SIZE", 40)
    monkeypatch.setattr(ev, "make_windows", lambda n, split, count, size=40, seed=ev.SEED: real(n, split, count, size, seed))
    monkeypatch.setattr(ev.ieee_data, "load", lambda: t)
    monkeypatch.setattr(fu, "RESULTS", tmp_path / "results")
    rng = np.random.default_rng(0)

    def outs(w, setup):
        if setup == ev.APPROVE_ALL:
            return [{"by": "all", "q": 0.0, "ms": 0.0} for _ in w.rows]
        by = "model" if setup in fu.MODEL_SETUPS else "logreg"
        out = []
        for i in w.rows:
            q = float(np.clip(t.is_fraud[i] * 0.5 + rng.random() * 0.5, 0, 1))
            o = {"by": by, "q": q, "ms": 90.0 if by == "model" else 0.0}
            if by == "model":
                s = {"approve": 1 - q, "review": 0.0, "decline": q}
                o.update(scores=s, orders=[s, s], prompt_tokens=228, argmax="decline" if q > 0.5 else "approve")
            out.append(o)
        return out

    def write(eval_id, stage, setup, w, thresholds):
        ev.trace_file(eval_id, stage, setup, w.id).write_text(
            ev.to_jsonl(ev.trace(t, w, setup, outs(w, setup), thresholds, {"adapter": {"dir": setup}}, eval_id)), encoding="utf-8")

    th = (0.6, 0.8)
    for s in (ev.FT_IEEE, ev.LOGREG_SAME, ev.LOGREG_ALL, ev.APPROVE_ALL):
        for w in fu.dev_windows(len(t)):
            write(fu.OLD_EVAL, "dev", s, w, None)
        for w in ev.make_windows(len(t), "test", 2):
            write(fu.OLD_EVAL, "eval", s, w, th)
    for w in fu.dev_windows(len(t)):
        write("e", "dev", fu.FT_FULL, w, None)
    for s in fu.SETUPS:
        for w in fu.new_windows(len(t), 2, 40):
            write("e", "eval", s, w, th)

    variant, thresholds = fu.old_settings(ev.FT_IEEE, len(t))
    assert variant in ("both", "fwd", "rev") and thresholds == th
    assert fu.old_settings(ev.LOGREG_ALL, len(t)) == ("both", th)
    got = fu.report("e", log=lambda m: None)
    assert got["test_rows"] == 80 and set(got["setups"]) == set(fu.SETUPS) and all(got["bar_latency"].values())
    assert set(got["sensitivity"]) == {"review $1", "review $5", "review $20"}
    md = (tmp_path / "results" / "e.md").read_text(encoding="utf-8")
    assert "Full adapter vs Phase 11's adapter" in md and "undefined" not in md and "nan" not in md.lower()
    json.loads((tmp_path / "results" / "e.json").read_text(encoding="utf-8"))
