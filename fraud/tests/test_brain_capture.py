import numpy as np

from fraud import brain, capture, costs, rules, runs, signals, windows
from fraud.data import TYPE_ID, Transactions
from system_one import load_config, make_engine


def tx(rows, fraud=()):
    s, t, a, o, d = zip(*rows)
    n = len(rows)
    f = np.zeros(n, bool)
    f[list(fraud)] = True
    return Transactions(np.array(s, np.int16), np.array([TYPE_ID[x] for x in t], np.int8), np.array(a, float),
                        np.array(o, np.int32), np.array(d, np.int32), f, np.zeros(n, np.int8), np.zeros(n, np.int8))


T = tx([
    (1, "PAYMENT", 10.0, 11, 12),
    (1, "TRANSFER", 1234.5, 13, 14),     # new receiver
    (1, "CASH_OUT", 1234.5, 15, 16),     # same amount this hour
    (1, "TRANSFER", 77.0, 17, 14),       # known receiver
    (2, "CASH_OUT", 5000.0, 18, 19),
], fraud=(1, 2))
S = signals.compute(T)


def mock_engine():
    cfg = load_config("config/mock.toml")
    cfg["engine"]["order_debias"] = True
    return make_engine(cfg)


def test_describe_reads_the_flags_and_no_ids():
    new, same, known = brain.describe(T, S, 1), brain.describe(T, S, 2), brain.describe(T, S, 3)
    assert "NEW account" in new and "no earlier transaction" in new
    assert "SAME amount was already moved once" in same
    assert "known account: received money once before" in known
    for i in range(len(T)):
        text = brain.describe(T, S, i)
        assert not any(str(acct) in text.replace("1,234.50", "") for acct in (T.orig[i], T.dest[i]))
        assert "balance" not in text.lower()


def test_options_map_to_actions_in_order():
    assert len(brain.OPTIONS) == len(brain.ACTIONS) == 3
    for opt, word in zip(brain.OPTIONS, ("approve", "analyst", "decline")):
        assert word in opt


def test_model_decider_records_scores_and_timing():
    out = brain.ModelDecider(mock_engine()).decide(T, S, 1)
    assert out["by"] == "model" and out["action"] in brain.ACTIONS
    assert set(out["scores"]) == set(brain.ACTIONS) and abs(sum(out["scores"].values()) - 1) < 1e-6
    assert len(out["orders"]) == 2 and out["ms"] >= 0 and out["prompt_tokens"] > 0


def test_mock_capture_writes_a_finished_trace(tmp_path):
    w = windows.Window(id="dev-test", split="dev", seed=0, steps=(1, 2), rows=np.arange(len(T)), weight=np.ones(len(T)),
                       fraud=2, span_rows=len(T), span_fraud=2)
    eng = mock_engine()
    recs = capture.run_window(T, S, w, capture.chain_for("hybrid", eng), costs.DEFAULT, "hybrid", eng.backend.info())
    path = runs.write_run("hybrid_mock_dev-test", recs, root=tmp_path)
    back = runs.read_run(path)
    assert back[0]["type"] == "header" and back[0]["app"] == "fraud" and back[0]["options"] == list(brain.OPTIONS)
    ticks = [r for r in back if r["type"] == "tx"]
    assert [r["by"] for r in ticks] == ["filter", "model", "model", "model", "model"]
    assert runs.finished(path)["summary"]["model_decisions"] == 4
    assert all(r["by"] != "model" or rules.FRAUD_FREE.isdisjoint({T.type[r["row"]]}) for r in ticks)
