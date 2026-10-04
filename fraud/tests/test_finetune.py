import numpy as np
import pytest

from fraud import brain, eval_ft, finetune, windows
from fraud.data import TYPE_ID, Transactions
from fraud.tests.test_brain_capture import S, T, mock_engine


def synthetic(steps=743, per_step=12, seed=0):
    """Legit PAYMENT/TRANSFER/CASH_OUT rows plus one fraud TRANSFER→CASH_OUT pair every hour."""
    rng = np.random.default_rng(seed)
    step, typ, amount, fraud = [], [], [], []
    kinds = [TYPE_ID["PAYMENT"], TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]]
    for s in range(1, steps + 1):
        for k in range(per_step):
            step.append(s); typ.append(kinds[k % 3]); amount.append(float(rng.integers(1, 999))); fraud.append(0)
        a = float(rng.integers(1000, 9999))
        step += [s, s]; typ += [TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]]; amount += [a, a]; fraud += [1, 1]
    n = len(step)
    return Transactions(np.array(step, np.int16), np.array(typ, np.int8), np.array(amount), np.arange(n, dtype=np.int32),
                        np.arange(n, dtype=np.int32) + n, np.zeros(n, bool), np.array(fraud, np.int8), np.zeros(n, np.int8))


@pytest.fixture(scope="module")
def t():
    return synthetic()


def hours(w):
    return set(range(w.steps[0], w.steps[1] + 1))


# ---------------------------------------------------------------- new test windows

def test_new_windows_share_no_hours_with_old_ones_or_each_other(t):
    kw = dict(size=60, fraud_target=4)
    new = eval_ft.new_windows(t, **kw)
    assert len(new) == eval_ft.N_NEW
    old = set().union(*(hours(windows.make(t, "test", k, **kw)) for k in eval_ft.OLD_TEST_SEEDS))
    seen = set()
    for w in new:
        assert w.split == "test" and w.seed >= eval_ft.NEW_SEED_FROM and w.id == f"test{w.seed}"
        assert not hours(w) & old and not hours(w) & seen
        seen |= hours(w)
    assert [w.seed for w in new] == sorted(w.seed for w in new)


def test_new_windows_are_deterministic(t):
    a = eval_ft.new_windows(t, size=60, fraud_target=4)
    b = eval_ft.new_windows(t, size=60, fraud_target=4)
    assert [w.id for w in a] == [w.id for w in b]
    assert all(np.array_equal(x.rows, y.rows) for x, y in zip(a, b))


# ---------------------------------------------------------------- training rows

def test_training_rows_are_train_steps_model_types_and_skip_dev_hours(t):
    held = finetune.held_out_steps(t, size=60, fraud_target=4)
    rows = finetune.training_rows(t, held, seed=0)
    lo, hi = windows.TRAIN
    assert np.all((t.step[rows] >= lo) & (t.step[rows] <= hi))
    assert set(np.unique(t.type[rows])) <= {TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]}
    assert not set(t.step[rows].tolist()) & held


def test_training_rows_take_all_fraud_and_three_legit_per_fraud(t):
    held = finetune.held_out_steps(t, size=60, fraud_target=4)
    rows = finetune.training_rows(t, held, seed=0)
    pool = ((t.step <= windows.TRAIN[1]) & ~np.isin(t.step, list(held))
            & np.isin(t.type, [TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"]]))
    n_fraud = int(t.is_fraud[pool].sum())
    assert int(t.is_fraud[rows].sum()) == n_fraud
    assert int((t.is_fraud[rows] == 0).sum()) == finetune.LEGIT_PER_FRAUD * n_fraud
    assert len(set(rows.tolist())) == len(rows)


def test_training_rows_depend_only_on_the_seed(t):
    held = finetune.held_out_steps(t, size=60, fraud_target=4)
    a, b = finetune.training_rows(t, held, seed=0), finetune.training_rows(t, held, seed=0)
    c = finetune.training_rows(t, held, seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---------------------------------------------------------------- examples

def test_examples_target_the_truth_in_either_option_order():
    rows = np.array([1, 2, 3, 4])  # fraud, fraud, legit, legit (T/S from test_brain_capture)
    ex = finetune.examples(T, S, rows, seed=0)
    orders = set()
    for (d, target), i in zip(ex, rows):
        want = "decline" if T.is_fraud[i] else "approve"
        assert d.options[target] == brain.OPTIONS[brain.ACTIONS.index(want)]
        assert d.options in (brain.OPTIONS, brain.OPTIONS[::-1])
        assert d.state == brain.describe(T, S, i) and d.context == brain.CONTEXT
        orders.add(d.options)
    many = finetune.examples(T, S, np.array([1, 2, 3, 4] * 10), seed=0)
    assert {d.options for d, _ in many} == {brain.OPTIONS, brain.OPTIONS[::-1]}


def test_encoded_examples_end_with_the_engine_prompt_and_target_one_label_token():
    engine = mock_engine()
    ex = finetune.examples(T, S, np.array([1, 3]), seed=0)
    enc = finetune.encode(engine, ex)
    for (d, target), (ids, tok) in zip(ex, enc):
        labels = engine.labels_for(len(d.options))
        assert ids == engine.backend.encode(engine.render(d, labels))
        assert engine.backend.decode([tok]).strip() == labels[target]


# ---------------------------------------------------------------- thresholds

def test_threshold_decider_maps_the_decline_score_and_keeps_the_rest():
    d = brain.ThresholdDecider(None, t_review=0.5, t_decline=0.9)
    outs = [{"by": "model", "action": "decline", "fallback": "approve" if p[0] >= p[2] else "decline",
             "scores": dict(zip(brain.ACTIONS, p)), "ms": 1.0}
            for p in ((0.9, 0.05, 0.05), (0.3, 0.1, 0.6), (0.0, 0.05, 0.95))]
    assert [d.apply(dict(o))["action"] for o in outs] == ["approve", "review", "decline"]
    out = d.apply(dict(outs[1]))
    assert out["fallback"] == "decline" and out["scores"]["decline"] == 0.6 and out["by"] == "model"
    assert out["thresholds"] == {"review": 0.5, "decline": 0.9}


def test_threshold_decider_is_a_model_decider_on_a_real_engine():
    d = brain.ThresholdDecider(mock_engine(), t_review=0.0, t_decline=1.01)
    assert isinstance(d, brain.ModelDecider)
    assert d.decide(T, S, 1)["action"] == "review"  # every score >= 0 and < 1.01


def test_pick_thresholds_minimises_cost_and_keeps_review_below_decline():
    # Two fraud rows with high scores, two legit rows with low ones: any threshold between them is perfect.
    scores = [np.array([0.95, 0.9, 0.02, 0.1])]
    truth = [np.array([1, 1, 0, 0])]
    amount = [np.array([1000.0, 1000.0, 1000.0, 1000.0])]
    filt = [np.zeros(4, bool)]
    tr, td, grid = eval_ft.pick_thresholds(scores, truth, amount, filt)
    assert all(a <= b for a, b, _ in grid)
    assert 0.1 < td <= 0.9 and tr <= td
    assert min(c for *_, c in grid) == next(c for a, b, c in grid if (a, b) == (tr, td))


def test_auc():
    assert eval_ft.auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert eval_ft.auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
