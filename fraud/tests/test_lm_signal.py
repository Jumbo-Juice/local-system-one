import numpy as np

from fraud import eval_lm_nuc as lm, history as h, ieee_brain, lm_signal as ls
from fraud.gbdt import Params

from .test_eval_lm_nuc import make


def test_folds_cover_every_fit_row_once_in_time_order():
    blocks = ls.folds(10, 4)
    assert len(blocks) == 4 and np.array_equal(np.concatenate(blocks), np.arange(10))


def test_a_row_is_scored_by_a_model_that_did_not_see_it():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 2))
    y = rng.integers(0, 2, 600).astype(np.float64)  # labels with no pattern: a model that saw them would score them above chance
    p = Params(trees=60, depth=4, lr=0.3, min_hessian=0.1, l2=0.1, row_sample=1.0, col_sample=1.0, threads=1)
    q = ls.out_of_fold(x, y, ls.folds(600, 3), p, log=lambda _: None)
    assert np.isfinite(q).all()
    assert abs(q[y == 1].mean() - q[y == 0].mean()) < 0.05


def test_the_signal_is_one_line_and_only_when_given():
    t, cache = make()
    plain = ieee_brain.describe_rich(t, 150, cache)
    assert "Risk score" not in plain
    signal = {**cache, "signal": np.full(len(t), 0.0342)}
    text = ieee_brain.describe_rich(t, 150, signal)
    assert text.startswith(plain) and text.endswith("Risk score from a tabular model: 0.034 (uncalibrated; higher is more suspicious).")


def test_a_row_with_no_signal_refuses_rather_than_guessing():
    t, cache = make()
    signal = {**cache, "signal": np.full(len(t), np.nan)}
    try:
        ieee_brain.describe_rich(t, 150, signal)
    except ValueError as err:
        assert "no tabular score" in str(err)
    else:
        raise AssertionError("expected a refusal")


def test_the_signal_setup_reads_history_and_signal():
    assert lm.needs_history("lm-rich") and lm.needs_history("lm-signal") and not lm.needs_history("lm-seed1")
    assert h.CACHE.name == "history.npz"
