import numpy as np
import pytest

from fraud import reviewer_check as rc

Q = np.array([0.9, 0.8, 0.7, 0.2])
Y = np.array([1, 1, 0, 1])
A = np.array([1000.0, 1000.0, 1000.0, 1000.0])
ONE = np.ones(4)


def test_cost_of_blocking_and_of_the_human():
    assert rc.queue_cost(Q, Y, A, ONE, np.inf) == 200.0        # all human
    assert rc.queue_cost(Q, Y, A, ONE, 0.8) == 100.0           # two correct blocks, two reviews
    assert rc.queue_cost(Q, Y, A, ONE, 0.7) == 100.0 + 50.0    # a wrong block costs 10% of 1,000


def test_best_threshold_stops_before_a_wrong_block_and_may_block_nothing():
    assert rc.best_threshold(Q, Y, A, ONE) == 0.8
    y = np.array([0, 1, 1, 1])  # the top score is legit and dear: blocking never pays below it
    a = np.array([1e6, 10.0, 10.0, 10.0])
    assert rc.best_threshold(Q, y, a, ONE) == np.inf


def test_cross_fold_scores_the_other_folds_threshold():
    k = np.array([0, 5, 0, 5])
    r = rc.cross_fold(Q, Y, A, ONE, k)
    # fold 0 (0.9 fraud, 0.7 legit) picks 0.9, which blocks nothing on fold 5; fold 5 (0.8, 0.2, both fraud) picks 0.2
    assert r["thresholds"] == pytest.approx([0.9, 0.2])
    assert r["blocked"] == 2 and r["wrong_blocks"] == 1 and r["cost"] == 100.0 + 0.0 + 100.0
