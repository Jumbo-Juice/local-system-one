import numpy as np
import torch

from fraud.eval_ieee import COSTS
from fraud.eval_lm_cost import big_frauds
from fraud.finetune_d3 import batch_loss
from fraud.finetune_ieee import cost_weights


def rows():
    rng = np.random.default_rng(0)
    amount = rng.lognormal(4, 1.2, 4000)
    return amount, np.arange(4000) % 4 == 0


def test_weights_keep_the_class_balance():
    amount, fraud = rows()
    w = cost_weights(amount, fraud, COSTS)
    assert np.isclose(w[fraud].mean(), 1) and np.isclose(w[~fraud].mean(), 1)


def test_a_bigger_fraud_counts_for_more_up_to_the_cap():
    amount, fraud = rows()
    w = cost_weights(amount, fraud, COSTS)
    order = np.argsort(amount[fraud])
    assert np.all(np.diff(w[fraud][order]) >= -1e-12)
    assert np.isclose(w[fraud].max(), np.partition(w[fraud], -1)[-1]) and (w[fraud] == w[fraud].max()).sum() >= 10


def test_legit_rows_are_weighted_by_friction_with_its_minimum():
    amount = np.array([1.0, 2.0, 50.0, 100.0, 100.0, 100.0])
    fraud = np.array([False, False, False, False, True, True])
    w = cost_weights(amount, fraud, COSTS, cap_pct=100.0)
    assert np.isclose(w[0], w[1])  # both under the 1 USD minimum
    assert np.isclose(w[3] / w[2], 2.0)


def test_no_weight_is_the_plain_loss_and_equal_weights_match_it():
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.2, 0.3, 0.1]])
    y = torch.tensor([0, 2])
    plain = torch.nn.functional.cross_entropy(logits, y)
    assert batch_loss(logits, y) == plain
    assert torch.isclose(batch_loss(logits, y, torch.ones(2)), plain)


def test_weights_move_the_loss_toward_the_heavy_row():
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.2, 0.3, 0.1]])
    y = torch.tensor([0, 2])
    each = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    assert torch.isclose(batch_loss(logits, y, torch.tensor([0.0, 2.0])), each[1])


def test_big_frauds_counts_only_frauds_at_or_over_the_amount():
    tx = lambda amount, truth, action: {"type": "tx", "truth": truth, "action": action, "tx": {"amount": amount}}
    trace = [[{"type": "header"}, tx(500.0, 1, "decline"), tx(900.0, 1, "approve"), tx(499.0, 1, "decline"), tx(800.0, 0, "decline")]]
    assert big_frauds(trace) == {"big_frauds": 2, "big_frauds_stopped": 1}
