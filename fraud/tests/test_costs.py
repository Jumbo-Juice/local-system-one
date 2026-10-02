import numpy as np
import pytest

from fraud.costs import Costs, score

C = Costs(friction_rate=0.1, friction_min=5.0, review_fee=20.0, review_share=0.25)


def test_unit_costs():
    fraud = np.array([1, 1, 0, 0])
    amount = np.array([1000.0, 1000.0, 1000.0, 10.0])
    s = score(Costs(0.1, 5.0, 20.0, 1.0), ["approve", "decline", "decline", "decline"], ["approve"] * 4, fraud, amount)
    assert s.cost.tolist() == [1000.0, 0.0, 100.0, 5.0]  # friction has a floor


def test_review_budget_in_arrival_order_then_fallback():
    fraud = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    amount = np.full(8, 100.0)
    wanted = ["review"] * 8
    fallback = ["decline", "approve", "approve", "decline", "decline", "approve", "approve", "approve"]
    s = score(C, wanted, fallback, fraud, amount)  # budget = floor(0.25 * 8) = 2
    assert s.final[:2] == ["review", "review"]
    assert s.final[2:] == fallback[2:]
    assert s.budget_left == [1, 0, 0, 0, 0, 0, 0, 0]
    assert s.cost[:3].tolist() == [20.0, 20.0, 100.0]  # third: fraud approved by fallback
    assert s.summary["reviews_wanted"] == 8 and s.summary["reviews_done"] == 2


def test_summary_metrics_and_natural_weights():
    fraud = np.array([1, 1, 0, 0])
    amount = np.array([500.0, 300.0, 100.0, 100.0])
    s = score(C, ["decline", "approve", "decline", "approve"], ["approve"] * 4, fraud, amount,
              weight=np.array([1.0, 1.0, 10.0, 10.0]))
    m = s.summary
    assert m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["fraud_amount_stopped"] == 500.0
    assert m["confusion"]["fraud"] == {"approve": 1, "review": 0, "decline": 1}
    assert m["cost"] == 300.0 + 10.0
    assert m["cost_natural"] == pytest.approx(300.0 + 100.0)
    assert m["precision_natural"] == pytest.approx(1 / 11)


def test_rejects_unknown_action():
    with pytest.raises(ValueError):
        score(C, ["hold"], ["approve"], np.array([0]), np.array([1.0]))
