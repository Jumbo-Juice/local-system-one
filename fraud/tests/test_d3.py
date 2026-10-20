import numpy as np
import pytest

from fraud import brain, capture, costs, d3, eval_d3, rules, windows
from fraud.tests.test_brain_capture import S, T


def fake_post(prefer_first=0.0):
    """A stand-in for /v1/systemone: decline-ish when the receiver is new, plus a bias toward the
    first option shown (so the two orders differ)."""
    seen = []

    def post(body):
        seen.append(body)
        new = "NEW account" in body["state"]
        answers = {}
        for q, spec in body["questions"].items():
            opts = list(spec["criteria"])
            base = {brain.OPTIONS[0]: 0.2, brain.OPTIONS[1]: 0.2, brain.OPTIONS[2]: 0.6} if new else \
                   {brain.OPTIONS[0]: 0.7, brain.OPTIONS[1]: 0.2, brain.OPTIONS[2]: 0.1}
            p = {o: base[o] + (prefer_first if k == 0 else 0.0) for k, o in enumerate(opts)}
            z = sum(p.values())
            answers[q] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": {o: v / z for o, v in p.items()}}
        return {"answers": answers, "usage": {"input_tokens": 120, "output_tokens": 0}}

    post.seen = seen
    return post


def test_request_is_prompt_v4_in_both_orders():
    body = d3.request(T, S, 1, "both")
    assert body["state"].startswith(brain.CONTEXT) and brain.describe(T, S, 1) in body["state"]
    assert list(body["questions"]) == ["fwd", "rev"]
    assert list(body["questions"]["fwd"]["criteria"]) == list(brain.OPTIONS)
    assert list(body["questions"]["rev"]["criteria"]) == list(brain.OPTIONS[::-1])
    assert all(q["type"] == "choice" and q["instructions"] == brain.QUESTION for q in body["questions"].values())
    assert list(d3.request(T, S, 1, "rev")["questions"]) == ["rev"]
    assert "balance" not in str(body).lower()
    with pytest.raises(ValueError):
        d3.request(T, S, 1, "sideways")


def test_decider_averages_orders_and_maps_actions():
    post = fake_post(prefer_first=0.3)
    out = d3.D3Decider(post, "both").decide(T, S, 1)
    assert len(post.seen) == 1 and len(out["orders"]) == 2
    assert out["orders"][0]["approve"] > out["orders"][1]["approve"]  # the first-shown bias moves with the order
    for a in brain.ACTIONS:
        assert out["scores"][a] == pytest.approx((out["orders"][0][a] + out["orders"][1][a]) / 2)
    assert out["action"] == "decline" and out["fallback"] == "decline" and out["prompt_tokens"] == 120
    single = d3.D3Decider(fake_post(), "fwd").decide(T, S, 3)
    assert len(single["orders"]) == 1 and single["action"] == "approve"


def test_thresholds_override_argmax():
    out = d3.D3Decider(fake_post(), "fwd", (0.05, 0.95)).decide(T, S, 1)  # decline score 0.6
    assert out["action"] == "review" and out["thresholds"] == {"review": 0.05, "decline": 0.95}


def test_debias_rule():
    assert eval_d3.choose_variant({"both": 0.910, "fwd": 0.900, "rev": 0.880}) == ("both", pytest.approx(0.010))
    assert eval_d3.choose_variant({"both": 0.904, "fwd": 0.900, "rev": 0.880})[0] == "fwd"
    assert eval_d3.choose_variant({"both": 0.904, "fwd": 0.880, "rev": 0.900})[0] == "rev"


def test_capture_trace_keeps_both_orders_for_the_dev_analysis():
    w = windows.Window("dev-test", "dev", 0, (1, 2), np.arange(len(T)), np.ones(len(T)), 2, len(T), 2)
    recs = capture.run_window(T, S, w, [rules.TypeFilter(), d3.D3Decider(fake_post(0.3), "both")], costs.DEFAULT, "hybrid-d3nano")
    model = [r for r in recs if r.get("type") == "tx" and r["by"] == "model"]
    assert model and all(len(r["orders"]) == 2 for r in model)
    traces = [recs]
    for variant in d3.ORDERS:
        v = eval_d3.variant_view(traces, variant)
        assert len(v["scores"][0]) == len(T)
    fwd = eval_d3.variant_view(traces, "fwd")["scores"][0]
    assert fwd[1] == pytest.approx(model[0]["orders"][0]["decline"])
    assert 0.0 <= eval_d3.flip_rate(traces) <= 1.0
