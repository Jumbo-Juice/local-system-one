import numpy as np
import pytest

from fraud import brain, capture, costs, d3, eval_d3, eval_lfm, rules, windows
from fraud.tests.test_brain_capture import S, T, mock_engine


def test_models_are_pinned():
    assert d3.MODELS["d1-omni-2b"][0].startswith("lfm/") and len(d3.MODELS["d1-omni-2b"][1]) == 64
    assert d3.gguf_path("d3-nano").parent.name == "d3"  # Phase 6 paths unchanged on disk
    assert len(eval_lfm.LFM3["sha256"]) == 64 and len(eval_lfm.SERVER_ZIP_SHA) == 64
    assert set(eval_lfm.SETUPS) == set(eval_lfm.MODELS)


def test_order_reads_dicts_and_engine_lists():
    assert eval_d3.order({"orders": [{"approve": 0.1, "review": 0.2, "decline": 0.7}]}, 0)["decline"] == 0.7
    assert eval_d3.order({"orders": [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]}, 1) == dict(zip(brain.ACTIONS, [0.2, 0.3, 0.5]))


def test_variant_decider_reads_one_order_and_applies_thresholds():
    eng = mock_engine()
    both = eval_lfm.VariantDecider(eng, "both").decide(T, S, 1)
    assert len(both["orders"]) == 2
    rev = eval_lfm.VariantDecider(eng, "rev", (0.0, 1.01)).decide(T, S, 1)
    assert rev["scores"] == pytest.approx(eval_d3.order(both, 1))
    assert rev["action"] == "review" and rev["thresholds"] == {"review": 0.0, "decline": 1.01}
    fwd = eval_lfm.VariantDecider(eng, "fwd").decide(T, S, 1)
    assert fwd["scores"]["decline"] == pytest.approx(eval_d3.order(both, 0)["decline"], abs=1e-6)


def test_engine_traces_feed_the_debias_analysis():
    w = windows.Window("dev-test", "dev", 0, (1, 2), np.arange(len(T)), np.ones(len(T)), 2, len(T), 2)
    recs = capture.run_window(T, S, w, [rules.TypeFilter(), eval_lfm.VariantDecider(mock_engine(), "both")],
                              costs.DEFAULT, "hybrid-lfm317b")
    model = [r for r in recs if r.get("type") == "tx" and r["by"] == "model"]
    assert model and all(len(r["orders"]) == 2 for r in model)
    for variant in d3.ORDERS:
        assert len(eval_d3.variant_view([recs], variant)["scores"][0]) == len(T)
    assert 0.0 <= eval_d3.flip_rate([recs]) <= 1.0
    v = eval_d3.variant_view([recs], "both")
    assert set(eval_lfm.type_aucs([recs], v)) == {"TRANSFER", "CASH_OUT"}
    assert eval_lfm.dev_cost(v, 0.5, 0.9) >= 0.0
