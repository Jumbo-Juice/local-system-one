import numpy as np
import pytest

from fraud import brain, d3, d3torch, finetune, finetune_d3
from fraud.tests.test_brain_capture import S, T
from fraud.tests.test_d3torch import FakeTokenizer, tiny


def tiny_examples(m, n=24):
    rows = np.array([1, 2, 3, 4] * (n // 4))
    return finetune_d3.encode(m, finetune.examples(T, S, rows, seed=0))


def test_training_stops_at_the_deadline_and_says_so(tmp_path):
    m = tiny()
    ticks = iter(range(0, 10_000, 10))  # every clock read is 10 s later
    out = finetune_d3.train(m, tiny_examples(m), tmp_path, steps=3, deadline=15, clock=lambda: next(ticks),
                            log=lambda _: None)
    assert out["stopped_early"] and out["steps_done"] < out["steps"] == 3
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_training_prompts_are_the_ones_the_decider_reads():
    m = d3torch.D3Torch(None, None, FakeTokenizer(), "cpu")
    rows = np.array([1, 2, 3, 4] * 3)
    ex = finetune.examples(T, S, rows, seed=0)
    enc = finetune_d3.encode(m, ex)
    seen = set()
    for (d, target), (ids, tgt), i in zip(ex, enc, rows):
        order = "fwd" if d.options == brain.OPTIONS else "rev"
        q = d3.request(T, S, int(i), order)["questions"][order]
        assert ids == m.encode(d3.request(T, S, int(i), order)["state"], q["instructions"], list(q["criteria"]))
        assert tgt == target
        seen.add(order)
    assert seen == {"fwd", "rev"}


def test_plan_takes_the_full_epoch_when_it_fits_the_step_budget():
    assert finetune_d3.plan_steps(3.0) == (1510, True)    # 1,510 × 3 s = 75 min < 110 min


def test_plan_cuts_to_the_step_budget_when_the_epoch_does_not_fit():
    assert finetune_d3.plan_steps(6.0) == (1100, True)    # floor(6,600 / 6)


def test_plan_refuses_below_half_an_epoch():
    assert finetune_d3.plan_steps(10.0) == (660, False)   # 660 < 755: ask the owner first


# ---------------------------------------------------------------- eval_d3ft

from fraud import eval_d3ft  # noqa: E402


def costs_of(*values):
    return {f"test{200 + k}": {"cost": float(v)} for k, v in enumerate(values)}


def test_bars_count_ties_as_not_worse_and_need_all_twenty_windows():
    mine = costs_of(*([800] * 10 + [900] * 10))
    phase5 = {"logreg": costs_of(*([800] * 10 + [850] * 10)),      # 10 ties, 10 dearer -> 10 of 20
              "hybrid-ft": costs_of(*([850] * 9 + [700] * 11))}     # 9 cheaper, 11 dearer -> 9 of 20
    b = eval_d3ft.bars(mine, phase5, 240.0)
    assert (b["barA"]["value"], b["barA"]["pass"]) == (10, True)
    assert (b["barC"]["value"], b["barC"]["pass"]) == (9, False)
    assert b["barB"]["pass"] and eval_d3ft.bars(mine, phase5, 250.0)["barB"]["pass"]
    assert not eval_d3ft.bars(mine, phase5, 251.0)["barB"]["pass"]
    short = eval_d3ft.bars(dict(list(mine.items())[:19]), phase5, 240.0)
    assert short["barA"] == {**short["barA"], "value": None, "pass": False}


def trace(rows):
    """One window: (row, truth, fwd decline, rev decline, prompt tokens) per model row, plus a filter row."""
    recs = [{"type": "tx", "row": -1, "by": "filter", "truth": 0}]
    for row, y, f, r, n in rows:
        recs.append({"type": "tx", "row": row, "by": "model", "truth": y, "prompt_tokens": n,
                     "orders": [{"decline": f}, {"decline": r}]})
    return [recs]


REF = trace([(1, 1, 0.9, 0.8, 454), (2, 0, 0.2, 0.1, 454), (3, 0, 0.4, 0.3, 460), (4, 1, 0.6, 0.7, 454)])


def test_repro_passes_for_small_drift():
    ours = trace([(1, 1, 0.91, 0.79, 454), (2, 0, 0.21, 0.1, 454), (3, 0, 0.39, 0.31, 460), (4, 1, 0.6, 0.7, 454)])
    r = eval_d3ft.repro_compare(ours, REF)
    assert r["pass"] and r["rows"] == 4 and r["fwd"]["median_abs_diff"] == pytest.approx(0.01)


def test_repro_fails_on_one_token_count():
    ours = trace([(1, 1, 0.9, 0.8, 454), (2, 0, 0.2, 0.1, 454), (3, 0, 0.4, 0.3, 459), (4, 1, 0.6, 0.7, 454)])
    r = eval_d3ft.repro_compare(ours, REF)
    assert not r["R1"] and r["R2"] and r["R3"] and not r["pass"]


def test_repro_fails_when_scores_drift():
    ours = trace([(1, 1, 0.95, 0.8, 454), (2, 0, 0.25, 0.1, 454), (3, 0, 0.45, 0.3, 460), (4, 1, 0.65, 0.7, 454)])
    r = eval_d3ft.repro_compare(ours, REF)   # +0.05 on every fwd row: same ranking, median drift 0.05
    assert r["R1"] and r["R2"] and not r["R3"] and not r["pass"]


def test_repro_fails_when_the_ranking_changes():
    ours = trace([(1, 1, 0.1, 0.8, 454), (2, 0, 0.21, 0.1, 454), (3, 0, 0.4, 0.3, 460), (4, 1, 0.6, 0.7, 454)])
    r = eval_d3ft.repro_compare(ours, REF)   # one fraud row drops to the bottom: fwd AUC 1.0 -> 0.5
    assert r["fwd"]["auc_phase6"] == 1.0 and r["fwd"]["auc"] == 0.5 and not r["R2"] and not r["pass"]


def test_repro_fails_on_different_rows():
    ours = trace([(1, 1, 0.9, 0.8, 454), (2, 0, 0.2, 0.1, 454), (3, 0, 0.4, 0.3, 460)])
    assert not eval_d3ft.repro_compare(ours, REF)["pass"]


def test_lr_warms_up_over_30_batches_and_decays_to_zero_at_the_planned_end():
    assert finetune_d3.lr_factor(0, 100) == pytest.approx(1 / 30)
    assert finetune_d3.lr_factor(29, 100) == pytest.approx(0.71)
    assert finetune_d3.lr_factor(99, 100) == pytest.approx(0.01)
    assert finetune_d3.lr_factor(99, 1510) == pytest.approx(1 - 99 / 1510)


def test_first_loss_is_cross_entropy_of_the_first_batch_targets(tmp_path):
    import torch
    enc = tiny_examples(tiny())
    with torch.no_grad():   # LoRA starts as a no-op (lora_B = 0), so batch 1 sees the base model
        logits = tiny().logits([e[0] for e in enc[:8]], 3)
        want = torch.nn.functional.cross_entropy(logits, torch.tensor([e[1] for e in enc[:8]])).item()
    out = finetune_d3.train(tiny(), enc, tmp_path, steps=1, log=lambda _: None)
    assert out["losses"][0] == pytest.approx(want, abs=1e-4) and not out["stopped_early"]


def test_the_lr_schedule_is_sized_to_the_planned_batches(tmp_path):
    out = finetune_d3.train(tiny(), tiny_examples(tiny()), tmp_path, steps=2, log=lambda _: None)
    # batch 1: 2e-4 × 1/30; batch 2: 2e-4 × 2/30 × (1 − 1/2)
    assert out["lrs"] == pytest.approx([2e-4 / 30, 2e-4 / 30])


def test_calibration_of_d3_decisions_without_outside_mass():
    # D3's readout is masked to the option codes, so its decisions carry no outside_mass.
    from fraud import eval as ev
    recs = [{"type": "tx", "by": "model", "truth": 0, "scores": {"approve": 0.9, "review": 0.05, "decline": 0.05}},
            {"type": "tx", "by": "model", "truth": 1, "scores": {"approve": 0.2, "review": 0.1, "decline": 0.7}}]
    c = ev.model_calibration([recs])
    assert c["decisions"] == 2 and c["top_right_rate"] == 1.0 and c["outside_mass_mean"] is None
