from types import SimpleNamespace

import pytest
import torch

from fraud import brain, d1torch, d3, finetune_d1
from fraud.tests.test_brain_capture import S, T


class FakeTokenizer:
    """Characters as token ids."""

    pad_token_id = 0
    bos_token = "^"

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return [1 + ord(c) % 60 for c in text]


def tiny(seed=0, pack=False, temperature=1.0):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(vocab_size=64, hidden_size=32, intermediate_size=64, head_dim=16, num_attention_heads=2,
                          num_key_value_heads=1, norm_eps=1e-5, conv_L_cache=3, sliding_window=8,
                          layer_types=["conv", "full_attention", "conv", "sliding_attention", "full_attention"],
                          rope_parameters={"full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
                                           "sliding_attention": {"rope_theta": 1e4}})
    lm = d1torch.Lfm3Text(cfg).float().eval()
    return d1torch.D1Torch(lm, FakeTokenizer(), "cpu", temperature, pack)


def test_prompt_is_the_repo_format():
    # The literal is written from prompt.prefix_text + suffix_layout (LiquidAI/d1-Omni-2B @ 431f1d667d), not our code.
    text = d1torch.prefix_text("<s>", "a state") + d1torch.suffix_text("Fraud?", ["no: approve it", "yes: decline_it"])
    assert text == ("<s><|im_start|>user\na state\n\n\nQUESTION:\nFraud?\n\nOptions:\nA no: approve it\n"
                    "B yes: decline it\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
                    "<think></think>")


def test_rope_keeps_the_full_dims_progression_with_a_zero_tail():
    cfg = SimpleNamespace(head_dim=128, rope_parameters={"full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5}})
    inv = d1torch.rope_frequencies(cfg, "full_attention")
    assert len(inv) == 64 and (inv[32:] == 0).all() and (inv[:32] > 0).all()
    assert inv[1].item() == pytest.approx(1e6 ** (-2 / 128))


def test_set_masks_state_sees_all_a_question_sees_state_and_itself():
    cfg = SimpleNamespace(sliding_window=2)
    question = torch.tensor([[-1, -1, 1, 1, 2, 2, -1]])
    real = torch.tensor([[1, 1, 1, 1, 1, 1, 0]]).bool()
    m = d1torch.masks_for(cfg, torch.tensor([2]), question, real)
    full, sliding = m["full_attention"][0, 0], m["sliding_attention"][0, 0]
    assert full[0].tolist() == [True] * 6 + [False]                           # a state token: all but padding
    assert full[2].tolist() == [True, True, True, True, False, False, False]  # question 1: state and itself
    assert full[5].tolist() == [True, True, False, False, True, True, False]  # question 2: state and itself
    assert sliding[4].tolist() == [False, False, False, True, True, False, False]  # causal within the window


def test_right_padding_does_not_change_a_row():
    m = tiny()
    short, long = list(range(1, 12)), list(range(5, 30))
    with torch.no_grad():
        alone = m.logits([short], 3)[0]
        padded = m.logits([short, long], 3)[0]
        other = m.logits([long], 3)[0]
    assert torch.allclose(alone, padded, atol=1e-4)
    assert not torch.allclose(alone, other, atol=1e-3)


def test_lone_full_attention_reads_later_tokens():
    m = tiny()

    def hidden(ids):
        return m.lm(torch.tensor([ids]), d1torch.masks_for(m.lm.cfg, torch.tensor([len(ids)]),
                                                           torch.full((1, len(ids)), -1), torch.ones(1, len(ids)).bool()))[0]

    with torch.no_grad():
        a, b = hidden(list(range(3, 15))), hidden(list(range(3, 14)) + [40])
    assert not torch.allclose(a[0], b[0], atol=1e-4)  # the first token hears the last


def test_temperature_divides_the_scores():
    seqs = [list(range(3, 20))]
    with torch.no_grad():
        assert torch.allclose(tiny(temperature=2.0).logits(seqs, 3), tiny().logits(seqs, 3) / 2, atol=1e-5)


def test_post_lone_and_packed():
    body = d3.request(T, S, 1, "both")
    lone, packed = tiny().post(body), tiny(pack=True).post(body)
    assert list(lone["answers"]) == ["fwd", "rev"] and list(packed["answers"]) == ["fwd", "rev"]
    for q, spec in body["questions"].items():
        assert list(lone["answers"][q]["probabilities"]) == list(spec["criteria"])
        assert sum(lone["answers"][q]["probabilities"].values()) == pytest.approx(1.0)
    assert packed["usage"]["input_tokens"] < lone["usage"]["input_tokens"]  # the state is read once
    one = d3.request(T, S, 1, "fwd")  # one question packed is the lone prompt
    a = tiny().post(one)["answers"]["fwd"]["probabilities"]
    b = tiny(pack=True).post(one)["answers"]["fwd"]["probabilities"]
    assert all(a[o] == pytest.approx(b[o], abs=1e-5) for o in brain.OPTIONS)


def test_a_merged_adapter_scores_like_the_lora_model(tmp_path):
    from peft import LoraConfig, get_peft_model

    lora = tiny()
    peft_model = get_peft_model(lora.lm, LoraConfig(**finetune_d1.LORA)).eval()
    torch.manual_seed(1)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.5)
    peft_model.save_pretrained(tmp_path)
    seqs = [list(range(3, 20)), list(range(7, 15))]
    with torch.no_grad():
        want, base = lora.logits(seqs, 3), tiny().logits(seqs, 3)
        merged = tiny().merge_adapter(tmp_path).logits(seqs, 3)
    assert torch.allclose(merged, want, atol=1e-4)
    assert not torch.allclose(merged, base, atol=1e-3)


def test_training_runs_under_checkpointing(tmp_path):
    from fraud.finetune_d3 import train

    m = tiny()
    enc = [(list(range(3, 20)), 0), (list(range(9, 25)), 2)] * 16
    out = train(m, enc, tmp_path, 12, log=lambda _: None, lora=finetune_d1.LORA)
    assert out["steps_done"] == 12 and (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert any("lora_" in n and p.grad is None for n, p in m.lm.named_parameters())  # zero_grad(set_to_none) after the last step


def test_plan_fills_the_hour_up_to_an_epoch():
    assert finetune_d1.plan_steps(5.0, 60.0) == 696
    assert finetune_d1.plan_steps(1.0, 60.0) == 1510
