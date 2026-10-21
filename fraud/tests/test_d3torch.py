import pytest
import torch

from fraud import brain, d3, d3torch
from fraud.tests.test_brain_capture import S, T


class FakeTokenizer:
    """Characters as token ids; the chat template is plain concatenation."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return {"input_ids": [1 + ord(c) % 60 for c in text]}


def tiny(seed=0, layer_types=("full_attention", "linear_attention", "full_attention", "linear_attention")):
    """A random 4-layer Qwen3.5 text model: softmax attention first, so non-causal attention shows."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=list(layer_types))
    lm = Qwen3_5TextModel(cfg).float().eval()
    readout = torch.randn(d3torch.MAX_OPTIONS, 32)
    return d3torch.D3Torch(lm, readout, FakeTokenizer(), "cpu")


def test_left_padding_does_not_change_a_row():
    m = tiny()
    short, long = list(range(1, 12)), list(range(5, 30))
    with torch.no_grad():
        alone = m.last_hidden([short])[0]
        padded = m.last_hidden([short, long])[0]
        other = m.last_hidden([long])[0]
    assert torch.allclose(alone, padded, atol=1e-5)
    assert not torch.allclose(alone, other, atol=1e-3)


def test_softmax_layers_read_the_whole_prompt_not_causally():
    m = tiny()
    seq = list(range(3, 20))
    with torch.no_grad():
        ours = m.last_hidden([seq])[0]
        causal = m.lm(input_ids=torch.tensor([seq])).last_hidden_state[0, -1]
    assert not torch.allclose(ours, causal, atol=1e-3)


def test_a_merged_adapter_scores_like_the_lora_model(tmp_path):
    from peft import LoraConfig, get_peft_model

    from fraud.finetune_d3 import LORA

    lora = tiny()
    peft_model = get_peft_model(lora.lm, LoraConfig(**LORA)).eval()  # injected LoRA dropout starts in train mode
    torch.manual_seed(1)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.5)  # a trained-looking adapter (lora_B starts at zero)
    peft_model.save_pretrained(tmp_path)
    seqs = [list(range(3, 20)), list(range(7, 15))]
    with torch.no_grad():
        want = lora.logits(seqs, 3)
        base = tiny().logits(seqs, 3)
        merged = tiny().merge_adapter(tmp_path).logits(seqs, 3)
    assert torch.allclose(merged, want, atol=1e-4)
    assert not torch.allclose(merged, base, atol=1e-3)


def test_with_softmax_attention_only_in_the_last_layer_the_last_position_is_the_stock_one():
    # The last position reads every position in both modes, so here non-causal changes nothing at
    # it: the stock transformers forward (final norm, RoPE, DeltaNet masks) is the oracle.
    m = tiny(layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"))
    seq = list(range(3, 20))
    with torch.no_grad():
        ours = m.last_hidden([seq])[0]
        stock = m.lm(input_ids=torch.tensor([seq])).last_hidden_state[0, -1]
    assert torch.allclose(ours, stock, atol=1e-5)


def test_post_answers_each_question_over_its_own_options_only():
    m = tiny()
    m.readout[:3] = 0.0           # three options score equally ...
    m.readout[3:] *= 1000.0       # ... and readout rows past the option count must not matter
    body = d3.request(T, S, 1, "both")
    r = m.post(body)
    assert list(r["answers"]) == ["fwd", "rev"]
    for q, spec in body["questions"].items():
        probs = r["answers"][q]["probabilities"]
        assert list(probs) == list(spec["criteria"])
        assert all(p == pytest.approx(1 / 3) for p in probs.values())
    rendered = (f"<system>{d3torch.SYSTEM_PROMPT}<user>"
                f"{d3torch.user_prompt(body['state'], brain.QUESTION, brain.OPTIONS)}<assistant>")
    assert r["usage"]["input_tokens"] == 2 * len(rendered)


def test_user_turn_is_the_d3_contract():
    # The literal is written from d3_format.user_prompt (vllm-sr/d3-nano @ 462efc3662), not from our code.
    text = d3torch.user_prompt("a state", "Is it fraud?", ["no: approve it", "unsure", "yes: decline it"])
    assert text == ("State:\na state\n\nQuestion:\nIs it fraud?\n\nOptions:\n"
                    "A: no: approve it\nB: unsure\nC: yes: decline it\n\n"
                    "Reply with only the code of the best option.")


def test_post_maps_each_readout_row_to_its_own_option():
    m = tiny()
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None, "c": None}}}}
    with torch.no_grad():
        h = m.last_hidden([m.encode("s", "i", ["a", "b", "c"])])[0]
    m.readout[:3] = 0.0
    m.readout[2] = 50 * h / h.dot(h)   # code C scores 50, A and B score 0
    probs = m.post(body)["answers"]["q"]["probabilities"]
    assert probs["c"] > 0.999 and probs["a"] < 1e-3
