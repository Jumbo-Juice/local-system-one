"""PyTorch + Hugging Face transformers backend (CPU, Intel XPU or CUDA).

Batches are left-padded. Position ids come from the attention mask, so a padded row
sees the same positions as the unpadded sequence. ``logits_to_keep`` limits the
LM head to the positions we read, so the full [B, T, V] logit tensor is never built.
"""

from __future__ import annotations

import platform

import numpy as np

from .base import Backend, NextTokenScores


class _Float32Head:
    """Replacement LM head that projects (bf16/fp16) hidden states with a float32 weight copy.

    In bf16 the head's output logits are rounded to steps of ~0.125 at typical magnitudes, so close
    options can tie exactly. The transformer layers are unchanged.
    """

    @staticmethod
    def build(torch, old):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(old.weight.detach().float(), requires_grad=False)
                bias = getattr(old, "bias", None)
                self.bias = None if bias is None else torch.nn.Parameter(bias.detach().float(), requires_grad=False)

            def forward(self, hidden):
                return torch.nn.functional.linear(hidden.float(), self.weight, self.bias)

        return Head()


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class HFBackend(Backend):
    kind = "hf"

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        max_batch: int = 32,
        revision: str | None = None,
        attn_implementation: str | None = None,
        head_dtype: str = "model",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model
        self.device = resolve_device(device)
        if dtype == "auto":
            dtype = "float32" if self.device == "cpu" else "bfloat16"
        self.dtype_name = dtype
        self.max_batch = max_batch

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        kwargs = {"dtype": getattr(torch, dtype)}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, **kwargs)
        self.model.to(self.device).eval()
        if self.model.config.is_encoder_decoder:
            raise ValueError("HFBackend needs a decoder-only causal LM")
        if head_dtype not in ("model", "float32"):
            raise ValueError("head_dtype must be 'model' or 'float32'")
        self.head_dtype = head_dtype
        # Count before any head swap: a float32 head copy unties tied embeddings and would
        # otherwise be counted as extra model parameters.
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        if head_dtype == "float32" and dtype != "float32":
            head = _Float32Head.build(torch, self.model.get_output_embeddings()).to(self.device)
            self.model.set_output_embeddings(head)

        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        self.pad_id = int(pad if pad is not None else 0)
        self._vocab = int(self.model.get_output_embeddings().weight.shape[0])

    # -- tokenizer ---------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)

    def render_chat(self, system: str, user: str) -> str:
        tok = self.tokenizer
        if not tok.chat_template:
            return f"{system}\n\nUser:\n{user}\n\nAssistant:\n"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:  # some templates reject a system role
            merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return tok.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

    # -- forward passes ----------------------------------------------------------

    def _chunks(self, n: int):
        for start in range(0, n, self.max_batch):
            yield start, min(n, start + self.max_batch)

    def _pad_left(self, seqs: list[list[int]]):
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            if not s:
                raise ValueError("empty sequence in batch")
            ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        return ids.to(self.device), mask.to(self.device), position_ids.to(self.device)

    def _forward_last(self, seqs: list[list[int]], keep: int):
        """Logits for the last ``keep`` positions of a left-padded batch: [B, keep, V]."""
        ids, mask, pos = self._pad_left(seqs)
        out = self.model(
            input_ids=ids, attention_mask=mask, position_ids=pos,
            use_cache=False, logits_to_keep=keep,
        )
        return out.logits

    def next_token_logits(self, batch: list[list[int]]) -> np.ndarray:
        parts = []
        with self.torch.inference_mode():
            for a, b in self._chunks(len(batch)):
                logits = self._forward_last(batch[a:b], keep=1)[:, -1, :]
                parts.append(logits.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    def next_token_scores(self, batch, candidates):
        torch = self.torch
        out: list[NextTokenScores] = []
        with torch.inference_mode():
            for a, b in self._chunks(len(batch)):
                logits = self._forward_last(batch[a:b], keep=1)[:, -1, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                top_lp, top_id = logp.max(dim=-1)
                cands = candidates[a:b]
                width = max(len(c) for c in cands)
                index = torch.zeros((len(cands), width), dtype=torch.long)
                for i, c in enumerate(cands):
                    index[i, : len(c)] = torch.tensor(c, dtype=torch.long)
                gathered = logp.gather(1, index.to(self.device)).cpu().numpy().astype(np.float64)
                top_lp, top_id = top_lp.cpu().numpy(), top_id.cpu().numpy()
                for i, c in enumerate(cands):
                    out.append(NextTokenScores(gathered[i, : len(c)], int(top_id[i]), float(top_lp[i])))
        return out

    def continuation_logprobs(self, prefixes, continuations):
        torch = self.torch
        seqs = [list(p) + list(c) for p, c in zip(prefixes, continuations)]
        result: list[np.ndarray] = []
        with torch.inference_mode():
            for a, b in self._chunks(len(seqs)):
                conts = continuations[a:b]
                max_len = max(len(c) for c in conts)
                logits = self._forward_last(seqs[a:b], keep=max_len + 1).float()
                logp = torch.log_softmax(logits, dim=-1)
                for i, cont in enumerate(conts):
                    # token j of a continuation of length L is predicted at kept index max_len - L + j
                    start = max_len - len(cont)
                    rows = logp[i, start : start + len(cont), :]
                    tgt = torch.tensor(cont, dtype=torch.long, device=self.device)
                    result.append(rows.gather(1, tgt[:, None])[:, 0].cpu().numpy().astype(np.float64))
        return result

    def generate(self, batch, max_new_tokens):
        torch = self.torch
        out: list[list[int]] = []
        with torch.inference_mode():
            for a, b in self._chunks(len(batch)):
                ids, mask, _ = self._pad_left(batch[a:b])
                gen = self.model.generate(
                    input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=self.pad_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                for row in gen[:, ids.shape[1]:].cpu().tolist():
                    out.append([t for t in row if t != self.pad_id])
        return out

    def info(self) -> dict:
        import torch
        import transformers

        device_name = platform.processor() or "cpu"
        if self.device.startswith("cuda"):
            device_name = torch.cuda.get_device_name(0)
        elif self.device.startswith("xpu"):
            device_name = torch.xpu.get_device_name(0)
        return {
            "kind": self.kind,
            "model": self.model_name,
            "commit": getattr(self.model.config, "_commit_hash", None),
            "parameters": self.n_parameters,
            "dtype": self.dtype_name,
            "head_dtype": self.head_dtype,
            "quantisation": "none (weights in %s)" % self.dtype_name,
            "device": self.device,
            "device_name": device_name,
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "max_batch": self.max_batch,
        }
