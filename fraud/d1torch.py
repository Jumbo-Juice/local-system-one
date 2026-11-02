"""d1-Omni-2B (LiquidAI) in our own torch code, so LoRA can train it (Phase 9).

transformers has no LFM3 class yet (Observed: none up to 5.22.1) and the repo's own loader is remote
code, which this project does not run. So this module re-implements, from reading the repo's source
(Documented, LiquidAI/d1-Omni-2B @ 431f1d667d, read 2026-11-02), what a text question needs:
- the LFM3 text layers of modeling_lfm3.py: gated short convolution, attention with QK RMSNorm and a
  sigmoid output gate, RoPE per layer type, sandwich norm; a checkpoint's text weights load by name;
- the prompt of prompt.py (state style json_only, no system turn, "<think></think>" lead);
- the masks of attention.py, mode "set": on full-attention layers the state sees every token, every
  token sees the state, a question token its own question; sliding layers stay causal within their
  window and convolutions stay causal, so a question packed after another hears it through them;
- the readout of prompt.py: an option scores max(logit of its code, logit of " code") at the
  question's last token, over the tied embeddings, divided by calibration.json's temperature.

``D1Torch.post`` answers a /v1/systemone body, so d3.D3Decider runs on it unchanged. ``pack=True``
packs a body's questions after one state, as llama-server does (the reproduction check);
``pack=False`` reads each question as its own lone prompt, the row the fine-tune trains on.

Files: fraud/models/d1-omni-hf/ (git-ignored), downloaded at REVISION, sha256 checked on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

D1_DIR = Path(__file__).resolve().parent / "models" / "d1-omni-hf"
REVISION = "431f1d667de5afcc9b0c58b2d9c170a7b38b358b"
SHA256 = {  # Hugging Face LFS metadata of REVISION
    "model.safetensors": "1553fff04c54377ff527d2016fc479425ef60a0b3bc7a2e61d32f80a75c32aaf",
    "tokenizer.json": "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44",
}
TEXT_PREFIX = "model.language_model."  # the text layers' keys in model.safetensors
CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = "QUESTION:\n"
LEAD = "<think></think>"


def prefix_text(bos: str, state: str) -> str:
    """Everything before the first question: prompt.prefix_text for a text state."""
    return f"{bos}<|im_start|>user\n{state}\n\n\n{HEADER}"


def suffix_text(instructions: str, options) -> str:
    """The question up to the answer slot: prompt.question_block + suffix_layout for a choice."""
    lines = "\n".join(f"{code} {text.replace('_', ' ')}" for code, text in zip(CODES, options))
    return f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n{LEAD}"


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(size)), eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype)


def rope_frequencies(cfg, kind: str) -> torch.Tensor:
    """Inverse frequencies with LFM3's zero tail: the rotated part follows the full head dimension's progression."""
    p = cfg.rope_parameters[kind]
    angles = int(p.get("partial_rotary_factor", 1.0) * cfg.head_dim) // 2
    inv = 1.0 / (p["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / cfg.head_dim))
    return torch.cat((inv, torch.zeros(cfg.head_dim // 2 - angles)))


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.heads, self.repeat, self.d = h, h // kv, d
        self.q_proj = nn.Linear(cfg.hidden_size, h * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.o_proj = nn.Linear(h * d, cfg.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d, cfg.norm_eps), RMSNorm(d, cfg.norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, h, bias=False)

    def forward(self, x, rope, mask):
        b, t, _ = x.shape
        shape = (b, t, -1, self.d)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = rope
        q_r = q.float() * cos + rotate_half(q.float()) * sin  # RoPE in float32, as the reference
        k_r = k.float() * cos + rotate_half(k.float()) * sin
        k_r, v = k_r.to(q.dtype).repeat_interleave(self.repeat, 1), v.repeat_interleave(self.repeat, 1)
        out = F.scaled_dot_product_attention(q_r.to(q.dtype), k_r, v, attn_mask=mask).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj((out * gate).reshape(b, t, -1))


class ShortConv(nn.Module):
    """The gated short convolution. Rows are right-padded and it is causal, so a pad never reaches a real token."""

    def __init__(self, cfg):
        super().__init__()
        n = cfg.hidden_size
        self.conv = nn.Conv1d(n, n, cfg.conv_L_cache, groups=n, bias=False, padding=cfg.conv_L_cache - 1)
        self.in_proj = nn.Linear(n, 3 * n, bias=False)
        self.out_proj = nn.Linear(n, n, bias=False)

    def forward(self, x):
        gate, out_gate, value = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
        mixed = self.conv(gate * value)[..., : x.shape[1]]
        return self.out_proj((out_gate * mixed).transpose(-1, -2))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "conv":
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg)
        self.feed_forward = MLP(cfg)
        for name in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
            setattr(self, name, RMSNorm(cfg.hidden_size, cfg.norm_eps))

    def forward(self, x, rope, mask):
        y = self.operator_norm(x)
        y = self.conv(y) if self.kind == "conv" else self.self_attn(y, rope, mask)
        x = x + self.post_operator_norm(y)
        return x + self.post_ffn_norm(self.feed_forward(self.ffn_norm(x)))


class Lfm3Text(nn.Module):
    """The LFM3 text model: token ids and masks per layer kind to the final-norm hidden states."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg.layer_types)
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        for kind in sorted(set(cfg.layer_types) - {"conv"}):
            self.register_buffer(f"{kind}_inv_freq", rope_frequencies(cfg, kind), persistent=False)
        self.checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing = True

    def rope(self, t: int, device) -> dict:
        out = {}
        for kind in set(self.cfg.layer_types) - {"conv"}:
            freqs = torch.arange(t, device=device, dtype=torch.float)[None, :, None] * getattr(self, f"{kind}_inv_freq")
            emb = torch.cat((freqs, freqs), dim=-1)
            out[kind] = (emb.cos().unsqueeze(1), emb.sin().unsqueeze(1))
        return out

    def forward(self, ids, masks):
        hidden, rope = self.embed_tokens(ids), self.rope(ids.shape[1], ids.device)
        for layer in self.layers:
            args = (hidden, rope.get(layer.kind), masks.get(layer.kind))
            if self.checkpointing and self.training:
                hidden = checkpoint(layer, *args, use_reentrant=False)
            else:
                hidden = layer(*args)
        return self.embedding_norm(hidden)


def masks_for(cfg, state, question, real) -> dict:
    """The attention masks per layer kind, [B, 1, T, T]: attention.Sees (mode "set") on full-attention layers,
    causal within the window on sliding ones. state [B]: state lengths; question [B, T]: each token's question,
    -1 on the state and on padding; real [B, T]: the tokens that are not padding."""
    t = real.shape[1]
    pos = torch.arange(t, device=real.device)
    q, k = pos[None, :, None], pos[None, None, :]
    s = state[:, None, None]
    full = (q < s) | (k < s) | (question[:, :, None] == question[:, None, :])
    sliding = (k <= q) & (k > q - cfg.sliding_window)
    return {"full_attention": (full & real[:, None, :])[:, None], "sliding_attention": sliding[:, None]}


class D1Torch:
    """The LFM3 text layers of d1-Omni-2B, its code readout, temperature and tokenizer."""

    def __init__(self, lm, tokenizer, device: str, temperature: float = 1.0, pack: bool = False):
        self.lm, self.tokenizer, self.device, self.temperature, self.pack = lm, tokenizer, device, temperature, pack
        self.bos = getattr(tokenizer, "bos_token", None) or ""
        self.code_ids = [self._code_tokens(code) for code in CODES[:8]]

    def _code_tokens(self, code: str) -> list[int]:
        """prompt.readout_ids for a choice: the code's token, and " code" when that is one token too."""
        own = self.tokenizer.encode(code, add_special_tokens=False)
        spaced = self.tokenizer.encode(f" {code}", add_special_tokens=False)
        return own + ([spaced[0]] if len(spaced) == 1 and spaced != own else [])

    @classmethod
    def load(cls, root: Path = D1_DIR, device: str = "xpu", check_sha: bool = True, pack: bool = False) -> "D1Torch":
        """Text weights only, BF16 (the vision tower, audio encoder and speech head are never read)."""
        from safetensors import safe_open
        from transformers import AutoTokenizer

        from .d3 import sha256

        if check_sha:
            for name, want in SHA256.items():
                if sha256(root / name) != want:
                    raise RuntimeError(f"{root / name}: sha256 differs from the pinned one (revision {REVISION})")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        cfg = SimpleNamespace(**config["text_config"])
        temperature = json.loads((root / "calibration.json").read_text(encoding="utf-8"))["default"]
        tokenizer = AutoTokenizer.from_pretrained(str(root))
        torch.manual_seed(0)
        with torch.device("meta"):
            lm = Lfm3Text(cfg)
        state = {}
        with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(TEXT_PREFIX):
                    state[key[len(TEXT_PREFIX):]] = f.get_tensor(key).to(torch.bfloat16)
        missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith("_inv_freq")]
        if missing or unexpected:
            raise ValueError(f"checkpoint keys differ: missing {missing[:5]}, unexpected {unexpected[:5]}")
        del state
        for kind in set(cfg.layer_types) - {"conv"}:
            setattr(lm, f"{kind}_inv_freq", rope_frequencies(cfg, kind))
        lm = lm.to(device).eval().requires_grad_(False)
        return cls(lm, tokenizer, device, temperature, pack)

    def info(self) -> dict:
        return {"kind": "d1-torch", "model": "d1-omni-2b", "revision": REVISION, "dtype": "bfloat16",
                "temperature": self.temperature, "device": str(self.device), "attention": "set",
                "pack": self.pack}

    def merge_adapter(self, adapter_dir) -> "D1Torch":
        """Fold a fraud.finetune_d1 LoRA adapter into the text layers (inference runs at base speed)."""
        from peft import PeftModel

        self.lm = PeftModel.from_pretrained(self.lm, str(adapter_dir)).merge_and_unload().eval()
        return self

    def encode(self, state: str, instructions: str, options) -> list[int]:
        """Token ids of the lone prompt, ending at the answer slot (attention.pack_set's first row)."""
        return self.tokenizer.encode(prefix_text(self.bos, state) + suffix_text(instructions, options),
                                     add_special_tokens=False)

    def _state_len(self, state: str, instructions: str, options, ids: list[int]) -> int:
        return len(ids) - len(self.tokenizer.encode(suffix_text(instructions, options), add_special_tokens=False))

    def _read(self, rows, n: int):
        """Tempered option logits, [slots, n]. rows: (ids, state length, each token's question, slot positions)."""
        width = max(len(r[0]) for r in rows)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.tensor([r[0] + [pad] * (width - len(r[0])) for r in rows], device=self.device)
        question = torch.tensor([r[2] + [-1] * (width - len(r[2])) for r in rows], device=self.device)
        real = torch.tensor([[1] * len(r[0]) + [0] * (width - len(r[0])) for r in rows], device=self.device).bool()
        state = torch.tensor([r[1] for r in rows], device=self.device)
        hidden = self.lm(ids, masks_for(self.lm.cfg, state, question, real))
        where = [(i, p) for i, r in enumerate(rows) for p in r[3]]
        slots = hidden[torch.tensor([w[0] for w in where], device=self.device),
                       torch.tensor([w[1] for w in where], device=self.device)].float()
        tokens = sorted({t for g in self.code_ids[:n] for t in g})
        at = {t: j for j, t in enumerate(tokens)}
        logits = slots @ self.lm.embed_tokens.weight[torch.tensor(tokens, device=self.device)].float().T
        scores = [logits[:, [at[t] for t in g]].max(-1).values for g in self.code_ids[:n]]
        return torch.stack(scores, dim=-1) / self.temperature

    def logits(self, seqs, n: int):
        """Option logits at the last token of each lone prompt, [B, n]. A lone prompt reads itself whole on the
        full-attention layers (attention.py: "a lone question reads exactly as under prompt")."""
        return self._read([(s, len(s), [-1] * len(s), [len(s) - 1]) for s in seqs], n)

    def post(self, body: dict) -> dict:
        """Answer a /v1/systemone body (d3.request: choice questions with the same options, no descriptions)."""
        qs = body["questions"]
        names = list(qs)
        opts = [list(qs[q]["criteria"]) for q in names]
        suffixes = [suffix_text(qs[q]["instructions"], o) for q, o in zip(names, opts)]
        first = self.encode(body["state"], qs[names[0]]["instructions"], opts[0])
        with torch.no_grad():
            if self.pack:
                state = self._state_len(body["state"], qs[names[0]]["instructions"], opts[0], first)
                ids, question, ends = list(first), [-1] * state + [1] * (len(first) - state), [len(first) - 1]
                for k, suffix in enumerate(suffixes[1:], 2):
                    own = self.tokenizer.encode(HEADER + suffix, add_special_tokens=False)
                    ids, question = ids + own, question + [k] * len(own)
                    ends.append(len(ids) - 1)
                rows, tokens = [(ids, state, question, ends)], len(ids)
            else:
                seqs = [self.encode(body["state"], qs[q]["instructions"], o) for q, o in zip(names, opts)]
                rows = [(s, len(s), [-1] * len(s), [len(s) - 1]) for s in seqs]
                tokens = sum(len(s) for s in seqs)
            probs = torch.softmax(self._read(rows, len(opts[0])), dim=-1).cpu().tolist()
        answers = {q: {"type": "choice", "probabilities": dict(zip(o, p))} for q, o, p in zip(names, opts, probs)}
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
