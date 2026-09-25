"""A model-free backend for tests and for running the demo without weights.

The tokenizer is a small regex tokenizer with a growing vocabulary. The default
"model" returns deterministic pseudo-random logits seeded by the token sequence,
so its decisions are arbitrary but repeatable. Tests can pass ``logit_fn`` to
script exact logits.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Callable

import numpy as np

from .base import Backend

SPECIAL = ["<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_SPECIAL_RE = "|".join(re.escape(s) for s in SPECIAL)
# A leading space attaches to the following word or symbol, like GPT-style BPE pre-tokenizers.
_PIECE = re.compile(rf"{_SPECIAL_RE}|\s?[A-Za-z]+|\s?\d|\s?[^\sA-Za-z\d]|\s+")


class MockBackend(Backend):
    kind = "mock"

    def __init__(
        self,
        vocab_size: int = 32768,
        seed: int = 0,
        logit_fn: Callable[[list[int]], np.ndarray] | None = None,
        delay_ms: float = 0.0,
        delay_per_seq_ms: float = 0.0,
        max_batch: int = 64,
    ):
        self._vocab_size = vocab_size
        self.seed = seed
        self.logit_fn = logit_fn
        self.delay_ms = delay_ms
        self.delay_per_seq_ms = delay_per_seq_ms
        self.max_batch = max_batch
        self._ids: dict[str, int] = {s: i for i, s in enumerate(SPECIAL)}
        self._pieces: list[str] = list(SPECIAL)
        self.forward_calls = 0  # number of batched forward passes, for tests

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def token_id(self, piece: str) -> int:
        if piece not in self._ids:
            if len(self._pieces) >= self._vocab_size:
                raise ValueError("mock vocabulary is full")
            self._ids[piece] = len(self._pieces)
            self._pieces.append(piece)
        return self._ids[piece]

    def encode(self, text: str) -> list[int]:
        pieces = _PIECE.findall(text)
        if "".join(pieces) != text:
            raise ValueError("mock tokenizer could not cover the text")
        return [self.token_id(p) for p in pieces]

    def decode(self, ids: list[int]) -> str:
        # Ids never produced by encode() have no text yet; random logits can still pick them.
        return "".join(self._pieces[i] if i < len(self._pieces) else f"<unused{i}>" for i in ids)

    def render_chat(self, system: str, user: str) -> str:
        return f"<|system|>{system}<|end|>\n<|user|>{user}<|end|>\n<|assistant|>"

    def _logits(self, ids: list[int]) -> np.ndarray:
        if self.logit_fn is not None:
            return np.asarray(self.logit_fn(ids), dtype=np.float32)
        digest = hashlib.blake2b(np.asarray(ids, dtype=np.int64).tobytes(), digest_size=8)
        rng = np.random.default_rng([self.seed, int.from_bytes(digest.digest(), "little")])
        return (2.0 * rng.standard_normal(self._vocab_size)).astype(np.float32)

    def next_token_logits(self, batch: list[list[int]]) -> np.ndarray:
        rows = []
        for start in range(0, len(batch), self.max_batch):
            chunk = batch[start : start + self.max_batch]
            self.forward_calls += 1
            if self.delay_ms or self.delay_per_seq_ms:
                time.sleep((self.delay_ms + self.delay_per_seq_ms * len(chunk)) / 1000)
            rows.extend(self._logits(list(s)) for s in chunk)
        return np.stack(rows)

    def info(self) -> dict:
        return {"kind": self.kind, "model": "mock (seeded random logits)", "seed": self.seed,
                "vocab_size": self._vocab_size, "device": "cpu"}
