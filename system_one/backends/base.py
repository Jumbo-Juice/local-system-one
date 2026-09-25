"""The one interface between the decision engine and a model runtime.

A backend owns a tokenizer, a chat template and a forward pass. The engine never
generates text: it asks for next-token scores after a prompt, once per decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NextTokenScores:
    """Scores for one sequence at the next-token position.

    ``logprobs`` are log-softmax values over the FULL vocabulary, gathered at the
    requested candidate token ids (same order). ``top_id`` / ``top_logprob`` are the
    unconstrained argmax token, kept as a diagnostic.
    """

    logprobs: np.ndarray
    top_id: int
    top_logprob: float


class Backend(ABC):
    """Model + runtime behind the engine. Select one by config (see ``config.py``)."""

    #: Short runtime name for reports, e.g. "hf" or "mock".
    kind: str = "abstract"

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Width of the logits returned by :meth:`next_token_logits`."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenise text that already contains any chat-template special tokens."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode` for a list of ids."""

    @abstractmethod
    def render_chat(self, system: str, user: str) -> str:
        """Prompt text for one system + user message, ending where the assistant reply starts."""

    @abstractmethod
    def next_token_logits(self, batch: list[list[int]]) -> np.ndarray:
        """Full next-token logits for each sequence: float32 array of shape [B, vocab_size].

        Must run ONE forward pass per chunk of the batch (no generation loop).
        """

    def next_token_scores(
        self, batch: list[list[int]], candidates: list[list[int]]
    ) -> list[NextTokenScores]:
        """Log-probs of candidate tokens after each sequence.

        Default: computed on the host from :meth:`next_token_logits`. Backends override
        this to avoid copying the full vocabulary off the device.
        """
        logits = self.next_token_logits(batch).astype(np.float64)
        out = []
        for row, cand in zip(logits, candidates):
            logz = _logsumexp(row)
            top = int(row.argmax())
            out.append(
                NextTokenScores(
                    logprobs=row[np.asarray(cand, dtype=np.int64)] - logz,
                    top_id=top,
                    top_logprob=float(row[top] - logz),
                )
            )
        return out

    def continuation_logprobs(
        self, prefixes: list[list[int]], continuations: list[list[int]]
    ) -> list[np.ndarray]:
        """Per-token log-probs of each continuation given its prefix (teacher forcing).

        Default: one next-token query per continuation token, all in one batch.
        Backends override this with a single pass over ``prefix + continuation``.
        """
        seqs, targets, owners = [], [], []
        for i, (prefix, cont) in enumerate(zip(prefixes, continuations)):
            for j, tok in enumerate(cont):
                seqs.append(list(prefix) + list(cont[:j]))
                targets.append([tok])
                owners.append(i)
        scores = self.next_token_scores(seqs, targets) if seqs else []
        result: list[list[float]] = [[] for _ in prefixes]
        for owner, score in zip(owners, scores):
            result[owner].append(float(score.logprobs[0]))
        return [np.asarray(r, dtype=np.float64) for r in result]

    def generate(self, batch: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Greedy autoregressive generation. Only used as a benchmark baseline."""
        raise NotImplementedError(f"{type(self).__name__} has no text-generation baseline")

    def info(self) -> dict:
        """Model / runtime details for benchmark records."""
        return {"kind": self.kind}


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))
