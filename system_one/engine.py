"""Single-token decisions read from next-token logits.

A decision is (state, instruction, options). The engine renders one chat prompt that
ends with an answer prefill such as ``{"choice": "``. It then reads the next-token
log-probs of the option labels only. One forward pass answers a whole batch of
decisions; there is no generation loop. See docs/research.md for sources.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backends.base import Backend

LABEL_SLOT = "<label>"
DEFAULT_TEMPLATE = '{"choice": "<label>"}'

SYSTEM_LABEL = (
    "You are a fast decision module. Read the state and the question, then choose exactly "
    "one option. Reply only with the label of that option."
)
SYSTEM_TEXT = (
    "You are a fast decision module. Read the state and the question, then choose exactly "
    "one option. Reply only with the option text, copied exactly."
)

LETTERS = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))
PAIRS = tuple(a + b for a in LETTERS for b in LETTERS)
MAX_OPTIONS = 255  # Jev's documented per-question limit [S5]; used here as our own cap.
_TAIL = 4  # prompt tokens that key the label-validation cache


class TokenMappingError(ValueError):
    """The options cannot be mapped to distinct single tokens in this prompt context."""


@dataclass(frozen=True)
class Decision:
    instruction: str
    options: tuple[str, ...]
    state: str = ""
    context: str = ""

    def __post_init__(self):
        opts = tuple(str(o) for o in self.options)
        object.__setattr__(self, "options", opts)
        if not opts:
            raise ValueError("a decision needs at least one option")
        seen = set()
        for o in opts:
            if o in seen:
                raise ValueError(f"duplicate option {o!r}")
            seen.add(o)


@dataclass
class DecisionResult:
    decision: Decision
    index: int
    probs: list[float]  # softmax over the allowed tokens (or sequences), in option order
    labels: list[str]  # label shown to the model per option; empty strings in text mode
    outside_mass: float  # full-vocabulary probability NOT on any allowed answer
    method: str  # "single_token" or "multi_token" (slower fallback)
    top_token: str | None  # unconstrained argmax token at the answer position
    prompt_tokens: int

    @property
    def choice(self) -> str:
        return self.decision.options[self.index]

    @property
    def prob(self) -> float:
        return self.probs[self.index]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(zip(self.decision.options, self.probs), key=lambda x: -x[1])


@dataclass
class _Prepared:
    pos: int
    decision: Decision
    ids: list[int]
    labels: list[str]
    method: str
    tokens: list[int] = field(default_factory=list)  # single-token candidates
    continuations: list[list[int]] = field(default_factory=list)  # multi-token sequences


class Engine:
    def __init__(
        self,
        backend: Backend,
        *,
        answer: str = "label",
        answer_template: str = DEFAULT_TEMPLATE,
        multi_token: str = "auto",
        system_prompt: str | None = None,
    ):
        if answer not in ("label", "text"):
            raise ValueError("answer must be 'label' or 'text'")
        if multi_token not in ("auto", "never", "always"):
            raise ValueError("multi_token must be 'auto', 'never' or 'always'")
        if answer == "label" and multi_token == "always":
            raise ValueError("multi_token='always' only applies to answer='text'")
        if answer_template.count(LABEL_SLOT) != 1:
            raise ValueError(f"answer_template must contain {LABEL_SLOT} exactly once")
        self.backend = backend
        self.answer = answer
        self.multi_token = multi_token
        self.system_prompt = system_prompt or (SYSTEM_LABEL if answer == "label" else SYSTEM_TEXT)
        prefill, self.suffix = answer_template.split(LABEL_SLOT)
        # BPE vocabularies attach a space to the NEXT token (" A"), so a trailing space in the
        # prefill moves into the answer token instead of ending the prompt.
        self.prefill = prefill.rstrip(" ")
        self.lead = prefill[len(self.prefill):]
        self._cache: dict = {}
        self._pair_pool: list[str] | None = None
        self.last_stats: dict = {}

    # -- prompts -------------------------------------------------------------------

    def render(self, d: Decision, labels: list[str]) -> str:
        parts = []
        if d.state:
            parts.append(f"State:\n{d.state}")
        if d.context:
            parts.append(f"Context:\n{d.context}")
        parts.append(f"Question: {d.instruction}")
        if self.answer == "label":
            parts.append("Options:\n" + "\n".join(f"{l}: {o}" for l, o in zip(labels, d.options)))
            parts.append("Reply with the label of one option.")
        else:
            parts.append("Options:\n" + "\n".join(f"- {o}" for o in d.options))
            parts.append("Reply with one option, copied exactly.")
        return self.backend.render_chat(self.system_prompt, "\n\n".join(parts)) + self.prefill

    def labels_for(self, n: int) -> list[str]:
        if n > MAX_OPTIONS:
            raise TokenMappingError(
                f"{n} options exceeds the {MAX_OPTIONS}-option limit; use tournament sampling"
            )
        if self.answer == "text":
            return [""] * n
        if n <= len(LETTERS):
            return list(LETTERS[:n])
        pool = self._pairs()
        if n > len(pool):
            raise TokenMappingError(
                f"only {len(pool)} two-letter labels are single tokens for this tokenizer; "
                f"{n} requested. Use tournament sampling."
            )
        return pool[:n]

    def _pairs(self) -> list[str]:
        """Two-letter labels that are single tokens after the prefill (validated once on a probe)."""
        if self._pair_pool is None:
            probe = Decision("probe", ("x",))
            text = self.render(probe, ["A"])
            ids = self.backend.encode(text)
            pool, used = [], set()
            for label in PAIRS:
                tok = self._single_token(text, ids, label)
                if tok is not None and tok not in used:
                    pool.append(label)
                    used.add(tok)
                if len(pool) == MAX_OPTIONS:
                    break
            self._pair_pool = pool
        return self._pair_pool

    # -- token mapping ---------------------------------------------------------------

    def _continuation(self, text: str, ids: list[int], answer: str) -> list[int]:
        """Token ids of ``answer`` + suffix exactly as the tokenizer splits them after this prompt."""
        full = self.backend.encode(text + self.lead + answer + self.suffix)
        if full[: len(ids)] != ids:
            raise TokenMappingError(
                f"answer {answer!r} merges with the end of the prompt {self.prefill[-12:]!r}; "
                "change answer_template"
            )
        return full[len(ids):]

    def _single_token(self, text: str, ids: list[int], label: str) -> int | None:
        try:
            rest = self._continuation(text, ids, label)
        except TokenMappingError:
            return None
        if rest and self.backend.decode([rest[0]]) == self.lead + label:
            return rest[0]
        return None

    def _map(self, d: Decision, text: str, ids: list[int], labels: list[str]) -> tuple[str, list, list]:
        """Validate the answer tokens for this prompt. Cached by the prompt's last tokens."""
        key = (tuple(ids[-_TAIL:]), tuple(labels) if self.answer == "label" else d.options)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self.answer == "label":
            tokens = []
            for label in labels:
                tok = self._single_token(text, ids, label)
                if tok is None:
                    raise TokenMappingError(
                        f"label {self.lead + label!r} is not one token after {self.prefill[-12:]!r}"
                    )
                tokens.append(tok)
            _require_distinct(self.backend, d.options, tokens)
            hit = ("single_token", tokens, [])
        else:
            conts = [self._continuation(text, ids, o) for o in d.options]
            firsts = [c[0] for c in conts]
            distinct = len(set(firsts)) == len(firsts)
            if self.multi_token == "always" or (not distinct and self.multi_token == "auto"):
                hit = ("multi_token", [], conts)
            else:
                _require_distinct(self.backend, d.options, firsts)
                hit = ("single_token", firsts, [])
        self._cache[key] = hit
        return hit

    def prepare(self, d: Decision, pos: int = 0) -> _Prepared:
        labels = self.labels_for(len(d.options))
        text = self.render(d, labels)
        ids = self.backend.encode(text)
        method, tokens, conts = self._map(d, text, ids, labels)
        return _Prepared(pos, d, ids, labels, method, tokens, conts)

    # -- decisions -------------------------------------------------------------------

    def decide(self, d: Decision) -> DecisionResult:
        return self.decide_batch([d])[0]

    def decide_sequential(self, decisions: list[Decision]) -> list[DecisionResult]:
        """Baseline: one forward pass per decision."""
        return [self.decide_batch([d])[0] for d in decisions]

    def decide_batch(self, decisions: list[Decision]) -> list[DecisionResult]:
        """All single-token decisions share one batched forward pass (chunked by max_batch).

        Multi-token fallbacks run in a second batched pass, one sequence per option.
        """
        t0 = time.perf_counter()
        preps = [self.prepare(d, i) for i, d in enumerate(decisions)]
        t1 = time.perf_counter()
        results: list[DecisionResult | None] = [None] * len(preps)

        single = [p for p in preps if p.method == "single_token"]
        if single:
            scores = self.backend.next_token_scores([p.ids for p in single], [p.tokens for p in single])
            for p, s in zip(single, scores):
                lp = np.asarray(s.logprobs, dtype=np.float64)
                results[p.pos] = DecisionResult(
                    decision=p.decision, index=int(np.argmax(lp)), probs=_softmax(lp),
                    labels=p.labels, outside_mass=_outside(lp), method="single_token",
                    top_token=self.backend.decode([s.top_id]), prompt_tokens=len(p.ids),
                )

        multi = [p for p in preps if p.method == "multi_token"]
        if multi:
            prefixes = [p.ids for p in multi for _ in p.continuations]
            conts = [c for p in multi for c in p.continuations]
            per_token = self.backend.continuation_logprobs(prefixes, conts)
            k = 0
            for p in multi:
                lp = np.array([per_token[k + j].sum() for j in range(len(p.continuations))])
                k += len(p.continuations)
                results[p.pos] = DecisionResult(
                    decision=p.decision, index=int(np.argmax(lp)), probs=_softmax(lp),
                    labels=p.labels, outside_mass=_outside(lp), method="multi_token",
                    top_token=None, prompt_tokens=len(p.ids),
                )

        t2 = time.perf_counter()
        self.last_stats = {
            "decisions": len(preps),
            "prepare_s": t1 - t0,
            "forward_s": t2 - t1,
            "total_s": t2 - t0,
            "multi_token": len(multi),
            "prompt_tokens": sum(len(p.ids) for p in preps),
        }
        return results  # type: ignore[return-value]


def _softmax(lp: np.ndarray) -> list[float]:
    w = np.exp(lp - lp.max())
    return (w / w.sum()).tolist()


def _outside(lp: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.exp(lp).sum()))


def _require_distinct(backend: Backend, options, tokens) -> None:
    first: dict[int, int] = {}
    for i, tok in enumerate(tokens):
        if tok in first:
            j = first[tok]
            raise TokenMappingError(
                f"options {options[j]!r} and {options[i]!r} both map to token {tok} "
                f"({backend.decode([tok])!r}); use labels or allow the multi-token fallback"
            )
        first[tok] = i
