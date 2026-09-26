# local-system-one

A local proof of concept of a **System One-style decision engine**, inspired by Jev, running on an
ordinary open-weight LLM.

**This is not Jev and does not reproduce Jev.** Jev's model, architecture and training are not
public. This project reproduces only the publicly described *inference pattern*: answer
multiple-choice decisions with one forward pass, reading the next-token logits of the option
labels. All inference runs locally. No remote inference API is called.

Sources, and which claims are documented, observed, inferred or our own choices:
[`docs/research.md`](docs/research.md). Machine inspection and runtime choice:
[`docs/machine.md`](docs/machine.md).

## How it works

1. A decision is `(state, instruction, options)`. The engine renders one chat prompt that lists
   the options with labels `A`, `B`, `C`, … (two-letter labels above 26). It ends with the
   assistant prefill `{"choice": "`.
2. It checks at runtime that each label is **exactly one distinct token at that position** by
   tokenising `prompt + label + '"}'` and requiring the prompt tokens as an exact prefix. A label
   that merges with the prefill, or two options sharing a token, raise `TokenMappingError`.
3. **One forward pass** (no generation) gives the next-token log-probs. Only the label tokens are
   read. The result has the chosen option, a softmax over the allowed tokens, and the
   probability mass *outside* the allowed tokens (a diagnostic).
4. Many independent decisions share one batched forward pass (left padding, attention mask,
   mask-derived `position_ids`, `logits_to_keep=1`).
5. `answer="text"` makes the model answer with the option text instead. If the options' first
   tokens differ, one token decides. Otherwise a slower **multi-token fallback** scores each full
   option sequence.
6. **Tiered goals** (`system_one/goals.py`): slow tiers choose goals from fixed lists, and faster
   tiers see those goals as context. **Tournament sampling** (`system_one/tournament.py`): large
   option sets are split into groups, and the winners are regrouped until one is left. Each
   round is one batch.

The probabilities are softmax values from an ordinary LLM. **They are not calibrated.** Measured:
all tested models are overconfident (see Known limitations).

```python
from system_one import Decision, load_config, make_engine

engine = make_engine(load_config("config/default.toml"))
r = engine.decide(Decision(
    instruction="Which team should handle this ticket?",
    options=("returns and exchanges", "shipping and delivery", "billing and payments"),
    state="My shoes arrived in the wrong size. Can I exchange them?",
))
print(r.choice, r.probs, r.outside_mass)          # one forward pass
results = engine.decide_batch([...])               # many decisions, one forward pass
```

## Run it on the Lenovo

Tested on Windows 11, Intel Core Ultra 7 256V, Arc 140V iGPU, 16 GB RAM, Python 3.13.

```bash
py -3.13 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

The first real run downloads `Qwen/Qwen2.5-1.5B-Instruct` (3.1 GB) from Hugging Face into the
standard HF cache. After that everything runs offline (set `HF_HUB_OFFLINE=1` to be sure).

```bash
.venv/Scripts/python -m system_one check
```

```bash
.venv/Scripts/python -m system_one decide --question "Which colour is the sky on a clear day?" --options red blue yellow
```

Tests. The `model` tests load Qwen2.5-0.5B-Instruct (1 GB) and skip when it is not in the HF
cache. Download it once with `.venv/Scripts/python -c "from huggingface_hub import snapshot_download;
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"`, or point them at another cached model with
`SYSTEM_ONE_TEST_MODEL=Qwen/Qwen2.5-1.5B-Instruct`.

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/python -m pytest -m "not model"
```

### The 2D demo

```bash
.venv/Scripts/python -m demo.sim
```

```bash
.venv/Scripts/python -m demo.sim --config config/mock.toml
```

```bash
.venv/Scripts/python -m demo.sim --headless --ticks 100
```

Four agents collect gems, eat food and avoid roaming hazards. Every tick, **all agents' due
decisions go through the engine in one batched forward pass**. Each agent has three tiers:

| tier | every | options |
|---|---|---|
| strategy | 12 ticks | collect gems / find food / avoid hazards / explore |
| target | 6 ticks, or when the target is reached or gone | every gem (24) or food item (12) on the map, safe spots, or regions. Sets larger than 8 use a **tournament**, one round per tick |
| action | every tick | move north / south / east / west / stay, each labelled with its outcome, e.g. `move west (target: 2 steps)`, `move north (wall)` |

The agent is **aware of its condition**. The strategy state spells out consequences ("about 20
ticks until starving"). A change in condition (an energy band, low health, an adjacent hazard)
makes the strategy re-decide on the next tick. Lower tiers see the target without its stale
distance. In a 4-seed × 250-tick comparison with the 3B, this took starvation from 2 of 4 runs to
0 of 4, halved "stuck" time and raised gems by 22%. `docs/research.md` → Observed → Agent
awareness has the diagnosis.

The window shows each agent's strategy and target with their probabilities, the tournament
progress, the action probabilities as bars, the outside-token mass, and per tick the batch size
and forward-pass latency. Agents glide between cells over about one tick; this is rendering
only. Space pauses, Esc quits. `--config config/mock.toml` runs without a model (random
decisions). `--no-goals` gives flat control (action tier only) for comparison.

`--plan-budget N` (default 1) lets each agent make at most N planning decisions (strategy,
target, tournament groups) per tick, next to its move. The rest wait for later ticks. This keeps
tick latency nearly constant. `-1` runs a whole tournament round per tick, which is how the
comparison below was measured.

Most responsive setup on the Lenovo, one agent with the 3B model:

```bash
.venv/Scripts/python -m demo.sim --config config/lenovo-3b.toml --agents 1
```

That runs at ~5 ticks/s: ~137 ms for a move-only tick and ~230 ms when a planning decision
rides along. Over 2 × 80 ticks the median was 194–195 ms, p90 236–238 ms and max 265–266 ms.
Without the budget, planning ticks took 550–880 ms.

`prefix_cache = true` (on in the Lenovo configs) reuses the keys/values of prompt prefixes that
repeat across ticks, which saves ~9–10% here. A 3B pass costs ~80 ms even for 8 tokens on this
GPU, so reuse cannot go much further. Prompt orders that reuse more (`prompt_order =
"question_first"` or `"options_first"`) were up to 21% faster but hurt accuracy, and the agent
played badly. Details: `docs/research.md` → Observed → Prefix caching.

## Benchmarks

Machine: Lenovo 83HM, Intel Core Ultra 7 256V, Intel Arc 140V iGPU (8 GB shared), 15.6 GB RAM,
Windows 11. Runtime: PyTorch 2.14.0+xpu, transformers 5.17.0. Model: Qwen2.5-1.5B-Instruct,
bfloat16, no quantisation. Median of 5 runs after warm-up. Raw data:
`bench/results/bench_Qwen2.5-1.5B-Instruct_bfloat16_20260926_050240.{json,md}`.

| prompt tokens | batch | sequential ms/decision | batched ms/decision | speed-up | batched decisions/s |
|---:|---:|---:|---:|---:|---:|
| ~165 | 1 | 64.2 | 66.1 | 1.0× | 15.1 |
| ~165 | 8 | 65.0 | 39.8 | 1.6× | 25.1 |
| ~165 | 32 | 65.2 | 36.3 | 1.8× | 27.5 |
| ~350 | 8 | 101.5 | 88.3 | 1.15× | 11.3 |
| ~350 | 32 | 101.0 | 77.0 | 1.3× | 13.0 |
| ~1120 | 8 | 272.1 | 270.2 | 1.0× | 3.7 |
| ~1120 | 32 | 267.7 | 297.9 | 0.9× | 3.4 |

Generating the answer as text instead (16 decisions, ~350-token prompts):

| method | batch | ms/decision |
|---|---:|---:|
| single-token read | 1 | 99.8 |
| generate `{"choice": "X"}` (7 tokens) | 1 | 358.0 |
| single-token read | 8 | 88.3 |
| generate `{"choice": "X"}` (7 tokens) | 8 | 127.4 |

On this iGPU, prompt processing is compute-bound. Batching therefore helps much less than on a
data-centre GPU: at most 1.8× here, and nothing for long prompts. The single-token read's
advantage over generation is 3.6× at batch 1 but only 1.4× at batch 8. Details and caveats:
[`docs/research.md`](docs/research.md) → Observed.

### Demo results

![demo window with Qwen2.5-1.5B on the Arc 140V](docs/demo.png)

With the real model on the Lenovo, the demo runs at ~3 ticks/s: 300–450 ms per tick for 6–9
decisions in one batch. Tiered goals vs flat control, 100 ticks, 4 agents, mean of 3 seeds
(`bench/results/demo_compare_20260926_051317.json`):

| setup | gems | food eaten | hazard hits | forward ms/tick (median) |
|---|---:|---:|---:|---:|
| model, tiered goals | 36.0 | 59.3 | 10.7 | 312 |
| model, flat (action tier only) | 15.7 | 11.0 | 5.7 | 198 |
| random decisions (mock), tiered goals | 17.7 | 10.7 | 8.3 | 3 |

Flat control was no better than random. Tiered goals gave 2.3× the gems and 5.4× the food, but
also more hazard hits. Tournament vs one full decision (1.5B; 30 problems each; one correct option):

| options | one full decision | tournament, groups ≤10 |
|---:|---:|---:|
| 10 | 93% | 93% |
| 26 | 87% | 93% |
| 40 | 47% | 100% |
| 80 | 17% | 80% |

Reproduce:

```bash
.venv/Scripts/python -m bench.bench --gen-baseline
```

```bash
.venv/Scripts/python -m bench.model_eval --models Qwen/Qwen2.5-1.5B-Instruct
```

```bash
.venv/Scripts/python -m bench.tournament_compare
```

```bash
.venv/Scripts/python -m bench.demo_compare
```

## Switch model and runtime (NUC 12)

Only the config changes. There are no code changes.

1. Install the PyTorch build for the NUC's hardware (see `requirements.txt`: the `xpu` index for
   Intel Arc/Iris, a CUDA index such as `cu128` for NVIDIA (check pytorch.org for the current
   one), `cpu` otherwise), then `pip install -r requirements.txt`.
2. Copy `config/nuc.example.toml` to `config/nuc.toml`. Set `model` (any decoder-only Hugging Face
   chat model), `device` (`auto`, `cuda`, `xpu` or `cpu`), `dtype` and `max_batch`.
3. Select it per command with `--config config/nuc.toml`, or for the whole shell with the
   environment variable `SYSTEM_ONE_CONFIG=config/nuc.toml`.
4. Verify it: `python -m system_one --config config/nuc.toml check`. This checks full-vocabulary
   logits, single-token labels for the new tokenizer, and batched vs sequential agreement. It
   also prints how many of 6 easy direction questions the model gets right. That score is
   informational (the 1.5B gets 4/6; it mixes up east and west) and does not affect OK/FAILED.

Adding a different runtime (for example llama.cpp or OpenVINO) means one class that implements
`system_one.backends.base.Backend` (tokenize, chat template, `next_token_logits` for a batch)
and one entry in `BACKENDS` in `system_one/config.py`. The engine, goals, tournaments and demo
do not change.

## Layout

```
system_one/            engine.py (decisions), goals.py (tiers), tournament.py, config.py
system_one/backends/   base.py (interface), hf.py (PyTorch + transformers), mock.py (no model)
demo/                  world.py (grid world), brain.py (tiers + state text), sim.py (window / headless)
bench/                 bench.py, model_eval.py, tournament_compare.py, demo_compare.py, results/
config/                default.toml (Lenovo), lenovo-3b.toml, nuc.example.toml, mock.toml
docs/                  research.md (sources + Observed results), machine.md
tests/                 pytest suite (mock tests always; `model` tests when weights are cached)
```

## Known limitations

- **Not calibrated.** Probabilities are softmax values renormalised over the allowed tokens.
  On a 58-item check, Qwen3-1.7B averaged 0.99 confidence at 57% accuracy, and the chosen 1.5B
  0.77 at 66%. Do not branch on them as if they were calibrated.
- **Small models make poor goal decisions.** No model that fits this machine met our accuracy
  rule. All fail rule-based goal selection (30–40%). The 1.5B strongly prefers "find food" and
  is also sensitive to option order. S1 describes this limitation: one forward pass cannot
  derive goals. Tiered goals structure the problem; they do not make the model smarter.
- **Accuracy falls fast with many options in one decision** (1.5B: 47% at 40 options, 17% at 80).
  Use tournaments; they cost about 2× latency.
- **The prefill is not neutral.** Reading the next token after `{"choice": "` agreed with the
  model's own unprefilled bare-label answer only ~50% of the time on ambiguous decisions.
- **Probability resolution**: with a bf16 LM head, close options can tie exactly (logits are
  rounded to ~0.125 steps). The default config computes the head in float32 (`head_dtype`,
  +0.9 GB); the benchmark tables above were measured with the bf16 head.
- **Batching gains are small on this iGPU** (≤1.8×). Long prompts in large batches were slower
  than sequential. bf16 batching changes probabilities by up to ~0.07 versus sequential and can
  flip near-ties. float32 is exact but ~7× slower on this GPU.
- **Latency is not constant**: it grows with total prompt tokens in the batch. A demo tick that
  also re-plans (long target lists) takes 2–3× longer than an action-only tick. The first forward
  pass in a new process takes 1.5–4 s on the XPU (kernel compilation). The demo and benchmarks
  warm up first; a one-off `decide` from the command line pays it.
- **Prefix caching gains are small** (~9–10% on demo moves) with the prompt order that keeps
  accuracy. Orders that put the question or options first reuse more but lost accuracy
  (3B navigation 94% → 62–81%). The cache needs full-attention models; it turns itself off
  otherwise.
- Not implemented: Score and yes/no question types, quantised backends, training or calibration
  of any kind.
- The eval sets are small (58 items; 30 problems per tournament cell). Treat the accuracy
  numbers as smoke tests.
- **Averages hide traps.** Single decisions were 97–100% correct in a move probe. But a
  deterministic model repeats a mistake every time it returns to the same state, so an agent can
  loop forever. Test agents in closed loop, not only per decision. Hazard avoidance by the
  strategy tier is still weak (11–44% in the probe).
- The demo state text gives relative offsets and names conditions in words, because the model
  cannot do the arithmetic in one pass. That is part of the demo design, not of the engine.

## Sources

- Sean Goedecke, [Two techniques for working with System One models](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/) (primary)
- Sean Goedecke, [Jev means structured output is interesting again](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/)
- Sean Goedecke, [System One models like Jev can train their own replacements](https://www.seangoedecke.com/system-one-models-can-train-their-own-replacements/)
- [sgoedecke/system-one](https://github.com/sgoedecke/system-one) (reference implementation linked from the primary source)
- TypeSafe, [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
