# Research notes

Short, implementation-focused notes for this proof of concept. Every claim has a label:

- **Documented**: stated in a source (cited as [S1]..[S5], [R]).
- **Observed**: measured on this machine (see the "Observed" section at the end).
- **Inferred**: follows from the sources but is not stated in them.
- **Implementation choice**: our own decision.

This project is **not** Jev and does not reproduce Jev. It reproduces only the publicly described
*inference pattern* on an ordinary open-weight LLM.

## Sources

| Id | Source | Where read |
|----|--------|-----------|
| S1 (primary) | Sean Goedecke, [Two techniques for working with System One models](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/) | the user's local Obsidian vault (web clipping). Not included in this repository: the articles are copyrighted; follow the links |
| S2 | Sean Goedecke, [Jev means structured output is interesting again](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/) | same vault folder |
| S3 | Sean Goedecke, [System One models like Jev can train their own replacements](https://www.seangoedecke.com/system-one-models-can-train-their-own-replacements/) | same vault folder |
| S4 | [sgoedecke/system-one](https://github.com/sgoedecke/system-one) (linked from S1): `system_one/inference.py`, `demo/labels.py`, `demo/wikirace/run.py`, `demo/README.md` | GitHub `main`, read 2026-09-26 |
| S5 | TypeSafe, [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (linked from S1 and S2) | web, read 2026-09-26 |
| R (secondary) | `deep-research-report.md` in the repo root | local file |

R was supposed to be pasted into the prompt. It was not, so the copy in the repo was used. R mixes
facts with speculation. Its claims about Jev internals are treated as unverified (see "Unknown").

## What "System One" means in the sources

- **Documented [S2]:** the name comes from Kahneman's System One (fast) vs System Two (slow and
  reflective).
- **Documented [S1, S2]:** Jev takes a human-language prompt but outputs only *decisions*: answers to
  user-provided multiple-choice questions. S3 describes such models as fast, general-purpose
  classifiers.
- **Documented [S5]:** end-to-end latency of 70–500 ms; up to 255 choices per question.
- **Documented [S1]:** any LLM can be turned into a System One-style model without changing the model,
  as long as you can read the logits and prefill the prompt.

## Single-token decisions and logits

- **Documented [S2]:** instead of generating a structured answer token by token, prefill the
  response with `"choice": "` and generate a single token, restricted to the allowed choices. The
  model reads all input tokens in parallel, so one forward pass gives the answer.
- **Documented [S4]:** the reference code applies the chat template and appends the prefill
  `choice_index:`. Only the logits of the allowed tokens are kept, and a softmax is taken over them.
  It checks at start-up that each index is one token that decodes correctly after the prefill.
  It reports `confidence = 1 - normalised entropy`.
- **Documented [S1, footnote 3]:** indexes vs labels. Labels (an arbitrary token associated with each
  choice) worked much better than indexes on Wikiracing, but not on Doom. **Documented [S4]:** the label
  variant uses the first 100 two-letter upper-case strings (`AA`, `AB`, ...) that are single tokens.
- **Documented [S2, footnote 5]:** multi-token options are an open question. Suggestions include
  mapping them to single tokens, having the model output an index, or scoring only the first token.
- **How this differs from normal generation. Documented [S2]:** normal LLM output is autoregressive:
  one forward pass per output token, each conditioned on the last. A decision needs exactly one forward
  pass over the prompt, then one read of the next-token distribution at the last position. There is
  no sampling loop.
- **Inferred:** the returned probabilities are the model's next-token softmax, re-normalised over the
  allowed tokens. Nothing in the method makes them calibrated. S2 (footnote 6) doubts Jev's own
  calibration claim.
- **Implementation choice:** labels are validated by tokenising `prompt + label + closing text` and
  checking that the prompt tokens are an exact prefix and the label is exactly one token. S4 checks
  decoding only. A decode check can pass even when the tokenizer would merge the label with the
  character before or after it.
- **Implementation choice:** we also report the probability mass that falls *outside* the allowed
  tokens, as a diagnostic of how well the prompt constrains the model.

## Batching

- **Documented [S1, S2]:** many single-token decisions go into one forward pass through ordinary
  inference batching. S1 credits this for the approach's consistent speed.
- **Documented [S1 and its footnote 2]:** the Qwen3-8B Doom demo made 6–7 batched decisions per loop.
  The loop took ~500 ms on an RTX 4090 and ~190 ms on an H100. The tool-calling version made one
  decision every ~600 ms (also on the H100).
- **Documented [S4]:** left padding, `position_ids` from the cumulative attention mask, and
  `logits_to_keep=1`. Optional shared-prefix KV caching (`cache_prefix`) is recommended for more than
  3 questions.
- **Implementation choice:** we use the same padding scheme. Prefix caching is implemented differently
  from S4: a persistent LRU of prefixes reused *across* calls, not one shared prefix within a
  batch (see Observed → Prefix caching).

## Tiered goals

- **Documented [S1]:** passing only game inputs as choices did not work. The model held "shoot" 100%
  of the time and wandered. One ~200 ms forward pass is enough to react, but not enough to work out a
  short-term goal. The fix: periodically ask the model to choose from a *fixed* set of short-term
  goals, and put the chosen goal into the fast prompt.
- **Documented [S1]:** a proposed layered system: a strategic goal every ~10 s, a tactical subgoal
  every ~5 s based on it, specific targets every ~1 s, and an inner input loop every ~100 ms. S1
  suggests writing all possible goals down ahead of time rather than generating them with an LLM.
- **Documented [S4]:** the Doom demo plans with `goal`, then `target` (whose candidates depend on the
  *newly chosen* goal). Both are committed together, and it re-plans after 3 *applied* control
  inferences. This is a count of decisions, not a wall-clock timer.
- **Implementation choices** (`system_one/goals.py`): tier periods are counted in ticks, not
  seconds. A tier's options may depend on the goals above it, and its prompt context lists those
  goals. When a goal changes, every lower tier is re-decided on the next tick. All due decisions
  of all agents in a tick share one batch, so a new goal reaches lower tiers one tick later. A
  tier with more options than the tournament group size runs as a tournament, one round per
  tick, and keeps its old goal until the tournament ends. The demo uses three tiers (strategy,
  target, action) instead of S1's four.

## Tournament choice sampling

- **Documented [S1]:** a Wikipedia page can have more than 1000 links. S1's layer stopped working
  well after about 100 choices. For large sets Jev first scores options independently, then makes
  an explicit choice [S5]. S1 tried that with Qwen3-8B, and it failed: hundreds of links got the
  same top score.
- **Documented [S1]:** tournament sampling. S1 put about a hundred links into each choice, then
  ran a second pass over the winners. In S1's words: "Ordinary LLMs are way better at relative
  judgements than absolute ratings."
- **Documented [S4]:** groups are contiguous, at most `group_size` each, in the original order. The
  winners are regrouped recursively until one remains.
- **Inferred:** a tournament can eliminate the best option if a group mis-ranks it. Its final
  probabilities only cover the last round's options. They are not a distribution over all options.

## Stated limitations and edge cases

- **Documented [S1, S2]:** far less flexible than an LLM. No long-form output.
- **Documented [S2]:** no test-time compute, so quality is capped near non-reasoning LLMs.
- **Documented [S2]:** "can't hallucinate" is a semantic dodge. The model can still pick the wrong
  option.
- **Documented [S2, footnote 6]:** the calibration claim is unsupported. The probabilities may be plain
  logit probabilities.
- **Documented [S1, S2]:** multi-token options, the choice-count ceiling (~100 in S1's layer), and
  labels vs indexes are all open issues.
- **Documented [S1]:** latency depends on the hardware (4090 vs H100). The state must be converted to
  text for a text-only model.
- **Documented [S1, S3]:** a generic System One model is larger and slower than a task-specific
  classifier.
- **Documented [S2]:** a model fine-tuned only for structured output (such as Jev) will likely beat
  retrofitted open models.

## What is unknown about Jev

- **Documented [S1]:** how Jev works is not known. People guess diffusion, Transformer tweaks, or a
  new model type. **Documented [S5]:** TypeSafe says it built a new model architecture, a parallel
  sampler and a training method called Reinforcement Learning for Calibrated Decisions (RLCD). It
  gives no details on the architecture, model size, training data, or RLCD.
- **Unverified (from R only):** that "JEV" means "Joint Embedding Vectors"; per-question output heads;
  CLIP-style joint embedding of state and options; a 32k context window; a "Noul" question type;
  benchmark and pricing tables beyond S5. No primary source above states any of these.
- **Not used:** R's "OpenJev" per-option entailment scoring (one forward pass per option, then a
  softmax over entailment scores). It is a different technique from single-token decisions and is not
  used here.

## What this project reproduces, and what it does not

Reproduces (the inference pattern only):

- single-token choice decisions read from the next-token logits of an ordinary local LLM
- batching of independent decisions into one forward pass
- tiered goals
- tournament sampling for large choice sets

Does not reproduce:

- Jev's model, architecture, training, or RLCD
- calibrated probabilities
- TypeSafe's API or SDK
- Score or yes/no question types
- any remote inference

## Observed

Measured on the Lenovo (Core Ultra 7 256V, Arc 140V iGPU via PyTorch XPU; see
`docs/machine.md`). Raw data is in `bench/results/`.

### Logits and padding

- One forward pass with `logits_to_keep=1` returns full `[B, 151936]` next-token logits for Qwen2.5.
  No text is generated.
- Left-padded rows vs the same sequence run alone: max |Δlogit| ≈ 5e-5 in float32 (XPU and CPU).
  In bfloat16 it is up to 0.33 (kernel-shape precision noise); the argmax is the same.
- With transformers 5.17, *omitting* explicit `position_ids` also gave correct results for
  Qwen2.5 (same ~5e-5 error). The library seems to derive positions from the mask. We still
  pass them explicitly, which is safe across library versions and models.
- The first XPU forward pass in a process takes 1.5–4 s (kernel compilation). Later ones take
  tens of ms. Benchmarks and the demo warm up first.

### Probability resolution in bf16

- In bfloat16 the LM head's output logits are rounded to steps of ~0.125 at typical magnitudes
  (20–40). Options whose logits differ by less than a step **tie exactly**. That step is ~13% in
  relative probability. Example: in text mode, Qwen2.5-1.5B scored "move north" and "move south"
  exactly 0.500 / 0.500.
- **Implementation choice:** `head_dtype = "float32"` swaps in a float32 copy of the LM head only.
  The transformer layers stay bf16. On the 58-item eval set (1.5B) this removed both exact ties
  (the example became 0.501 / 0.499). Accuracy (66%) was unchanged, and latency was not
  measurably affected (58-decision batch: 3.2 s vs 3.5 s). It costs +0.93 GB of device memory.
  It does **not** reduce batched-vs-sequential drift (max |Δp| 0.061 vs 0.048); that comes from
  the bf16 layers.
- The Lenovo default config enables it. The 3B config does not (memory). The benchmark tables
  and the demo comparison below were measured before it was enabled, with the bf16 head.

### Tokenisation of labels (Qwen2.5 tokenizer)

- `{"choice": "` + `A` → tokens `[..., ' "', 'A', '"}']`. `Answer:` + ` A` → `[..., ':', ' A']`.
  Both give clean single-token labels.
- `choice_label:` + `A` → `[..., ':A']`: the letter **merges** with the colon, so `A` is not the
  natural next token. `choice_label:` + `AB` → `[..., ':', 'AB']` is fine. S4 only checks by
  decoding, which would not catch the single-letter case. Our in-context check rejects it
  (`tests/test_decisions.py::test_hf_colon_prefill_merges_single_letters`).

### Batching vs sequential

On the 58-item eval set (`bench/eval_set.py`), one batched pass vs 58 single passes:

| Model | dtype | same choice | max \|Δp\| | median \|Δp\| |
|-------|-------|-------------|-----------|--------------|
| Qwen2.5-0.5B-Instruct | float32 | 58/58 | < 1e-4 | < 1e-4 |
| Qwen2.5-0.5B-Instruct | bfloat16 | 56/58 | 0.065 | 0.010 |
| Qwen2.5-1.5B-Instruct | float32 | 58/58 | < 1e-4 | < 1e-4 |
| Qwen2.5-1.5B-Instruct | bfloat16 | 58/58 | 0.048 | 0.004 |

The two bf16 flips were near-ties in the sequential run (top-2 margins 0.056 and 0.000).
Batching is exact in float32. In bf16 it changes probabilities by a few percent.

### Model selection

Rule, fixed before running: *pick the smallest model with ≥ 80% accuracy in every category*
(general, navigation, goal, target) of `bench/eval_set.py` (58 items; a smoke test, not a
benchmark). Label mode, `{"choice": "<label>"}` template, bf16 on the Arc 140V.

| Model | general | navigation | goal (numeric rules) | target (nearest of 4–12) | goal in words* | mean chosen p |
|-------|--------:|-----------:|------:|-------:|------:|------:|
| Qwen2.5-0.5B-Instruct | 79% | 38% | 30% | 12% | 20% | 0.57 |
| Qwen2.5-1.5B-Instruct | 93% | 81% | 30% | 50% | 50% | 0.77 |
| Qwen3-1.7B (no thinking) | 93% | 38% | 30% | 75% | 50% | 0.99 |
| Qwen2.5-3B-Instruct | 86% | 94% | 40% | 75% | 40% | 0.97 |

\* `goal_words` was added after the first run and is not part of the rule. It gives the
same situations with the condition named in words ("Energy: LOW") and no rule list.

- **No model passes the rule.** Every model fails goal selection from numeric rules: the Qwen2.5
  models answer "find food" for all 10 items. This matches S1's report that one forward pass is
  not enough to derive a goal (its Doom model held "shoot" 100% of the time).
- Qwen2.5-3B in bf16 ran **out of XPU memory** at batch 32. It works with `max_batch = 8`.
- The `Answer: <label>` template scored the same or 1–3 items better than `{"choice": "` on every
  model. That is within noise on 58 items, so we keep the S2-documented JSON prefill.
- Text mode (the model writes the option text) with the multi-token fallback was the most
  accurate for the 3B (78%), but 2.4× slower than label mode.
- **Fallback choice (made after seeing the results, so it is an implementation choice, not the
  rule):** Qwen2.5-1.5B-Instruct as the Lenovo default. It is Apache-2.0, about 2× faster than
  the 3B, and fits batch 32. `config/lenovo-3b.toml` selects the 3B. Consequences for the demo:
  the state names conditions in words, targets are given as relative offsets, and goals come
  from a slower tier. These are the design responses S1 describes.

### Confidence vs accuracy (not calibration training, just a measurement)

Chosen-option probability vs correctness on the same 58 items (label mode):

| Model | mean chosen p | accuracy | ECE (5 bins) | items with p ≥ 0.9: accuracy |
|-------|------:|------:|------:|------:|
| Qwen2.5-0.5B-Instruct | 0.57 | 0.40 | 0.19 | 4 items: 1.00 |
| Qwen2.5-1.5B-Instruct | 0.77 | 0.66 | 0.11 | 22 items: 0.86 |
| Qwen3-1.7B | 0.99 | 0.57 | 0.42 | 56 items: 0.59 |
| Qwen2.5-3B-Instruct | 0.97 | 0.71 | 0.27 | 51 items: 0.75 |

All four are overconfident. Qwen3-1.7B is the worst case: 0.99 average confidence at 57%
accuracy. The engine's probabilities are softmax values, **not calibrated** probabilities. The
sample is small.

### Latency and throughput (step 4)

Qwen2.5-1.5B-Instruct, bf16, Arc 140V, torch 2.14 XPU, transformers 5.17. Median of 5 runs after
warm-up, on an otherwise idle machine. Raw data and the full table:
`bench/results/bench_Qwen2.5-1.5B-Instruct_bfloat16_20260926_050240.{json,md}`. An earlier run
overlapped a model download and was 10–15% slower; its files are kept but not used here.

| prompt tokens | batch | sequential ms/decision | batched ms/decision | batched speed-up |
|---:|---:|---:|---:|---:|
| ~165 | 1 | 64.2 | 66.1 | 1.0× |
| ~165 | 8 | 65.0 | 39.8 | 1.6× |
| ~165 | 32 | 65.2 | 36.3 | 1.8× |
| ~350 | 8 | 101.5 | 88.3 | 1.15× |
| ~350 | 32 | 101.0 | 77.0 | 1.3× |
| ~1120 | 8 | 272.1 | 270.2 | 1.0× |
| ~1120 | 32 | 267.7 | 297.9 | **0.9× (slower)** |

Run-to-run spread (min–max over the 5 runs) was mostly within ±3% of the median, up to +5% in a
few cells. There was one outlier run: 97.9 ms in the ~165-token, batch-1 cell (median 66.1 ms).

- **Surprising, negative result:** batching helps far less here than S1/S2 imply. On this iGPU,
  prompt processing is compute-bound. A batch of 32 × 165 tokens is ~16 TFLOP of matmuls, and bf16
  matmul peaks at ~20–28 TFLOPS (measured). So per-decision cost falls at most 1.8×. Throughput
  tops out at ~27 decisions/s for short prompts and ~13/s at ~350 tokens.
- For ~1100-token prompts, batch 32 was 10% *slower* per decision than running sequentially.
  **Inferred, not verified:** a padded batch needs an explicit attention mask, which may push SDPA
  onto a slower kernel than the unmasked causal path a single sequence can use.
- **Inferred:** batching mainly raises matmul efficiency (small matmuls ran at ~7 TFLOPS, large
  ones at ~25). A data-centre GPU has far more compute per byte of weights, so the gain there
  should be larger. Not measured.
- fp32 matmuls run at ~3.9 TFLOPS on this GPU (7× slower than bf16), so exact fp32 batching is
  expensive.

Text-generation baseline (same run; 16 decisions, ~350-token prompts, greedy, no prefill):

| answer | batch | ms/decision | new tokens | same choice as single-token |
|---|---:|---:|---:|---:|
| single-token read (prefill, one pass) | 1 | 99.8 | 0 | — |
| single-token read (prefill, one pass) | 8 | 88.3 | 0 | — |
| generate JSON `{"choice": "X"}` | 1 | 358.0 | 7 | 88% |
| generate JSON `{"choice": "X"}` | 8 | 127.4 | 7 | 88% |
| generate bare label | 1 | 147.2 | 2 | 50% |
| generate bare label | 8 | 85.2 | 2 | 56% |

- At batch 1, a single-token decision is 3.6× faster than generating the JSON answer (S2 reports
  2–3× with Qwen2.5-1.5B). At batch 8 the gap shrinks to 1.4×, because extra decode steps are cheap
  next to the compute-bound prefill.
- A bare-label reply agrees with the single-token read only ~50% of the time on these (ambiguous,
  synthetic) decisions. **The `{"choice": "` prefill changes the answer**; it is not a neutral
  read-out of what the model would have said.
- Batched bare-label generation (85.2 ms) was ~3% faster than the batched single-token read
  (88.3 ms), although it does strictly more work. Not explained; not investigated.

### Tournament sampling vs one full decision (step 6)

Task: "Which of these is a <category>?", with exactly one category member hidden among n−1
words from nine other categories (`bench/tournament_compare.py`). 30 problems per size,
Qwen2.5-1.5B bf16. A full decision uses letters up to 26 options and two-letter labels above.
Tournament groups are contiguous and near-equal in size; one batched pass per round. Raw data:
`bench/results/tournament_compare_*.json`.

| options | full decision | tournament, groups ≤10 | tournament, groups ≤26 | full ms | tournament(10) ms |
|---:|---:|---:|---:|---:|---:|
| 10 | 93% | 93% (1 round, identical) | — | 59 | 54 |
| 20 | 80% | 97% | — | 67 | 125 |
| 26 | 87% | 93% | — | 68 | 140 |
| 27 | 70% (two-letter labels) | 97% | — | 68 | 141 |
| 40 | 47% | 100% | 83% | 92 | 173 |
| 80 | 17% | 80% | 73% | 120 | 268 |

- **One full decision degrades quickly with the number of options** on this 1.5B model: already
  47% at 40 options and 17% at 80. S1 saw its layer stop working well after ~100 options
  with an 8B model.
- **Tournaments fix most of it**, as S1 reports: 100% at 40 options and 80% at 80 with groups
  of 10, at about 2× the latency (two rounds). Smaller groups did better than groups of 26.
- The switch from single letters to two-letter labels (26 → 27 options) coincides with a drop
  from 87% to 70%. With 30 problems per cell (±~8 points), this suggests but does not prove
  that two-letter labels hurt this model. S1 found labels better than indexes for many
  choices, but it did not compare single letters with letter pairs.
- Where the errors happen: with groups of 10 and up to 40 options, every tournament error was
  a round-1 loss (accuracy equals round-1 survival). At 80 options the answer survived round 1
  in 93% of problems but won the final of 8 in only 80% overall, so most errors came in the
  final. With groups of 26, most errors were round-1 losses (40 options: 87% survived, 83%
  won).

### Goal selection is sensitive to option order

Same 20 goal items (10 numeric-rule, 10 in-words), four orderings of the four goals
(Qwen2.5-1.5B). In two orderings the model picked "find food" in 16 and 18 of 20 items. With
"find food" first (label `A`), it picked label `B` ("collect gems") in 9 of 10 numeric-rule
items. In the fourth ordering its choices split between "find food", "collect gems" and
"explore". So the choice depends on both content and position. Accuracy ranged from 20% to 50%
across orderings.

### East/west confusion with terse prompts

With a bare state ("The target is 3 cells east of you.") and options north/south/east/west,
Qwen2.5-1.5B answered "west" for east targets (p≈0.77). With the options reordered it got both
east and west wrong. With the demo-style context ("You are at (5,5). North is up. ...") it chose
east, but only at p=0.47. The label mapping is correct: the probabilities follow the labels
consistently across orderings. `python -m system_one check` prints this as an informational
score (4/6 on the Lenovo). It is not a pass/fail criterion.

### Demo: tiered goals vs flat control (step 7)

`bench/demo_compare.py`: 100 ticks, 4 agents, seeds 0–2, Qwen2.5-1.5B bf16. "Flat" is the
action tier only: its prompt lists the nearest gem, food and hazard as offsets. "Random" is the
mock backend with tiered goals. Mean of 3 seeds
(`bench/results/demo_compare_20260926_051317.json`):

| setup | gems | food eaten | hazard hits | deaths | decisions/tick | forward ms/tick (median, p90) |
|---|---:|---:|---:|---:|---:|---:|
| model, tiered goals | 36.0 | 59.3 | 10.7 | 1.7 | 6.9 | 312, 472 |
| model, flat | 15.7 | 11.0 | 5.7 | 0.7 | 4.0 | 198, 202 |
| random decisions, tiered goals | 17.7 | 10.7 | 8.3 | 1.7 | 6.0 | 3, 6 |

- Flat control was **no better than random** at collecting gems and food. This matches S1's
  report that passing only the raw game inputs as choices did not work well.
- Tiered goals gave 2.3× the gems and 5.4× the food of flat control, at ~1.6× the tick latency.
  They also took **more hazard hits** (10.7 vs 5.7): purposeful movement walked into roaming
  hazards more often. Three seeds only.
- The strategy tier chose "find food" most of the time, consistent with the eval above.
- In the window, the real model ran at ~3 ticks/s (300–450 ms per tick with 6–9 decisions
  per batch).
- This comparison ran just before a wording fix in the demo state text ("1 steps" → "1 step").
  The results were not re-run after it.

### Smoother ticks: planning budget and warm-up (Qwen2.5-3B, one agent)

The demo with the 3B model and one agent, 60–80 headless ticks (forward pass per tick):

| setup | move-only ticks | ticks with planning | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| whole tournament round per tick, 1-decision warm-up | 177 ms | 266–880 ms (2–4 decisions) | — | — | 2.8 s |
| budget 1, 1-decision warm-up | 163 ms | 377 ms mean (2 decisions) | 226 | 286 | 2.8 s |
| budget 1, warm-up with padded long batch | ~160 ms | ~260 ms | 214–222 | 256–266 | 277–464 ms |

- **Implementation choice:** `plan_budget` (per agent per tick) in `step_all`. Tournaments can
  now be advanced one group at a time (`Tournament.pending(limit)`). A 24-option target takes 4
  ticks instead of 2, but no tick carries more than one planning decision.
- A new (batch, length) shape costs ~100–120 ms extra the first time it is seen (e.g. batch 1 at
  200 tokens: 262 ms, then 156 ms). The multi-second stalls came from the first *padded* batch
  and the first large activation allocation. A warm-up batch with prompts of different lengths
  (up to ~400 tokens) removed them.
- While fixing this we found that a strategy change did not cancel a target tournament that had
  started from the old strategy's options. Now a changed goal cancels tournaments in lower tiers.
- Unlike the 1.5B, the 3B chose "collect gems" with p≈1.0 in the window run (11 gems by
  tick 114). This was one run; it was not evaluated further.

### Prefix caching (reusing repeated prompt prefixes)

**Implementation choice** (`HFBackend(prefix_cache=True)`): the engine tells the backend how many
leading tokens of each prompt do not depend on the state or context. The backend keeps an LRU of
per-layer keys/values for such prefixes. A prefix is stored the second time it is seen, taken from
that pass's own cache, so there is no extra forward pass. Later calls feed only the suffix. Cached
prefixes of different lengths are left-padded in one batch, with a mask and mask-derived
positions. Only full-attention models are supported.

- **Correctness:** cached vs uncached logits agree to 9e-5 in float32 (Qwen2.5-0.5B, mixed batch:
  two rows sharing a prefix, one with another prefix, one without). In bf16 they agree to ≤0.4 with
  the same argmax, the same noise level as padding.
- **Pass cost has a floor.** Qwen2.5-3B, batch 1: 83 ms at 8 tokens, 99 ms at 64, 132 ms at 150,
  187 ms at 300. That is ~80 ms fixed plus ~0.35 ms per token, so skipping prefix tokens can save
  at most ~40% of a ~150-token decision on this GPU.
- **Prompt order decides how much can be reused**, and it changes accuracy (58-item eval, label mode):

| prompt order | reusable prefix | Qwen2.5-1.5B | Qwen2.5-3B | 3B navigation |
|---|---|---:|---:|---:|
| state first (default) | system prompt + chat header (~35 tokens) | 66% | 71% | 94% |
| options first | + the option list | 34% | 71% | 81% |
| question first | + question and options | 48% | 67% | 62% |

- **Demo, 3B, one agent, 80 ticks per setup:**

| setup | move-only tick | move + plan tick | gems in 80 ticks |
|---|---:|---:|---:|
| no cache | 151 ms | 256 ms | 9 |
| cache, state first | 138 ms | 230 ms | 9 |
| cache, options first | 131 ms | 206 ms | 2 |
| cache, question first | 119 ms | 165 ms | 0 |

- The orders that reuse more are faster, but the agent then plays badly (0–2 gems vs 9). This
  matches the accuracy drop. **We keep the state-first order with the cache on:** ~9–10% faster,
  with identical prompts. Final check with the demo's warm-up (which now also compiles the cached
  path): move-only ticks 137 ms, tick median 194–195 ms, p90 236–238 ms, max 265–266 ms, over two
  80-tick runs.
- Honest summary: on this iGPU the fixed per-pass cost dominates short decisions. Prefix reuse
  helps only a little unless the prompt is reordered, and reordering hurt these small models.
  Larger models may be less order-sensitive; not tested.

### Agent awareness: why the 3B agent starved, and the fix

Symptom (user report): with the 3B model and one agent, the agent ran out of energy and died.

- **Strategy tier ignored energy.** `bench/strategy_probe.py`: 36 demo-style states (energy 95 → 0,
  hazard far/adjacent, food near/far). Truth was fixed in advance. With the demo's wording
  ("Energy: LOW (24/100)"), the 3B chose "collect gems" in **36/36**, with P(find food) = 0.00
  even at energy 0. Spelling out consequences ("about 24 ticks until starving, then 5 health lost
  per tick") gave P(find food) of 0.38 / 0.73 / 0.85 / 0.95 / 1.00 at energy 34 / 25 / 15 / 5 / 0, and
  0.00 at 36 and above (food cases 80%, gem cases 100%). Adding a priority sentence to the
  question overcorrected: food in every state, even at 95 energy. Adjacent hazards were handled
  poorly by all variants (11–44%).
- **Move tier got trapped.** A tick-by-tick trace showed the agent, with strategy "find food",
  alternating between (8,0) and (8,1) for 38 ticks, one cell from its food at (7,0), until it
  starved. Cause: the goal context repeated the target's option text, "food at (7,0), 1 step
  away". That distance was measured when the target was chosen and had gone stale. With that
  context the 3B chose "move south" (p=0.55). Without the distance it chose "move west" (0.81),
  and with no target line at all "move west" (1.00).
- **Averages hide traps.** A 60-case single-decision probe (`bench/action_probe.py`) found 97–100%
  of moves approaching the target for every variant. But the model is deterministic, so one wrong
  answer in one spot repeats each time the agent comes back, and a 2-cell loop never ends. Only
  closed-loop runs show this.
- **Implementation choices** (on by default, `aware=True`): consequence wording in the strategy
  state; re-deciding the strategy on the next tick when the agent's condition changes band
  (energy 35/25/15/5/0, health < 40, hazard adjacent) instead of waiting up to 12 ticks; showing
  lower tiers the target without its stale distance (`Tier.describe`); and move options labelled
  with their outcome, e.g. `move west (reach the target)`, `move north (wall)`, `move east (HAZARD:
  -25 health)`, using true walking distances. The model still makes every decision. Precedent for
  the labels: S4's Doom demo fed its model an A*-computed waypoint bearing.
- **Closed-loop result** (`bench/survival_compare.py`, 3B, one agent, budget 1, 4 seeds × 250 ticks):

| setup | runs that starved | stuck ticks* | gems | food eaten | lowest energy | forward ms/tick (median) |
|---|---:|---:|---:|---:|---:|---:|
| old prompts | 2 of 4 | 41.5 | 26.2 | 6.0 | 0 (twice) | 206 |
| aware, plain move options | 0 of 4 | 37.8 | 25.2 | 8.0 | 2–25 | 188 |
| aware + labelled move options | 0 of 4 | 17.8 | 32.0 | 8.0 | 16–20 | 214 |

\* ticks spent in streaks of 6+ ticks without getting closer to the current target.

Four seeds is a small sample. The labels cost ~25 ms per tick (longer options). One of the
"aware, plain" runs came within 2 energy of starving; every labelled run kept at least 16.

### Dungeon demo: one agent in closed loop (Qwen2.5-3B)

A single-agent scenario built to be harder than the gem field: nine rooms joined by corridors, a
key that opens the exit, two enemies that chase, and gems, food and potions (`demo/dungeon/`).
The agent sees only the rooms it has visited. Every decision is recorded to JSONL and replayed in
`demo/dungeon/viewer.html` (README → The dungeon demo).

**Implementation choices, fixed before the first model run** (commit `e227283`):

- Rules: 400-tick limit; energy 100, −1 per tick, −5 health per tick at 0; food +40 energy,
  potions +40 health; enemies chase within 6 steps on two of every three ticks, hit for 30 and are
  then stunned for 3 ticks; one enemy lives in the key room. 12 gems, 3 food, 2 potions. The
  exit is in the room farthest from the start.
- The strategy tier is offered only goals that are possible now (no "get the key" before the key
  has been seen, no "go to the exit" without the key). An impossible goal would be a trap option,
  like a move into a wall. The model still ranks every possible goal.
- A tier with exactly one possible option is committed without a model call
  (`step_all(skip_single=True)`), and the replay says so. This saves a ~100 ms pass and does not
  present a forced p = 1.0 as a model judgement.
- Walking distances in the text are measured over the cells the agent knows.

**Development run (seed 100, excluded from the evaluation).** With the first demo's move wording
(`move west (target: 2 steps)`), the 3B walked back and forth between two cells for 90 ticks and
starved. In one doorway state it chose `move east (target: 4 steps)` at p = 0.98. Probing that
one recorded state:

| variant of the doorway prompt | choice |
|---|---|
| as recorded | east 0.98 (away from the target) |
| without the word "doorway" | west 0.68 |
| without the neighbour list | east 1.00 |
| options in reverse order | west 1.00 |
| outcomes worded `closer` / `farther`, same numbers | west 1.00 |
| plus "Your last moves: west, east, west, east." | east 1.00 (and broke the next state) |

One state is not evidence, so both wordings went into the evaluation. The `closer/farther`
wording became the demo default before it ran.

**Pre-registered evaluation** (`bench/dungeon_eval.py` at commit `2d959dd`, seeds 0–7, every run
reported). Raw data: `bench/results/dungeon_eval_20260926_163151.json`. The first invocation
finished the two 3B setups, then ran out of XPU memory loading the 1.5B (the 3B was still
referenced; fixed). A second invocation reused the finished 3B traces (the runs are
deterministic) and ran the 1.5B and random setups with identical prompts.

| setup (seeds 0–7) | escaped | died: enemy | died: starvation | key picked up | rooms seen | gems | hits | stuck ticks* | forward ms, median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B, `closer` wording (demo default) | 0 / 8 | 7 | 1 | 2 | 3.6 | 4.0 | 3.6 | 5.8 | 165 / 276 |
| 3B, `steps` wording (first demo) | 0 / 8 | 3 | 5 | 1 | 2.9 | 3.0 | 1.5 | 63.4 | 153 / 257 |
| 1.5B, `closer` wording | 0 / 8 | 7 | 1 | 1 | 3.0 | 3.0 | 3.8 | 19.4 | 82 / 135 |
| random decisions (mock) | 0 / 8 | 1 | 7 | 0 | 1.5 | 2.0 | 0.5 | 54.4 | 0.4 / 0.8 |

Means over the 8 runs, except counts. \* Ticks in streaks of 6+ ticks without getting closer to
the same target (as in the survival comparison above). Latency: per-run medians, then the median
over runs.

- **No run escaped, and no agent even saw the exit room** (the room farthest from the start). Runs
  ended after 64–117 ticks on average (3B `closer`: 84). The agents never got near the end game.
- **The wording fixed the loops and exposed the enemies.** On the 3B, `closer/farther` cut stuck
  ticks from 63.4 to 5.8 and starvation deaths from 5 to 1, but enemy deaths rose from 3 to 7.
  An agent that moves with purpose meets the enemies more often; the gem field showed the same
  with hazards (tiered vs flat control, above).
- **Most hits were chosen.** In the 3B `closer` runs, 21 of 29 hits came from the agent choosing
  the move labelled `ENEMY: -30 health` (median probability of the chosen move 0.999), usually
  when every other move was labelled `farther`. 14 of the 29 hits happened while the strategy was
  "flee the enemy". The 1.5B: 15 of 30 hits walking into the enemy, 12 standing next to one.
- The 3B's strategy tier spent 32% of ticks on "flee the enemy", 29% on "explore", 27% on
  "collect gems" and 6% on "get the key".
- Tournaments never ran in any evaluated run: no agent knew more than 8 gems at once before it
  died. The viewer's tournament display was checked only on a mock run with `--group-size 3`.
- Random decisions mostly starved near the start (1.5 rooms seen, 54 stuck ticks).
- The pre-registered replay (seed 0, 3B `closer`) dies to the enemy at tick 143, after 8 gems and
  5 rooms, without the key.

**Post-hoc change: enemy consequences in the move labels** (`enemy_aware`, after the results
above). The traces showed where the hits came from. In the 3B `closer` runs, 21 of 29 hits came from choosing the move labelled
`ENEMY: -30 health`. That label gives the damage but not that the agent stays where it is, so the
only move toward the target looked like progress. The change: a move into an
enemy reads `ENEMY: you stay here and lose 30 health`, a cell next to an enemy reads `next to an
enemy: it can hit you for 30 health`, and flee targets are only cells the agent reaches before any
visible enemy, by a path that never passes next to one. The earlier safe spots were ranked only by
distance from the enemy, and one lay beyond it, so fleeing walked into the enemy (seed 0). To check
the change is not fitted to seeds 0–7, it also ran on seeds 8–15, which had never been run.
Decision rule, set after seeing its seeds 0–7 and before the comparison runs on seeds 8–15 had
finished: make it the default only if it clearly beats the pre-registered wording on seeds 8–15.

| 3B setup | seeds | escaped | died: enemy | died: starvation | key picked up | rooms seen | hits | stuck ticks | forward ms, median / p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-registered wording | 0–7 | 0 / 8 | 7 | 1 | 2 | 3.6 | 3.6 | 5.8 | 165 / 276 |
| `enemy_aware` | 0–7 | 0 / 8 | 7 | 1 | 2 | 3.8 | 3.6 | 2.8 | 172 / 304 |
| pre-registered wording | 8–15 (new) | 0 / 8 | 5 | 3 | 0 | 3.5 | 3.6 | 4.2 | 170 / 314 |
| `enemy_aware` | 8–15 (new) | 0 / 8 | 6 | 2 | 0 | 3.5 | 3.4 | 3.2 | 171 / 309 |

Raw data: `bench/results/dungeon_eval_20260926_163345.json` (it reuses the pre-registered 3B
traces for seeds 0–7).

- **Why the wording did not help: the move tier follows the target.** Across the 3B runs on
  seeds 0–7, whenever some move led toward the target the model chose one 97% (pre-registered
  wording) and 98% (`enemy_aware`) of the time. In the decisions where every such move led next to
  or into an enemy and a safe move existed, it took the risky move every time (13 of 13, and 7 of
  7). Spelling out the consequence did not change that. Staying alive would need the planning
  tiers to choose targets away from the enemies, and they did not (for example, "get the key"
  while the key's guard stood next to it).
- A guess that the goal "flee the enemy" pulled the model toward options containing the word
  "enemy" was **not supported**: while fleeing, it chose enemy-mentioning moves in 32–34% of mixed
  decisions, below their 41–43% share of the open moves.
- **Not adopted.** Outcomes were the same with and without it on both seed ranges, so the demo
  default stays the pre-registered wording; `--enemy-aware` remains available. Eight runs per
  cell is a small sample: a difference of one or two deaths is noise.
- **Inferred, not tested:** better move labels alone will not fix this. The move tier follows the
  first half of its question ("which move brings you closer to your current target?") and ignores
  the second ("do not move into a wall or an enemy") when they conflict. The fix has to come from
  the target and strategy tiers choosing targets away from the enemies, or from a bigger model.
  The 3B is the largest model tested here; larger ones are planned for the NUC.

### Shooter demo: the dungeon redone as a room-clearing shooter (1.5B and 3B)

The dungeon above was never escaped. Its doorways were one cell wide, so a chasing enemy could
trap the agent in one, and the agent's only answer to an enemy was to walk away. The redo
(`demo/shooter/`) keeps one agent, the fog of war, the key and the exit, and changes the game into
a room-clearing shooter in the spirit of *Enter the Gungeon*. The first dungeon and its results
stay as they were (`demo/dungeon/`).

**Rules (implementation choices, frozen before any model run; commit `879a339`).**

- Nine rooms, 6–9 × 5–7 cells, in a 3 × 3 layout. Doorways and corridors are **three cells wide**.
- Every room except the start holds 1–2 enemies (the key room 3, the exit room 2). Each enemy is
  a gunner (3 HP) or, with probability 0.3, a brute (4 HP). Enemies wake when the agent steps
  into their room or shoots one of them. They never leave their room.
- **Rooms seal.** While the agent is inside a room with living enemies, its doorways are walls
  (for walking and for bullets) until every enemy in it is dead. Fights happen in open rooms.
- The agent's gun fires one bullet every 2 ticks at the enemy the agent chooses. The bullet flies
  3 cells per tick toward where the enemy stood and stops at the first wall or enemy, so a moving
  enemy can be missed. Gunners aim visibly for one tick, then fire a bullet that flies 1 cell per
  tick (15 damage); the agent can step out of its line. Brutes walk at the agent, hit for 20 when
  adjacent, then back off for 3 ticks. The agent has 100 health; 3 potions heal 40.
- No hunger (starvation was a failure mode of the first dungeon that had nothing to do with
  enemies).

**Beatable before any model ran (Observed; `bench/shooter_calibration.py`,
`bench/results/shooter_calibration_20260926_233244.json`).** A hand-written reference bot (BFS
plus "shoot the nearest visible enemy", seeing only what the prompts describe) set the ceiling.
Rule variants were compared on dev seeds 1000–1059 with four bots; none is the model.

| rule variant (60 dev seeds) | reference bot | dodges on half the ticks | never dodges | random |
|---|---:|---:|---:|---:|
| first draft (enemy bullets 20, brutes 25) | 60 | 47 | 15 | 0 |
| **enemy bullets 15, brutes 20 (frozen)** | **60** | **55** | **31** | **0** |
| draft + gunners fire every 6 ticks | 60 | 49 | 26 | 0 |
| bullets 15 + every 6 ticks | 60 | 55 | 43 | 0 |
| draft + 5 potions | 60 | 55 | 23 | 0 |
| draft + fewer enemy HP (2/3) | 60 | 57 | 53 | 0 |

Decision rule (implementation choice): keep the variant in which the reference bot always wins
and a bot that never dodges wins about half the time. Dodging decides survival: the same bot
without dodging lost 29 of 60 seeds.

**Decisions and text (implementation choices).**

| tier | every | options |
|---|---|---|
| strategy | 12 ticks, or at once when health band, fighting, sealed room, key, rooms seen or key reachability change | the goals possible now, from: explore, fight the enemies here, get the key, go to the exit, drink a health potion |
| target | 6 ticks, or when reached, gone or unsafe | unexplored rooms behind known doorways, the key, the exit, potions, or up to 5 **firing spots**: cells with a clear line to an enemy, ≥ 3 cells from every enemy, out of every bullet's path |
| move (control head) | every tick | the open moves and stay, e.g. `move west (safe; closer: 4 steps to the target)`, `stay (BULLET: -15 health)` |
| shoot (control head) | every tick | `shoot gunner #4, 3 cells east (AIMING at you; 3 hits to kill; clear line)` … and `hold fire`; only-option (no model call) while the gun reloads or no enemy is in sight |

- The two control heads run side by side in one batched pass, like the Doom demo's control
  heads [S4], and neither sees the other's answer (`Tier.context_from`).
- Moves into walls, sealed doorways and enemies are not offered: they would be trap options, like
  an impossible goal.
- "closer/farther" is measured along routes that avoid every cell an enemy can hurt next tick; a
  move into danger is labelled only with its damage.
- The move state gives the bearing of the route's next waypoint (the farthest route cell in a
  straight line), not of the target. Precedent: the Doom demo gave its model an A*-computed
  waypoint bearing [S4].

**Development on dev seeds 1000–1001 (Observed; excluded from the evaluation).** Each change
below came from a failure in a recorded run, checked by re-asking the recorded prompts.

1. *The shoot head's answer was the option order.* In the first 1.5B run (seed 1000) it chose
   "hold fire" in 34 of 34 decisions and died at tick 38 without firing. Re-asking the same 34
   prompts:

   | variant of the shoot prompt | 1.5B fires | 3B fires |
   |---|---:|---:|
   | as recorded (hold fire listed last) | 0 / 34 | 34 / 34 |
   | hold fire listed first | 34 / 34 | 34 / 34 |

   Asked only which of the two enemies to aim at, the 1.5B picked the nearest in 11 of 34 states
   when it was listed first and in 34 of 34 when it was listed last; the 3B in 23 and 26 of 34.
   The 3B fires whatever the order; the 1.5B's choice follows the order. Order mattered for the
   other tiers too: reversing the options changed the chosen move in 19 of 39 recorded states
   (1.5B) and 14 of 39 (3B), and the target in 7 of 12 and 10 of 12.

   **Change: order averaging** (`Engine(order_debias=True)`, off by default; on in the evaluated
   shooter, later set per model after the replication below).
   Every decision with two or more options is read twice in the same batch, with the options as
   listed and reversed, and the two softmaxes are averaged. It doubles the rows per pass. It does
   not remove the model's bias; it stops a fixed order from deciding for it. The viewer shows both
   readings. The evaluation also runs both models without it (`-listed`) to measure what it does.
2. *"closer" led next to a brute.* With plain walking distance, the 1.5B chose
   `move west (next to a brute: -20 health; closer: 3 steps)` at p = 0.78 when the short route
   passed the brute. Change: "closer" along routes that avoid danger (above).
3. *The 3B stood still.* It chose `stay (safe; no closer)` in 235 of 362 states where a safe move
   toward the target existed, and timed out. Re-asking recorded move prompts with other questions
   (states with a safe closer move / states with a risky option):

   | move question | 3B: safe closer chosen | 3B: risky chosen | 1.5B: safe closer | 1.5B: risky |
   |---|---:|---:|---:|---:|
   | "Which move is best? Never step into a bullet's path; otherwise get closer to your target." (first) | 70 / 130 | 15 / 64 | 103 / 130 | 7 / 64 |
   | "Which move brings you closer to your target? Do not step into a bullet's path or next to a brute." | 71 / 130 | 14 / 64 | 130 / 130 | 8 / 64 |
   | "Which move brings you closer to your current target?" | 129 / 130 | 13 / 64 | 130 / 130 | 10 / 64 |
   | **"Which move is best? Get closer to your target, but never step into a bullet's path."** (adopted) | 128 / 130 | 11 / 64 | 130 / 130 | 9 / 64 |

4. *The 3B followed the compass, not the route.* With the adopted question it still timed out on
   both dev seeds (277 and 223 stuck ticks). In each run one state repeated for over 200 ticks:
   `move west (wall)` at p = 0.71 (target "3 cells west and 6 cells north"), and a doorway where
   it chose `move east (farther)` because the target lay east but the route went west first.
   Over unique recorded states it chose the safe closer move 400 of 417 times: the failures were
   single states that repeat forever, because nothing moves in an empty corridor. Changes: walls
   are no longer options, and the move state gives the waypoint bearing.

| dev runs (ticks until the run ended) | 1.5B seed 1000 | 1.5B seed 1001 | 3B seed 1000 | 3B seed 1001 |
|---|---|---|---|---|
| first version | died at 38, 0 shots | – | – | – |
| + order averaging, safe-route "closer" | escaped, 137 | – | out of time, 231 stuck | – |
| + adopted move question | escaped, 125 | escaped, 210 | out of time, 277 stuck | out of time, 223 stuck |
| + waypoint bearing, no wall options (evaluated version) | escaped, 141 | escaped, 125 | escaped, 236 | escaped, 262 |

Four dev runs are not evidence of a win rate; the evaluation below is.

**Pre-registered evaluation** (`bench/shooter_eval.py` at commit `7951472`, seeds 0–9, every run
reported). Raw data: `bench/results/shooter_eval_20260927_002315.json`; traces in
`demo/output/` were pruned to the best run per setup (not committed; the runs are deterministic, so
`python -m bench.shooter_eval` with the same seeds regenerates them).

| setup (seeds 0–9) | escaped | died: gunner | died: brute | out of time | key picked up | rooms seen | rooms cleared | kills | hits taken | shots on target | held fire* | forward ms, median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.5B, order averaging (pre-registered default; still the demo)** | **7** | 2 | 0 | 1 | 9 | 6.5 | 4.6 | 8.5 | 3.0 | 89% | 63% | 245 / 410 |
| **3B, order averaging (pre-registered default)** | **2** | 8 | 0 | 0 | 10 | 6.9 | 4.8 | 9.1 | 6.7 | 93% | 0% | 338 / 803 |
| 1.5B, listed order | 0 | 6 | 2 | 2 | 5 | 3.1 | 0.7 | 0.7 | 6.1 | 89% | 98% | 176 / 317 |
| 3B, listed order (the 3B demo since the replication below) | 7 | 2 | 0 | 1 | 9 | 7.9 | 6.5 | 11.4 | 5.4 | 91% | 3% | 195 / 387 |
| random decisions (mock) | 0 | 0 | 0 | 10 | 0 | 1.0 | 0 | 0 | 0 | – | – | 0.8 / 1.9 |
| reference bot (not a model) | 10 | 0 | 0 | 0 | 10 | 7.5 | 6.1 | 11.2 | 0.4 | – | – | – |
| bot that never dodges (not a model) | 5 | 5 | 0 | 0 | 9 | 6.3 | 4.7 | 9.1 | 8.1 | – | – | – |

Means over the 10 runs, except counts. \* Share of the shoot decisions put to the model (with an
enemy in sight and the gun ready) answered "hold fire". Latency: per-run medians, then the median
over runs. Median escape tick: 1.5B 151, 3B 234 (2 runs), 3B listed 163, reference bot 148.

- **The level is beatable, and the 1.5B beats it most of the time.** 7 of 10 escapes with the
  1.5B, against 0 of 32 model runs in the first dungeon. The 1.5B holds fire in 63% of its shoot
  decisions and still clears rooms: shots that are fired hit 89% of the time.
- **Order averaging decides whether the 1.5B plays at all.** Without it the 1.5B held fire in 98%
  of shoot decisions, killed 0.7 enemies per run and never escaped (0 vs 7 of 10; two-sided
  Fisher exact p = 0.003).
- **For the 3B the pre-registered default did worse than the listed order** (2 vs 7 of 10;
  p = 0.07, so not conclusive on 10 seeds). Per decision, averaging did not make its moves
  riskier: in the 275 states with both a risky and a safe move, the averaged choice was risky 61
  times, the listed-order reading alone 58 times. It changed 14% of the 3B's moves and 29% of its
  targets relative to the listed reading, so the runs diverge early; which difference matters is
  not known. Replication on new seeds: below.
- **Almost every hit was a chosen risk.** Classifying every hit by the move chosen on that tick:

  | setup | hits | chose a move labelled `BULLET` or `next to a brute` while a safe move existed | no safe move | chose a move labelled safe |
  |---|---:|---:|---:|---:|
  | 1.5B, averaged | 30 | 29 | 0 | 1 |
  | 3B, averaged | 67 | 66 | 0 | 1 |
  | 1.5B, listed | 61 | 43 | 18 | 0 |
  | 3B, listed | 54 | 52 | 1 | 1 |

  The 3B took the risky move in 22% of the states where both were offered (61 of 275), the
  1.5B in 6% (24 of 387). A typical 3B case: standing on its firing spot with a gunner 3 cells
  north, it chose `move north (BULLET: -15 health)` at p = 0.97 over `stay (safe; on the target)`
  (seed 0, tick 24). Inferred, not tested: while "fight the enemies here" is the goal, the 3B moves
  toward the enemy it reads about, as the first dungeon's move tier moved toward its target.
- The two hits after a move labelled safe were not investigated here (later: a labelling defect,
  fixed; see "Post-hoc fixes" below).
- **Labelling defect found after the run (not fixed at the time, so the demo stayed the evaluated
  version; fixed later, below):** when the agent stands on its target and a bullet will cross its
  cell, the stay option reads `stay (BULLET: -15 health; reach the target)` instead of "on the target".
- One 1.5B run (seed 3) never left the start area: 389 stuck ticks on one explore target (cause
  found later, below).
- Random decisions never left the start room: the mock backend's choice is a fixed function of
  the prompt, so it repeats in a static state.
- The pre-registered replays (seed 0) both die to a gunner: the 1.5B at tick 98 after the key, the
  3B at tick 48.

**Post-hoc replication on seeds 10–19** (never run before; decision rule written into
`bench/shooter_eval.py` and committed as `88d739d` before the run: switch the 3B demo to the listed
order only if it also escapes more often than averaging there). Raw data:
`bench/results/shooter_eval_20260927_005732.json`.

| setup | seeds 10–19: escaped | died: gunner | out of time | hits taken | forward ms, median / p90 | seeds 0–19: escaped |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B, order averaging | 7 | 3 | 0 | 4.3 | 207 / 444 | 14 / 20 |
| 3B, order averaging | 6 | 4 | 0 | 6.0 | 323 / 784 | 8 / 20 |
| 3B, listed order | 7 | 2 | 1 | 4.2 | 195 / 404 | 14 / 20 |

- 7 > 6, so by the rule **the 3B demo now reads options in the listed order**
  (`[shooter] order_debias = false` in `config/lenovo-3b.toml`); the 1.5B keeps order averaging
  (`true` in `config/default.toml`). `Engine`'s own default stays off.
- The margin on the new seeds is one run, and over all 20 seeds the difference (8 vs 14) has a
  two-sided Fisher p = 0.11. The switch follows the pre-stated rule; it is not evidence that
  averaging harms the 3B. It does cost the 3B latency (p90 ~790 vs ~400 ms) and, in both seed
  sets, more hits (6.7 and 6.0 vs 5.4 and 4.2).
- Both models now escape about 7 in 10 runs (14 of 20 each). The reference bot's ceiling on
  seeds 0–9 was 10 of 10.
- Replays in `docs/`: the pre-registered seed-0 runs of both models (both died), the 3B's current
  setting on seed 0 (`shooter_replay_3b_listed_seed0.html`, escaped at tick 163), and the 1.5B's
  first escaped seed (`shooter_replay_1.5b_seed1_escaped.html`, chosen after the results).

**Post-hoc fixes found in the traces of both runs above** (commit `21f6358`; the check on new seeds
was written into `bench/shooter_eval.py` in the same commit, before it ran). The fixes change what
the model reads, so the demo is no longer the version evaluated above.

1. *The stay label.* A risky stay on the target read "reach the target"; it now reads "on the
   target". In the traces of both evaluations the defective option appeared in 150 move states and
   was chosen 33 times.
2. *Sleeping brutes.* All 4 hits taken after a move labelled `safe` (3 from the main run, 1 from
   the replication) were the same case: the move entered a room next to a sleeping brute, which
   entering wakes before enemies act, so the brute hit in the same tick. The labels only counted
   awake brutes. Now a sleeping brute counts when the move enters its room.
3. *No fight from the corridor.* In the stuck 1.5B run (seed 3), two awake brutes stood just inside
   a doorway while the agent was in the corridor. Fights were offered only inside the room, so the
   only goal was "explore" and every move read "no safe route to the target". A seen room's awake
   enemies can now be fought from its doorways and the corridors out of it. The first version of
   this fix (by line of sight) made the reference bot flip-flop between two cells on a dev seed,
   and the next one had it fight rooms it had not seen; both were corrected before the commit.

The same functions drive the reference bot: 60/60 on dev seeds and 30/30 on seeds 0–29 before
and after the fixes, with fewer hits (16 → 5 and 12 → 3). The bot that never dodges fell from 17
to 9 of 30, because it now also fights through doorways.

Check (seeds 30–39, never run before; `bench/results/shooter_eval_20260928_123222.json`, run on a
different machine, an RTX 3060 Ti with CUDA): 1.5B 5/10 escaped, 3B listed order 5/10, reference bot
10/10, bot that never dodges 3/10. No hit came after a move labelled safe. Both models timed out
on seed 37; the 1.5B because a sleeping brute next to the key left every move reading "no safe
route to the target", a side effect of fix 2 (not fixed yet).

### Ammo and the fire head

**Ammo** (on by default since `0fdaf31`; rules in the README, chosen with bots only by a rule
written before the runs, `bench/shooter_calibration.py --ammo`). `--classic` and `--game classic`
reproduce the game without ammo that everything above measured.

**The 1.5B held fire (Observed).** One run per model on seed 0 with ammo (1.5B with order
averaging, 3B in listed order): both died to a gunner, the 3B after 8 kills, the 1.5B with no shot
fired. It held fire in 53 of 53 shoot decisions. The two order readings of tick 9: with the options
as listed (`hold fire` last) p(hold) = 0.92; reversed (`hold fire` first) 0.19, with 0.81 spread
over the three enemies (0.15, 0.18, 0.48). Averaged, hold fire won with 0.56. Adding the three
shoot options up (0.44) would not have changed the choice, so the problem is not only votes split
across enemies.

**Re-asking recorded states (dev probe; seed 0 with ammo, 53 states, and dev seeds 1000–1001
without ammo, 136 states; order averaging on).** Held fire in the recorded runs: 53 of 53 and 55 of
136.

| shoot question and options | held fire |
|---|---:|
| as recorded: one `shoot <enemy>` option per enemy, `hold fire` last | 108 / 189 |
| "Do you shoot this tick?": `shoot at an enemy (N in sight, clear line)`, `hold fire` | 161 / 189 |
| "Which shot do you take this tick?": `shoot (N enemies in sight, clear line)`, `hold fire` | 0 / 189 |
| with ammo, "What do you do with your gun this tick?", same merged option | 0 / 53 |

A yes/no question made it worse; the merged option under the game's own question made it fire.

**Change (implementation choice, `fb3f106`).** With `fire_head` the shoot head offers one `shoot
(N enemies in sight, clear line)` option next to `reload` and `hold fire`, and an aim head in the
same batch (`Which enemy do you shoot?`, context: the strategy only) picks the enemy; with one enemy
in sight the aim is committed without a model call. The aim head sees neither the shoot head's
answer nor the move head's. On for the 1.5B (`[shooter] fire_head = true`), off for the 3B (which
held fire 0 of 62 on seed 0), always off in the classic game.

**Pre-registered check** (written into `bench/shooter_eval.py` and committed as `fb3f106` before
the run; seeds 40–49 with ammo, never run before; rule: keep the fire head on for the 1.5B if it
escapes at least as many seeds as the old head). Raw data:
`bench/results/shooter_eval_ammo_20260929_134450.json` (Lenovo, Arc 140V).

| setup (seeds 40–49, with ammo) | escaped | died: gunner | died: brute | out of time | key picked up | rooms cleared | kills | shots on target | held fire | reload chosen when offered | hits taken | stuck ticks | forward ms, median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.5B, fire head (the 1.5B demo)** | **6** | 3 | 0 | 1 | 7 | 4.6 | 8.8 | 92% | 0% | 23% | 4.1 | 2.1 | 324 / 812 |
| 1.5B, one shoot option per enemy | 0 | 6 | 1 | 3 | 4 | 0.6 | 0.6 | 93% | 98% | 94% | 5.4 | 123.2 | 476 / 931 |
| reference bot (not a model) | 10 | 0 | 0 | 0 | 10 | 6.4 | 11.8 | – | – | – | 0.2 | – | – |

- 6 vs 0 of 10 (two-sided Fisher p = 0.011): by the rule the fire head stays on.
- Without it the 1.5B still barely fought: it held fire in 98% of shoot decisions and, when
  `reload` was on offer, chose it 94% of the time. Two of its runs never left the start area
  (593 and 591 stuck ticks).
- With it: one run ran out of bullets for 40 ticks and still escaped (seed 47). Hits after a move
  labelled safe: 0 in both setups.
- Forward-pass latency was lower with the fire head (324 vs 476 ms median), although it adds a
  decision; why was not investigated.
- The 3B with ammo has not been evaluated; seed 0 of the ammo game is development data for this
  change.
