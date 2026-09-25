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
| S1 (primary) | Sean Goedecke, [Two techniques for working with System One models](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/) | Obsidian vault `OneDrive/Documents/Obsidian/Juji's/Clippings/` (byte-identical copy in repo root) |
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
  user-provided multiple-choice questions. S3 calls such models "fast general classifiers".
- **Documented [S5]:** end-to-end latency of "70ms-500ms"; up to 255 choices per question.
- **Documented [S1]:** any LLM can be turned into a System One-style model without changing the model,
  as long as you can read the logits and prefill the prompt.

## Single-token decisions and logits

- **Documented [S2]:** instead of generating a structured answer token by token, "prefill the response
  with `"choice": "` and generate one token, restricted to the user-provided choices". The model reads
  all input tokens in parallel, so one forward pass gives the answer.
- **Documented [S4]:** the reference code applies the chat template and appends the prefill
  `choice_index:`. Only the logits of the allowed tokens are kept, and a softmax is taken over them.
  It checks at start-up that each index is one token that decodes correctly after the prefill.
  It reports `confidence = 1 - normalised entropy`.
- **Documented [S1, footnote 3]:** indexes vs labels. Labels ("just picking some token to associate
  with the choice") worked "way better" on Wikiracing but not on Doom. **Documented [S4]:** the label
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
  inference batching. This is what makes the approach "consistently fast".
- **Documented [S1, footnote 2]:** the Qwen3-8B Doom demo made 6–7 batched decisions every ~500 ms on
  an RTX 4090 and every ~190 ms on an H100. The tool-calling version made one decision every ~600 ms.
- **Documented [S4]:** left padding, `position_ids` from the cumulative attention mask, and
  `logits_to_keep=1`. Optional shared-prefix KV caching (`cache_prefix`) is recommended for more than
  3 questions.
- **Implementation choice:** we use the same padding scheme. Shared-prefix caching is not implemented
  (see Limitations in the README).

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
- **Implementation choice:** see `system_one/goals.py`. Tier periods are counted in ticks. A child
  tier's options may depend on the parent's current goal.

## Tournament choice sampling

- **Documented [S1]:** a Wikipedia page can have more than 1000 links. S1's layer "stopped working
  well" after about 100 choices. Jev's own approach for large sets is "a 2 stage-system of scoring
  independently then making an explicit choice" [S5]. That failed for Qwen3-8B: hundreds of links got
  the same top score.
- **Documented [S1]:** tournament sampling is to "fed a hundred links at a time into each choice, then
  did a second pass with the chosen links". "Ordinary LLMs are way better at relative judgements than
  absolute ratings."
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

- **Documented [S1]:** "We don't know exactly how Jev works" (people guess diffusion, Transformer
  tweaks, or a new model type). **Documented [S5]:** TypeSafe mentions "a new model architecture,
  parallel sampler ... and training method we call Reinforcement Learning for Calibrated Decisions
  (RLCD)". It gives no details on architecture, size, training data, or RLCD.
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
- The first XPU forward pass in a process takes ~3–4 s (kernel compilation). Later ones take
  tens of ms. Benchmarks warm up first.

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
