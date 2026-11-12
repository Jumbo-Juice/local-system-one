# Final paper, revamped deck and primer: design (2026-11-12)

Status: approved in conversation on 2026-11-12; this file is for the owner's review before the
implementation plan.

## Why

The course has a new professor who doesn't know the project. His background is econometrics; he
taught scikit-learn and is now teaching PyTorch; he teaches by practical example and by flipping
expectations (for example two identical-looking graphs that turn out to differ only by scaling).
The owner will show him three things: the fraud console (unchanged), a research paper and a
revamped final deck. The midterm deck (`../midterm-proposal/`) stays exactly as it is, as proof of
work.

Success: a newcomer with scikit-learn knowledge follows how each method decides on one payment,
understands how a language model becomes a decision model, and leaves with the honest results.

## Deliverables

| deliverable | where | form |
|---|---|---|
| Research paper | `paper/` (new) | IEEE 2-column, about 7 pages, Typst, PDF exported to `paper/exports/` |
| Final deck | `slides/` (rewritten in place; the old deck stays in git history) | reveal.js, static, 12:00 talk then questions |
| Econometrics primer | `docs/econometrics-primer.md` | Markdown, for the owner only |

Untouched: `../midterm-proposal/`, the fraud console, `story/` (it must keep building).

## Rules for both paper and deck

Do:
- One worked example per method, on one payment followed through every method.
- Show the surprising picture first, explain on the next step.
- Most method time goes to the language model as a decision model (the novel part).
- PaySim gets its full stage, as the first cycle.

Don't:
- No phase numbers, setup ids or file paths on slides; plain names only. (File paths may appear in
  speaker notes as sources.)
- No term without a one-line example the first time it appears (CI, dev, window, adapter, logit…).
- No typed numbers: every number comes from committed results through a build script.
- No softened negatives: no setup showed a distinguishable saving; fairness wasn't measured.
- No explicit econometrics framing anywhere ("this is the logit model from econometrics",
  "bridge to…"). Ideas are explained in plain ML terms; the professor can make the link himself.
  The paper may use the standard term "softmax (multinomial logistic)" once.
- No IEEE-CIS row shown, as before: window-level results only, plus the committed PaySim trace
  labelled as PaySim.

## The worked payment

A real PaySim TRANSFER from the pinned trace
`fraud/runs/20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl` (untrained Qwen2.5-1.5B).
Each model-judged record holds `scores` (the averaged probabilities) and `orders` (the
probabilities for each option order). The build script picks the payment by a written rule
(Implementation choice: the first model-judged TRANSFER whose two orders disagree on the top
answer), so the choice isn't made by eye.

- Slide 08 shows its logits as log-probabilities (softmax is unchanged by adding a constant, so
  these are the logits up to a shift; the slide says so) and the softmax to three probabilities.
- Slide 09 shows the same payment's two orders disagreeing, then the average.
- Rules, logistic regression and boosted trees have no recorded score for this payment. Their
  slides show the mechanism with numbers labelled "illustration", never presented as measured.
- The trace is the untrained model; training's effect is shown at window level (slide 10), not on
  this payment, unless a trained trace for the same payment is committed.

## Deck

Midterm style: pipeline-diagram pages (as midterm page 4), numbered green component tags (01, 02 …),
static pages that read the same in the PDF (no sliders, toggles or animated scenes). The
answer-sheet extras of the current final deck go: section letters, Assumption badges, the
interactive scenes. Same base (white page, green ink, Archivo), pictures and numbers only, no code.

Running order, 12:00 of talk (presenters are a proposal: Juji, Swan, Ryu, Ponh):

| # | slide | shows | time | presenter |
|---|---|---|---|---|
| 01 | Title | | 0:25 | Juji |
| 02 | The problem | approve / review / decline per payment; 33.83 B USD card fraud losses [1]; can a small language model be the decision maker? | 0:45 | Juji |
| 03 | Flip: accuracy lies | approve-all is about 97% accurate yet the most expensive; reveal: judge by dollars, the cost formula | 1:00 | Juji |
| 04 | Data and split | PaySim (simulated), IEEE-CIS (real), windows of 1,000, time split, a final test locked away | 0:45 | Juji |
| 05 | Method: rules | the amount-aware rule, worked on the payment | 0:45 | Swan |
| 06 | Method: logistic regression | features × weights → score; a weight changes the odds of fraud | 0:50 | Swan |
| 07 | Method: boosted trees | many small trees, each correcting the last | 0:45 | Swan |
| 08 | Method: language model as decision model | pipeline: facts → lettered question → one forward pass → A/B/C logits → softmax; the real payment | 1:15 | Juji |
| 09 | From probabilities to an action | cut-offs; asked twice with options reversed, averaged | 0:45 | Juji |
| 10 | Method: training | a small adapter (LoRA) trained on the letter; on PaySim, from useless to matching the rules | 1:00 | Juji |
| 11 | Results: the final test | cost per window, each setup against logistic regression; no distinguishable saving; the interval as a paired difference | 1:00 | Ryu |
| 12 | Flip: same model, two verdicts | boosted trees' dev saving shrank on new windows | 0:45 | Ryu |
| 13 | What did work | far fewer good customers wrongly declined; speed under the 250 ms bar; the cheapest-looking language setup mostly copies the tree score | 0:45 | Ryu |
| 14 | Biases | assumed costs; option-order bias | 0:45 | Ponh |
| 15 | Takeaways | what holds, where to look (paper, console) | 0:25 | Ponh |

Methods (04–10) about 6:00 plus framing; results (11–14) about 3:15. Every number above is a
placeholder until bound. A references slide follows, not presented. Each slide has speaker notes:
what to say, and the source of each number.

Build: keep reveal.js (vendored), `slides/build_data.py`, `data.js`, the `{{path|format}}` binding,
`build_data.sentence()`/`diff()` and their tests. Remove the scene machinery (`js/scenes.js` and
scene-only CSS) once nothing uses it. `story/build_data.py` imports `slides/build_data.py`: its
functions and output keys that the story reads must keep working (`tests/test_story.py` passes).
`check_deck.mjs` and `tests/test_slides.py` are updated to the new slides; the PDF export is
regenerated in `slides/exports/`. Check at 1920 × 1080 and in `?print-pdf`.

## Paper

Title: *Transforming LLMs into classifiers for fraud detection* (the midterm's). Authors: the
four group members; names and student IDs to be given by the owner.

| section | pages | content |
|---|---|---|
| Abstract | | question, method in one sentence, honest headline |
| I Introduction | 0.6 | the decision before the money moves; [1]; the question; contributions; findings |
| II Related work | 0.5 | [2]–[7]; [13] as the idea's origin; the gap |
| III Data and protocol | 1.0 | PaySim [8], IEEE-CIS [9]; windows; time split; dev then a locked final test run once; cost model; pre-registration; paired differences and intervals; CRISP-DM [12], two cycles |
| IV Methods | 2.0 | A rules (amount-aware rule, customer line); B logistic regression; C boosted trees; D the language model as decision model (facts + lettered question, one pass, logits, softmax, both orders [5][6]); E models: Qwen2.5-1.5B [10] on PaySim, d1-Omni-2B [14] on IEEE-CIS, and why; F training with LoRA [11]; G probabilities to action, and a language model reading a tree score |
| V Results | 1.3 | A PaySim; B IEEE-CIS on dev; C the final test (Table II); D customer line; E speed; F the two flips |
| VI Biases and limitations | 1.0 | assumed costs; option-order bias; optimism from choosing on dev; the language model copying the tree score; fairness promised, not measured; columns that may not exist at decision time; PaySim simulated; one adapter per setup; no clean test rows left |
| VII Conclusion | 0.25 | |
| References | 0.4 | [1]–[13] as in `../midterm-proposal/PLAN.md`, plus [14] LiquidAI, "d1-Omni-2B", Hugging Face model card |

Figures: Fig. 1 the decision pipeline (as slide 08); Fig. 2 final-test cost per window against
logistic regression with intervals; Table I the softmax on the worked payment; Table II final-test
results.

Sources: the overall write-ups (`fraud/SUMMARY.md`, `fraud/RESULTS-FINAL.md`, `fraud/RESULTS.md`,
`slides-handoff.md`, the current deck's notes, `../midterm-proposal/PLAN.md`), not each phase's
detail. Claims use paper wording: measured results stated as results, inferences marked ("suggests",
"we infer"), implementation choices named.

Build: `paper/build_data.py` imports `slides/build_data.py` and writes `paper/data.json`; `main.typ`
reads every number from it. The IEEE template (`charged-ieee`, Typst Universe) is vendored in
`paper/vendor/` with a `SOURCE.txt`, so a build needs no network. Typst is installed only after
checking the package is authentic and with the owner's go-ahead. `tests/test_paper.py` rebuilds the
data, checks every number the source reads exists, and compiles when Typst is present.
`RUN-GUIDE.md` gets the build commands.

## Primer

`docs/econometrics-primer.md`, plain language, for the owner; nothing in the deck or paper links to
it. Each topic: the idea, a small worked example, where it appears in the deck or paper, how to
answer if asked.

1. Logistic regression as the logit model: odds, odds ratios, what a weight means.
2. Softmax as the multinomial logit.
3. Standardisation and normalisation: how scaling changes what a weight seems to say.
4. Expected loss: deciding by cost, not accuracy.
5. Confidence intervals and the bootstrap.
6. Paired differences.
7. Selection bias and specification search: why dev flatters, why we pre-registered.
8. Class imbalance and base rates.

## Open

- Author names and student IDs for the paper.
- Typst install: the owner's go-ahead after the authenticity check.
