# Slides handoff: the final presentation

For a new session. Build the final presentation of this project as a **reveal.js** deck with **polished animations**, **as dynamic as possible**, that also **exports to a static PDF** with nothing lost. It's a separate deck from the midterm proposal, and it builds on that deck's content.

Written 2026-11-04, after Phase 25. Repo: `C:\Users\juji\Desktop\local-system-one` on the NUC, branch `main`.

## Start here: ask the owner first

**The midterm proposal deck isn't on this machine.** Ask the owner for it before anything else. Don't guess at its content.

Ask these in the same message:

1. The midterm proposal deck, as a file or link.
2. The talk's length, audience and date, and who presents.
3. Whether the game demos (shooter, dungeon, grid) get a place in the deck, or only the fraud work.
4. Whether any IEEE-CIS row may appear on a slide (see "Live replay" below).

**Wait for the midterm deck before settling:** the title and framing, names and course details, the "what we proposed, what we did" slide, the research questions the talk must answer, and any visual continuity with the midterm.

**This can start before it arrives:** the data build, chart components, the fraud result slides and the PDF pipeline. Everything those need is in this file.

## What the owner asked for

- reveal.js only. Remotion was considered and dropped: "skip that remotion removed. switched back to reveal.js".
- "I want polished animations."
- "make sure the reveal js is as dynamic as possible".
- Exportable to a static PDF.
- "this is separate from midterm but will build upon the midterm proposal slide deck's in term of content."

## The story

**Thesis (Inferred, from the numbers below):** a small language model, fine-tuned, can decide approve, review or decline on real payment data in one forward pass, with 56 ms at p90. That saves about 2,800 USD per 1,000 payments against approving everything. It hasn't beaten logistic regression on dollars. The clearest gain is to good customers, and it comes from the decision rule, not the model.

The owner's constraint runs through the talk. **The final setup must be a language decision model at heart**, and cost is optimised within that. So the fair number for the language model is the setup with **no tabular score in its prompt**: `lm-rich` on dev, `final-rich-*` on the final test.

**Tell this turn honestly.** The rule registered before the final test recommended `final-signal-cust`, where d1-Omni-2B reads a boosted-tree score. After the test the owner judged it "really not a decision model doing its grunt work", because its rank correlation with that score is 0.984. The registered result stands as it came out, and the deck says both things.

### Proposed slides

Implementation choice: adjust them to the talk's length once it's known. Spare detail goes in vertical sub-slides for questions.

| # | slide | the point | the dynamic treatment |
|---|---|---|---|
| 1 | Title | Wording waits for the midterm deck | Headline assembles in stages, then settles |
| 2 | What we proposed | From the midterm deck; don't invent it | Each promise appears, then later returns on the closing slide with its outcome |
| 3 | One look | A multiple-choice question answered in one forward pass (`docs/media/one-look.png`) | Option bars grow to their recorded leanings, one option lights |
| 4 | The problem in dollars | A missed fraud costs its amount, a wrong decline 10% (at least 1 USD), a review 10 USD, at most 5% reviewed | Three cost cards fly in and stack into the cost formula by Auto-Animate |
| 5 | From simulation to real data | On simulated PaySim, the trained model mostly matched hand-written rules: 3,853 of 3,862 decisions. IEEE-CIS is real payments | Morph PaySim's result into the IEEE-CIS timeline |
| 6 | The discipline | Split by time, windows locked, predictions committed first | 590,540 rows split into fit, dev, test; FINAL's 20 windows lock with a padlock |
| 7 | The rule | Weigh the amount in each decision | **Rule explorer** (below) |
| 8 | The ladder | Frozen to fine-tuned to more columns, on dev | Bars step down by fragment, logreg and trees arrive as reference lines |
| 9 | The shortcut | A tree score in the prompt looks cheapest, and the model mostly copies it | 0.984 counter, the owner's verdict quoted, the bar fades to an outline |
| 10 | The final test | One run, 20 windows | **Window race** (below) |
| 11 | Can we tell them apart? | No distinguishable saving | **CI forest**: intervals extend from their means, the zero line sweeps through |
| 12 | The customer line | Far fewer good customers declined, about the same cost | **Toggle** morphs the bars and counters (read-out badge on) |
| 13 | Training on cost | Could weighting each row by dollars help? | A 5 USD and a 1,500 USD fraud start level, then their weights split (the top 10% of training frauds hold 41% of the dollars; largest weight 7.3, from the docstring of `fraud/eval_lm_cost.py`) |
| 14 | Predictions against results | We wrote predictions first, and several were wrong | **Flip cards**, one per prediction, tinted by outcome |
| 15 | Where the money moved | +124 fraud, -61 declines, -10 reviews = +53 | **Waterfall** builds a step at a time, then size bands show large frauds were already caught |
| 16 | Fast enough | p90 56 ms on an RTX 3060 Ti against a 250 ms bar | Latency needle against the bar, with the machine caveat |
| 17 | What we can't claim | No clean test rows left, one adapter per setup, assumed costs | Limits slide in one at a time, none skippable |
| 18 | Promises revisited | The midterm's promises with outcomes | Slide 2's cards return by Auto-Animate, stamped |
| 19 | What's next | New data, and a stated choice for the owner | Quiet fade, ending on a single sentence |

**Don't list a second training pass as a next step.** The owner has ruled it out twice. Options the owner has seen: three times the legit rows (about 8 hours) and a second prompt with missing C and D columns (about 3.5 hours). Candidate datasets, none downloaded or tested, the owner's answer so far being "Not yet": Xente, ULB, EMSCAD.

## How dynamic, and how

Treat these as requirements.

**Every content slide has real motion tied to its meaning.** Use all of these across the deck:

- **Auto-Animate** (`data-auto-animate`, `data-id`) between related slides: bars, tables and numbers morph instead of cut.
- **Fragments** for staged reveals, including custom ones driven by `fragmentshown` and `fragmenthidden`.
- **Data-driven SVG charts** that animate from the results data, with count-up numbers.
- **Interactive pieces** the presenter can drive live.
- **Slide backgrounds and transitions** that mark each act: setup, journey, verdict.

**Five pieces to build:**

1. **Rule explorer.** Sliders for amount and for the fitted estimate p. Live bars show the expected cost of approve (`p × amount`), decline (`(1 − p) × friction`) and review (`10 + lambda`), and the cheapest lights up. A toggle adds the customer line's flat 10 USD to the decline side. This is arithmetic in the page, with no model. Friction is `max(0.1 × amount, 1)`. Take lambda from `fits` in the Phase 25 JSON. Label p "fitted on dev windows, used only to weigh costs", never a probability.
2. **Window race.** Cumulative cost across FINAL's 20 windows for baseline, `final-rich-amt`, `final-rich-cust` and approve-all, from the per-window lists. Play, pause and a scrubber. Bars or lines grow in step, with the running total counting up.
3. **CI forest.** Paired differences with intervals. Sort, then sweep the zero line across and mark what crosses it.
4. **Customer line toggle.** Switch between `final-rich-amt`, `final-rich-cust` and the baseline. Bars and counters morph.
5. **Prediction cards.** Each shows the registered prediction, then flips on a fragment to the outcome.

**Live replay.** Decisions from recorded runs only, because nothing in the deck may run a model. IEEE-CIS traces are git-ignored and hold row-level amounts, so **ask the owner before any IEEE-CIS row goes on a slide or into the repo**. Without that yes, use window-level data, which is already committed, or the committed PaySim traces listed in `fraud/runs/README.md` (CC BY-SA 4.0), labelled as PaySim. Never make up rows to fill a replay.

**Polish bar:**

- One set of motion tokens in CSS: a short list of durations and easings, used everywhere.
- Animate `transform` and `opacity` where possible, so it stays smooth at 1920×1080.
- The presenter sets the pace. Every step is a fragment or a control, with no timer the presenter can't stop.
- Going backwards works. Re-entering a slide starts it cleanly or shows its final state, never a broken middle.
- Same data, same animation: no unseeded randomness.
- Honour `prefers-reduced-motion` by jumping to final states.
- No information exists only in motion.
- Speaker notes on every slide, with sources (`S` view).

## PDF export

- **Documented** ([reveal.js PDF export](https://revealjs.com/pdf-export/)): open the deck with `?print-pdf` and print from Chrome or Chromium, landscape, margins none, background graphics on. Edge is Chromium and is on this machine. The docs confirm only Chrome and Chromium.
- **Documented:** fragments print as separate pages by default. Set `pdfSeparateFragments: false` so each slide is one page with every fragment shown, and `pdfMaxPagesPerSlide: 1`.
- **Implementation choice:** in print mode every chart and component draws its **final state at once**, with no animation and no controls. The rule explorer shows a small table of worked examples. The toggle shows all three setups side by side. Flip cards show both faces. The race shows its final frame.
- **GIFs don't play in a PDF.** Use a still from `docs/media/`, or render one.
- **Scripted export with no new package:** drive headless Edge over its debugging protocol, using `Page.printToPDF` with `printBackground: true` and `preferCSSPageSize: true`. Decktape is the alternative the docs name, and it's a new package, so ask first.
- **Accept:** one page per slide, text selectable, every number present, nothing clipped, 0 console errors. Render pages to images and look at every one.

## Build setup

- **reveal.js is approved because the owner asked for it. Pin 6.0.2**, released 2026-09-10 (Observed from GitHub releases on 2026-11-04). Reveal 6 moved its plugin and module paths, so follow [its upgrade notes](https://revealjs.com/upgrading/), not older examples.
- **Vendor it** so the talk needs no network: download the 6.0.2 release into a new empty folder, inspect it, and copy only what's needed, with its licence, into `slides/vendor/`. Use the classic script build if the deck should open from `file://`, since ES modules need a server.
- **Anything else needs the owner's yes,** including D3, GSAP, Chart.js, anime.js and a `package.json`. The repo has no Node project, though Node v24.20.0 is installed. Default: hand-written SVG and the Web Animations API, which cover everything above.
- **Fonts:** bundle them or use a system stack. No network calls at show time.
- **Where (Implementation choice, change it if the owner prefers):** a new top-level `slides/`, reading from `fraud/results/` and `docs/media/`. It writes nothing into `fraud/`, `demo/`, `viewer/`, `runs/` or `bench/`.
- **Numbers come from data, not typing.** `slides/build_data.py`, standard library only, reads the committed JSON and writes `slides/data.js`. Slides bind to it. A test in `tests/test_slides.py` rebuilds and compares, and checks the key numbers against the table below. `pytest.ini` already collects `tests/`.
- **Show it:** `.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory slides`
- **Check it:** per `CLAUDE.md`, a hidden browser pane doesn't animate. Use headless Edge: `msedge --headless=new --window-size=1920,1080 --virtual-time-budget=4000 --screenshot=<png> <url>`. Step through fragments over the debugging protocol, and check forward, backward and re-entry on the animated slides.

## Rules that still apply

- **Label claims** Documented (cite) / Observed / Inferred / Implementation choice in notes and sources. On slides, use the status badges below.
- **Report negative results plainly,** and report before and after in customers as well as dollars.
- **A difference counts only if the 95% window-bootstrap CI of the mean paired difference excludes 0.** Say "not distinguishable", never "slightly better".
- **Model scores are uncalibrated.** Never show one as a probability.
- **No inference in the deck.** No model runs anywhere for this work, so no GPU and no LLM server are needed.
- **FINAL is spent, and so are all 38 test chains.** Don't re-score them. Adding a claim about them beyond the committed files would be a "second look", and that's the owner's decision.
- **Don't compare costs across different windows as if paired.** FINAL and dev use different rows. Show them on separate slides or clearly apart.
- **Don't mix numbers from two machines in one comparison.** Latency here isn't comparable with the laptop's. The README's laptop figures are about a tenth of a second per decision on Intel Arc.
- **Never commit** `fraud/data`, `fraud/models`, ignored traces, adapters or raw rows.
- **PaySim's balance columns are never used,** not even to compare.
- **This project is not Jev and doesn't reproduce Jev.** Keep the README's wording if Jev comes up.
- **Commits have no attribution lines.** This is the owner's rule, and it overrides any default. Commit locally. **Don't push unless the owner asks, each time.** `RUN-GUIDE.md` gets the deck's commands.
- **No secrets in the deck.**

## Status badges

Every result slide carries one.

| badge | meaning |
|---|---|
| **FINAL, registered** | The 20 FINAL windows, run once, the question registered before the run |
| **FINAL, read-out** | FINAL's recorded scores, asked for after the results were seen. Not a test |
| **Dev only** | 88 held-out dev chains. Flattered, because lambda was picked with these rows in, except logreg's |
| **Earlier test** | Older test windows, which aren't paired with FINAL |
| **Assumption** | Depends on a cost nobody measured, such as 10 USD per declined customer |

## The numbers

USD per 1,000-row window unless stated. Every cost rests on the assumed costs. All Observed. **For differences here, a plus sign means the first setup costs more.** The JSON's `mean_diff` is signed from the cheaper setup's side, so check the sign when binding.

### Final test: FINAL, registered

`fraud/results/eval_final/20261104-090649.json`, write-up `fraud/RESULTS-FINAL.md`. 20 windows, 20,000 rows, 630 frauds, 104,651 USD of fraud. Per-window costs: `setups.<name>.windows`.

| setup | what it is | mean cost | 95% CI | AUC | frauds stopped | fraud USD stopped | fraud let through | good customers declined | reviews |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `final-signal-amt` | d1 reading a tree score, rule | 1,943 | 1,651–2,338 | .883 | 404 | 81,090 | 23,562 | 928 | 702 |
| `final-signal-cust` | the same, customer line; registered pick, rejected by the owner | 1,992 | 1,683–2,383 | .883 | 335 | 75,314 | 29,338 | 247 | 804 |
| `final-logreg-amt` | **baseline**: logreg, rule | 2,106 | 1,946–2,292 | .853 | 384 | 78,547 | 26,105 | 1,051 | 729 |
| `final-rich-amt` | **d1 with no tree score**, rule | 2,427 | 2,023–2,896 | .859 | 375 | 74,491 | 30,161 | 1,081 | 895 |
| `ieee-approve-all` | approve everything | 5,233 | 4,312–6,121 | n/a | 0 | 0 | 104,651 | 0 | 0 |

| comparison | difference | 95% CI | windows | |
|---|---:|---|---|---|
| `final-rich-amt` vs baseline (third question) | +321 | -74 to 774 | baseline cheaper 12 of 20 | not distinguishable |
| `final-signal-amt` vs baseline (primary) | -163 | -462 to 238 | 15 of 20 cheaper | not distinguishable |
| `final-signal-cust` vs baseline, reported | -115 | -391 to 273 | 16 of 20 cheaper | not distinguishable |
| `final-signal-amt` vs `final-rich-amt`, reported | -484 | -972 to -103 | 17 of 20 cheaper | **distinguishable** |
| `final-signal-amt` vs approve-all, reported | -3,290 | -4,127 to -2,422 | 19 of 20 cheaper | **distinguishable** |

p90 latency 56 ms for both adapters, bar 250 ms, RTX 3060 Ti, bf16. Replay from traces: 0 rows differ, largest gap 0.04 USD.

`final-signal-cust` in final12, one window: 4,936 against the baseline's 2,066, with two frauds of 1,504.47 USD let through. Rank correlation with the tree score on FINAL rows: 0.984 (`extras.json`).

### FINAL, read-out: d1 with no tree score, with the customer line

`fraud/results/eval_final/rich_cust_readout.json`. Asked for after every FINAL result was seen. Replay: 0 rows differ, largest gap 0.02 USD. Per-window costs: `windows`.

| over 20 windows | `final-rich-amt` | `final-rich-cust` | baseline |
|---|---:|---:|---:|
| mean cost | 2,427 | 2,417 | 2,106 |
| 95% CI | 2,023–2,896 | 2,030–2,867 | 1,946–2,292 |
| good customers declined | 1,081 | **444** | 1,051 |
| good customers declined or reviewed | 1,899 | 1,327 | 1,703 |
| frauds stopped, of 630 | 375 | 311 | 384 |
| fraud USD let through | 30,161 | 33,508 | 26,105 |
| reviews | 895 | 974 | 729 |

- `final-rich-cust` vs baseline: +311, CI -75 to 746, baseline cheaper 12 of 20, not distinguishable.
- `final-rich-cust` vs `final-rich-amt`: -10, CI -91 to 87, not distinguishable.
- `final-rich-cust` vs approve-all: -2,816, CI -3,631 to -1,986, **distinguishable**. `final-rich-amt` vs approve-all: -2,806, CI -3,631 to -1,970, **distinguishable**.
- **Assumption:** at a true 10 USD per declined customer, 2,639 against the baseline's 2,632, +7, CI -405 to 481.

### The ladder: Dev only

`fraud/results/eval_lm_nuc/20261104-090217.json`, write-up `fraud/RESULTS-LM-NUC.md`. 88 chains, 88,000 rows, 3,409 frauds. All with the rule.

| setup | what changed | mean cost | AUC | good customers declined | frauds stopped |
|---|---|---:|---:|---:|---:|
| `lm-frozen` | no fine-tune | 3,697 | .7540 | 9,586 | 1,808 |
| `lm-phase12` | fine-tuned | 2,288 | .8795 | 3,644 | 2,169 |
| `lm-seed1` | same recipe, seed 1 | 2,186 | .8815 | 3,253 | 2,194 |
| **`lm-rich`** | **more columns, no tree score** | **2,112** | .8932 | 3,459 | 2,324 |
| logreg | reference | 2,125 | .8882 | 4,003 | 2,347 |
| `lm-signal` | reads a tree score | 1,920 | .9102 | 3,241 | 2,410 |
| signal alone | the tree score, no language model | 1,853 | .9108 | 3,052 | 2,452 |
| trees | reference | 1,830 | .9139 | 3,003 | 2,445 |

- Fine-tuning vs frozen: -1,409, CI -1,641 to -1,188, **distinguishable**.
- `lm-rich` vs `lm-phase12`: -176, CI -281 to -58, **distinguishable**.
- `lm-rich` vs logreg: -13, CI -142 to 125, not distinguishable. **This tie didn't hold on FINAL.**
- `lm-signal` vs signal alone: +67, CI -33 to 166, not distinguishable. Rank correlation .9842, and 2,233 of 88,000 decisions differ.
- Trees vs `lm-rich`: -282, CI -423 to -151, **distinguishable**.
- A second seed moved cost by about 100 (-102, CI -205 to 6). Use this as the yardstick for one adapter's noise.

### Training on cost: Dev only

`fraud/results/eval_lm_cost/20261104-140349.json`, write-up `fraud/RESULTS-LM-COST.md`. Pre-registered before training. Replay: 0 rows differ, largest gap 0.03 USD.

| setup | mean cost | 95% CI | AUC | frauds stopped, of 3,409 | of 500 USD or more, of 275 | fraud USD stopped | good customers declined | reviews |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `lm-rich-amt` | 2,112 | 1,945–2,294 | .8932 | 2,324 | 258 | 465,486 | 3,459 | 3,855 |
| `lm-weighted-amt` | 2,165 | 2,013–2,323 | .8801 | 2,194 | 259 | 454,533 | 3,038 | 3,766 |
| `lm-rich-cust` | 2,226 | 2,044–2,428 | .8932 | 1,996 | 253 | 442,092 | 945 | 4,196 |
| `lm-weighted-cust` | 2,249 | 2,081–2,422 | .8801 | 1,895 | 257 | 435,964 | 783 | 4,122 |

- `lm-weighted` vs `lm-rich` (primary): **+53, CI -39 to 142, not distinguishable.** Weighted cheaper in 46 windows, `lm-rich` in 42.
- vs logreg: +40, CI -76 to 157, not distinguishable. vs trees: +335, CI 219 to 450, **distinguishable**.
- Waterfall per window: fraud let through 1,351 → 1,475 (+124), wrong declines 323 → 262 (-61), review fees 438 → 428 (-10), total +53.

Frauds stopped by size, `lm-rich` → `lm-weighted`: under 50 USD 823 → 780 of 1,181; 50–100 507 → 472 of 817; 100–250 466 → 419 of 794; 250–500 270 → 264 of 342; 500 or more 258 → 259 of 275. **These bands were counted from traces and aren't in the JSON.** Copy them from `fraud/RESULTS-LM-COST.md` into a small data file with that source.

Prediction cards, registered in `fraud/eval_lm_cost.py`:

| prediction | outcome | |
|---|---|---|
| lower than `lm-rich` by about 100, not distinguishable | +53, not distinguishable | direction wrong |
| AUC falls by up to 0.01 | falls 0.0131 | direction right, larger |
| fraud dollars stopped rise | fall 10,953 | wrong |
| frauds of 500 USD or more stopped rise | 258 to 259 | by one fraud, no real change |
| frauds stopped overall fall a little | fall 130, 6% | right |
| beats logreg by about 100, not distinguishable | +40, not distinguishable | direction wrong |
| stays dearer than the trees | +335, distinguishable | right |

Reading (Inferred): the rule had already stopped 94% of the large frauds, leaving training little to gain there.

### The rule and earlier work

Different windows from FINAL, so don't pair them with it. Phases 14, 18 and 20 are **Earlier test**. Check the other write-ups for which rows they used before choosing a badge.

- **Phase 14,** `fraud/RESULTS-IEEE-AMOUNT.md`: on the same logreg scores, the rule takes two lines' 2,752 to 1,926, -825, CI -1,307 to -464, **distinguishable**, cheaper in 18 of 20. It also declines more good customers, 982 against 655.
- **Phase 18,** `fraud/RESULTS-RULE.md`: the customer line takes good customers declined 476 → 153 on logreg, +119, CI -2 to 245, not distinguishable.
- **Phase 16,** `fraud/RESULTS-COST-STRESS.md`: the rule wins at every friction rate tried, and stops winning near 50–60 USD flat per decline.
- **Phase 17,** `fraud/RESULTS-COLUMN-AUDIT.md`: without C, D and V columns logreg costs about 1,000 more. C's timing couldn't be cleared, so a production system may not have them.
- **Phase 20,** `fraud/RESULTS-TABULAR.md`: trees promised -295 on dev and delivered -97, CI -385 to 182, not distinguishable.
- **PaySim,** `README.md`, `docs/media/fraud-rule-book.png`: 3,853 of 3,862 decisions matched hand-written rules. Mean costs: rules 808, trained model 830, logreg 190,910, untrained model 5.1 million. Simulated data and a different cost setup from IEEE-CIS.

### Data and recipe

| | |
|---|---|
| IEEE-CIS split, by time | 590,540 rows: fit 354,324, dev 118,108, test 118,108 |
| Windows | 1,000 contiguous rows; FINAL's 20 chosen with seed 20261105 before this work began |
| Costs (`eval_ieee.COSTS`) | friction 10%, at least 1 USD; review 10 USD; at most 5% reviewed |
| Bootstrap | windows resampled, seed 20261102 (`eval_ieee.compare`) |
| Model | d1-Omni-2B, LoRA, one pass, 47,952 rows: 11,988 frauds and 35,964 legit |
| Hardware | Intel NUC 12 Extreme, RTX 3060 Ti, bf16, peak about 4 GiB |
| Run times | Training about 2 hours, scoring 118,000 dev rows about 81–85 min, with the longer prompt |

## Don't say

- "The language model beats logistic regression." It doesn't, in any shown result on FINAL.
- "`final-signal-cust` is the language model's win." It mostly follows a tree score, and the owner rejected it.
- "Slightly cheaper" for a difference whose CI includes 0.
- "x% probability of fraud" for a model score or p.
- "It saved x USD" with no mention that costs are assumed.
- Anything about the midterm you haven't read.

## Done means

- The owner has supplied the midterm deck, and the slides that depend on it use its real content.
- The deck runs with no network at 1920×1080, with every animated slide checked forward, backward and on re-entry.
- The PDF meets the acceptance line above.
- `slides/data.js` builds from the committed files, and `pytest -m "not model"` passes.
- `RUN-GUIDE.md` has the show and export commands.
- A local commit with no attribution lines. Push only when the owner asks.
