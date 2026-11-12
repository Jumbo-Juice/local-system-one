# Final paper, revamped deck and primer: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the final deck with a static, method-first, midterm-style deck; add an IEEE
2-column Typst paper; add a primer on the statistics behind the methods, for the owner.

**Architecture:** One source of numbers: `slides/build_data.py` reads committed results and is
extended, never reshaped (`story/build_data.py` imports it). The deck binds `{{path|format}}` in
HTML as now. The paper binds the same `path|format` keys through `paper/build_data.py`, which
writes `paper/data.json` (key → formatted string) using Python ports of the deck's formats, and
`paper/main.typ` reads it.

**Tech Stack:** Python 3.13 stdlib, reveal.js 6.0.2 (vendored), Node + headless Edge (`cdp.mjs`),
Typst (new) with the `charged-ieee` template vendored, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-final-paper-and-deck-design.md`

## Global Constraints

- No typed numbers: every number on a slide or in the paper is a binding; prose-only numbers go in
  `slides/manual_data.json` with a `_source`.
- Visible slide text (outside `<aside class="notes">`) has no phase numbers (`Phase \d+`), setup ids
  (`(final|lm|tab|ieee|audit|rule|signal)-[a-z0-9-]+`), file paths (`fraud/`, `.py`, `.json`,
  `.jsonl`) and no "econometric". Same for the paper's body text, except file paths in a
  "Code and data" footnote.
- No IEEE-CIS row: window-level IEEE-CIS results only; the one payment shown is PaySim, labelled as
  PaySim (`test_only_paysim_rows_reach_the_deck` stays).
- Every term is defined with an example where it first appears (window, dev, final test, interval,
  logit, softmax, adapter, LoRA).
- Negatives stated plainly: "no setup showed a distinguishable saving"; "fairness was not measured".
- Deck: 12:00 of talk (`MAX_TALK_S = 12 * 60`), 15 slides + one unpresented references slide,
  1920 × 1080, static (no scenes); steps (`data-from`) allowed, `?print-pdf` shows every final state.
- Presenters on tags and notes: Juji 01–04 and 08–10, Swan 05–07, Ryu 11–13, Ponh 14–15.
- Authors (paper and title slide), from the current title slide: Juji Vongvilay,
  Thu Ya Aung, Chanpichsoponha Samnang, Sai Thit Swan Thu.
- References [1]–[13] exactly as `../midterm-proposal/PLAN.md` → References, plus
  [14] LiquidAI, "d1-Omni-2B," Hugging Face model card, `LiquidAI/d1-Omni-2B`, 2026.
- No network at build or show time. `../midterm-proposal/`, `fraud/viewer/`, `story/` untouched.
- Commit after each task; never push.

## Review Focus

1. A key the story reads disappears or changes shape in `slides/build_data.py` → the story build
   must still pass: Task 2 runs `tests/test_story.py`.
2. A "reveal, then explain" step hides the explanation in the PDF → every `data-from` element must be
   shown under `?print-pdf`: Task 3 test `test_print_shows_every_step`.
3. The worked-payment rule matches nothing (trace replaced) → `worked_block()` raises `LookupError`,
   never picks silently: Task 2 test.
4. The paper and the deck format the same number differently (minus sign, thousands) → Task 6 test
   `test_formats_match_the_deck` runs fixed inputs through both.
5. The paper drifts outside 6–8 pages after an edit → Task 7 test `test_paper_compiles_and_is_six_to_eight_pages`.

---

### Task 1: Install Typst and vendor the IEEE template

**Files:**
- Create: `paper/vendor/charged-ieee/` (template files), `paper/vendor/charged-ieee/SOURCE.txt`
- Create: `paper/.gitignore` (`exports/` is ignored repo-wide; un-ignore with `!exports/` and keep the PDF)

- [ ] **Step 1: Vet before installing.** Use `supermatt:dependency-vetting` on the winget package
  `Typst.Typst` (publisher, repository github.com/typst/typst, release signature/hash) and on
  `charged-ieee` (typst/templates repository, Typst Universe). Report the findings to the owner and
  get a yes before Step 2.
- [ ] **Step 2: Install.** Run: `winget install --id Typst.Typst -e`. Verify: `typst --version`
  prints a version. Record the version in `SOURCE.txt`.
- [ ] **Step 3: Vendor the template.** Fetch the newest `charged-ieee` release once (for example
  `typst init @preview/charged-ieee` into the scratchpad), copy its `lib.typ` and any files it
  imports into `paper/vendor/charged-ieee/`, and write `SOURCE.txt`: URL, version, licence (MIT-0),
  date, sha256 of each file. If the template imports another package, vendor that too the same way.
- [ ] **Step 4: Smoke-compile.** A scratch `.typ` with `#import "vendor/charged-ieee/lib.typ": ieee`
  and `#show: ieee.with(title: [Test])` compiles with `typst compile` and no network (disconnect not
  needed: no `@preview` import remains). Expected: a PDF, exit 0.
- [ ] **Step 5: Commit** `paper/vendor/`, `paper/.gitignore`: "paper: Typst and the IEEE template, vendored".

### Task 2: New numbers in the deck's data build

**Files:**
- Modify: `slides/build_data.py` (add blocks; existing keys unchanged)
- Modify: `slides/manual_data.json`
- Modify: `slides/data.js` (regenerated)
- Test: `tests/test_slides.py`

**Interfaces:**
- Produces: `build()["worked"]` = `{"i", "type", "amount", "to", "hour", "truth", "orders": {"listed": [a, r, d], "reversed": [a, r, d]}, "scores": [a, r, d], "logits": {"listed": [...], "reversed": [...]}, "top": {"listed": str, "reversed": str, "averaged": str}, "action": str}`.
  Probabilities are in approve, review, decline order for both orders (the trace stores the reversed
  order mapped back; `system_one/engine.py:303-310`). `logits` are natural logs of the probabilities,
  each list shifted so its largest entry is 0 (softmax is shift-invariant; the slide says so).
- Produces: `build()["final"]["approve_all_accuracy"]` = `1 - frauds / rows` on the final test.
- Produces: `build()["manual"]["stress"]` = `{"_source": "fraud/RESULTS-COST-STRESS.md", "flat_low": 50, "flat_high": 60}`
  (the flat per-decline price at which the amount-aware rule stops winning; copy the values and
  wording from that file, correct them if it says otherwise).

- [ ] **Step 1: Write the failing tests** in `tests/test_slides.py`:
  - `test_worked_payment_is_paysim_and_its_orders_disagree(data)`: `w = data["worked"]`;
    `w["type"] == "TRANSFER"`; `w["top"]["listed"] != w["top"]["reversed"]`;
    `abs(sum(w["orders"]["listed"]) - 1) < 1e-6` and the same for reversed;
    `max(w["logits"]["listed"]) == 0`; `softmax(w["logits"]["listed"])` equals `w["orders"]["listed"]` to 1e-9.
  - `test_worked_rule_raises_when_nothing_matches(monkeypatch, tmp_path)`: point `build_data.TRACE` at a
    trace with only filter rows; `pytest.raises(LookupError)` from `build_data.worked_block()`.
  - `test_approve_all_accuracy(data)`: equals `1 - data["final"]["frauds"] / data["final"]["rows"]` and lies in (0.9, 1).
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_slides.py -k "worked or accuracy" -v`. Expected: FAIL (KeyError / AttributeError).
- [ ] **Step 3: Implement** `worked_block() -> dict` in `slides/build_data.py`: the first `tx` row
  with `by == "model"`, `tx.type == "TRANSFER"` and argmax of `orders[0]` ≠ argmax of `orders[1]`;
  `LookupError("no two-order TRANSFER that flips in the trace")` if none. Add `"worked": worked_block()`
  to `build()` and `approve_all_accuracy` to `final_block()`. Add `stress` to `manual_data.json`.
- [ ] **Step 4: Regenerate and run.** `.venv/Scripts/python slides/build_data.py`, then
  `.venv/Scripts/python -m pytest tests/test_slides.py tests/test_story.py -v`. Expected: the new
  tests PASS; `test_story.py` PASS (story untouched). Old deck-layout tests may still pass here; they
  change in Task 3.
- [ ] **Step 5: Commit** "deck data: the worked PaySim payment in both orders, approve-all accuracy, the cost-stress edge".

### Task 3: Static, midterm-style deck frame

**Files:**
- Modify: `slides/deck.css` (midterm look; remove scene and section-letter styles)
- Modify: `slides/js/main.js`, `slides/js/core.js` (keep formats, `bindAll`, steps; drop scene calls)
- Delete: `slides/js/scenes.js`, `slides/js/cards.js` if nothing uses them
- Create: `slides/js/charts.js` (static SVG charts drawn from `window.DATA` on load)
- Modify: `slides/check_deck.mjs` (signature without scene elements; forward, backward, re-entry stay)
- Modify: `slides/index.html` (frame only: title slide + one sample method slide to prove the frame)
- Test: `tests/test_slides.py`

**Interfaces:**
- Produces, used by Tasks 4–5: slide markup
  `<section id="sNN" data-n="N"><div class="page"><span class="tag">N · Name</span><div class="comp"><span class="num">NN</span>Component</div><h1>…</h1><div class="main">…</div><p class="foot">…</p></div><aside class="notes">[Name · m:ss] …</aside></section>`;
  pipeline pages use `<div class="pipe">` with `<div class="stage">` children (label, title, body), as
  midterm page 4; `charts.js` exposes `Charts.interval(el, rows)` (rows: `{label, mean, ci:[lo,hi]}`;
  zero line, one dot and bar per row) and `Charts.bars(el, rows)` (rows: `{label, value}`).
- Look (from `../midterm-proposal/index.html` and its CSS, copy values from there): white page,
  green ink `#0c6a54`, Archivo, green number chip, rule lines above blocks, body ≥ 38 px, h1 64 px,
  references 22–26 px; no right-edge timing marks, no Assumption badges, no section letters.

- [ ] **Step 1: Write the failing tests** (replace the obsolete ones):
  - Delete `test_scene_names_are_unique_and_every_scene_on_a_slide_exists`,
    `test_every_result_slide_has_a_status_badge`, `test_the_speed_backup_draws_the_consoles_points_and_says_its_sentence`,
    `test_the_recipe_slide_says_what_the_code_does` (no recipe or speed backups now); keep the data,
    binding, network, format, comparison-sentence and PaySim-only tests.
  - `test_no_scenes(html)`: no `data-scene`, no `scenes.js` reference.
  - `test_visible_text_follows_the_rules(html)`: on text outside notes, none of the Global
    Constraints patterns match (phase numbers, setup ids, paths, `econometric`).
  - `test_print_shows_every_step()`: `core.js` under `PRINT` adds `shown` to every `[data-from]`
    (assert on the source text of the rule, or via `check_deck.mjs` in Task 5 — pick one and keep it).
  - `test_every_slide_has_notes_and_the_talk_fits(html)`: `len(leaves) == 16` (15 + references),
    total ≤ `MAX_TALK_S`, references slide has no timing.
- [ ] **Step 2: Run** `pytest tests/test_slides.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the frame: CSS, `charts.js`, trimmed `main.js`/`core.js`, title slide
  (title, the four names and IDs, group) and one method slide stub; delete unused JS.
- [ ] **Step 4: Run** the tests (all but the count test PASS; the count passes after Task 5) and
  `node slides/export_pdf.mjs`; open `slides/exports/png/p-01.png`. Expected: midterm look, no overflow.
- [ ] **Step 5: Commit** "deck: a static midterm-style frame; scenes removed".

### Task 4: Slides 01–07 (problem, data, rules, logistic regression, trees)

**Files:** Modify `slides/index.html`; Test `tests/test_slides.py`

Content per spec running order. Bindings to use:
- 02: `manual.split`, reference [1] figure in prose with source in notes (33.83 B is in the midterm's
  source list; add it to `manual_data.json` as `nilson` with `_source` if not bound already).
- 03 (flip, two steps): step 0 shows approve-all's accuracy `final.approve_all_accuracy|pct` as a
  big "good" number; step 1 reveals its cost `final.setups.approve_all.mean|int` per window against
  the baseline `final.setups.baseline.mean|int`, then the cost formula from `costs`.
- 04: PaySim and IEEE-CIS, `final.windows_n`, `final.rows`, `final.frauds`, the time split from `manual.split`.
- 05–07: mechanism slides on the worked payment (`worked.type`, `worked.amount|usd2`, `worked.hour`).
  Numbers that the method did not record for this payment are drawn from an "illustration" block
  added to `manual_data.json` (`"illustration": {"_source": "Implementation choice: made-up numbers to show the mechanism", …}`)
  and the slide labels them "illustration". 06 says "each weight changes the odds of fraud"; no
  field name, no econometrics. 07: trees from `manual.recipe.trees`.

- [ ] **Step 1: Failing test** `test_illustration_numbers_are_labelled(html)`: every section that binds
  `manual.illustration.` contains the visible word "illustration".
- [ ] **Step 2: Run** it. Expected: FAIL until 05–07 exist with the label.
- [ ] **Step 3: Write** slides 01–07 with notes (`[Name · m:ss]`, what to say, `Source:` lines with
  file paths and claim labels).
- [ ] **Step 4: Run** `pytest tests/test_slides.py -v` and `node slides/export_pdf.mjs`; read
  `p-01.png`…`p-07.png`. Expected: PASS (except the slide count), no overflow, no text under the minimums.
- [ ] **Step 5: Commit** "deck: slides 01–07, the problem and the classical methods on one payment".

### Task 5: Slides 08–15 and references

**Files:** Modify `slides/index.html`, `slides/README.md`, `RUN-GUIDE.md` (Final presentation section);
Test `tests/test_slides.py`

- 08 pipeline page: facts → lettered question → one forward pass → `worked.logits.listed` (as
  "log-scores, shifted so the top is 0") → softmax → `worked.orders.listed`; foot: "softmax ignores a
  shift, so these are the logits up to a constant". Model named as `paysim.model` (PaySim stage).
- 09 (flip-like, two steps): the same payment, `worked.orders.listed` then `worked.orders.reversed`
  (top answers differ), then `worked.scores` averaged and the cut-offs → action `worked.action`.
- 10: LoRA in a picture (frozen weights + small adapter `manual.recipe.lora_r`); PaySim before/after
  from `manual.paysim` (untrained `mean_untrained_millions`, trained `mean_trained`, rules `mean_rules`).
- 11: `Charts.interval` over `final.diffs.*_baseline` with `said` sentences in the notes; the verdict
  word via `|dist`; one sentence defining the interval as "the difference on the same windows, and
  how far it could plausibly move".
- 12 (flip, two steps): trees on dev `rule.phase20.dev.mean|signed` per window, then on new windows
  `rule.phase20.diff.mean|signed` with `|dist`.
- 13: good customers declined, baseline `final.setups.baseline.good_declined` against the
  registered tree-reading setup with the customer line `final.setups.signal_cust.good_declined`, and
  the language model with no tree score `final.setups.rich_cust.good_declined`, labelled "read after
  the test" (it is a read-out, not a registered result; `fraud/SUMMARY.md`, first paragraph);
  `final.latency.p90` vs `final.latency.bar`; `final.rank_corr` for "mostly copies the tree score".
- 14: assumed costs (`costs`, `manual.stress.flat_low`–`flat_high`) and option-order bias (point
  back to 09's payment).
- 15: takeaways; "the paper and the analyst console have the details".
- References slide: [1]–[14] per Global Constraints; no timing tag.

- [ ] **Step 1: Failing tests**: `test_references_are_the_fourteen(html)` (14 entries, [14] names
  `d1-Omni-2B`); `test_flip_slides_reveal_in_two_steps(html)` (sections 03, 09, 12 each have a
  `data-from="1"` element).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Write** slides 08–15 and references with notes; update `slides/README.md` (15 slides,
  static, how built) and `RUN-GUIDE.md` (commands unchanged unless a script changed).
- [ ] **Step 4: Verify.** `pytest tests/test_slides.py tests/test_story.py -v` all PASS;
  `node slides/export_pdf.mjs` PASS and read every PNG; `node slides/check_deck.mjs <scratch>` PASS
  with 0 console errors; open `?print-pdf` once and confirm 03, 09, 12 show both steps.
- [ ] **Step 5: Commit** "deck: slides 08–15 on the language model, training, results and biases; references".

### Task 6: Paper data and formats

**Files:**
- Create: `paper/build_data.py`, `paper/data.json` (generated, committed)
- Test: `tests/test_paper.py`

**Interfaces:**
- Consumes: `slides/build_data.py` `build()` (loaded by path, as `story/build_data.py:50-55` does).
- Produces: `FORMATS: dict[str, Callable]` with the same names and output as `slides/js/core.js`
  `formats` (int, signed, ci, range, auc, auc4, aucrange, num, abs, pct, d2, usd2, dist), using `−`
  (U+2212) for minus and `js_round` from the deck build for rounding; `keys(typ: str) -> list[str]`
  returns every `v("path|fmt")` key in the Typst source; `render() -> str` writes
  `{"<path>|<fmt>": "<text>"}` sorted, for the keys `paper/main.typ` uses.

- [ ] **Step 1: Failing tests** in `tests/test_paper.py`:
  - `test_formats_match_the_deck`: for inputs `-115.4, 0.4, 1051, 2106.5, [-391.2, 273.8], 0.8834, 0.0315, 305369.89`
    and `{"distinguishable": False}`, each Python format equals the JS one, computed by running
    `slides/js/core.js`'s format functions in `node` (extract with the same approach
    `tests/test_slides.py` uses to run the console's code; if that is awkward, assert against the fixed
    strings the JS gives, written into the test once, e.g. `signed(-115.4) == "−115"`).
  - `test_data_json_is_what_the_build_writes`: file equals `render()`.
  - `test_every_key_in_the_paper_resolves`: every `keys(main.typ)` entry is in `data.json` and not `"None"`.
- [ ] **Step 2: Run** `pytest tests/test_paper.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** `paper/build_data.py`.
- [ ] **Step 4: Run** the tests. Expected: PASS (with a stub `main.typ` holding two `v()` keys).
- [ ] **Step 5: Commit** "paper: numbers from the deck's build, formatted as the deck formats them".

### Task 7: The paper

**Files:**
- Create: `paper/main.typ`, `paper/README.md`; output `paper/exports/transforming-llms-fraud.pdf`
- Modify: `tests/test_paper.py`, `RUN-GUIDE.md` (new "Paper (`paper/`)" section), `README.md` (Where things are)

**Interfaces:**
- Consumes: `paper/data.json`; in Typst `#let D = json("data.json")` and `#let v(k) = D.at(k)`.

Sections, page budget, figures and sources exactly as the spec's Paper table. Fig. 1 and Fig. 2 are
drawn in Typst (boxes/arrows; dots with interval lines and a zero line) from `data.json`, no extra
package. Table I: the worked payment's shifted logits and probabilities in both orders. Table II:
final-test setups (mean cost, interval, good customers declined, frauds stopped). Bias section covers
all nine biases in the spec. Body text follows Global Constraints; file paths only in a "Code and
data" footnote. Inferences say "suggests"/"we infer". Use "softmax (multinomial logistic)" at most once.

- [ ] **Step 1: Failing tests**: `test_paper_compiles_and_is_six_to_eight_pages` (skip if `typst`
  isn't on PATH; compile to PNG pages in `tmp_path` with `typst compile paper/main.typ {tmp}/p{p}.png`;
  6 ≤ page count ≤ 8); `test_paper_has_the_fourteen_references` (14 bibliography entries in
  `main.typ`, [14] names d1-Omni-2B); `test_paper_text_follows_the_rules` (Global Constraints
  patterns absent outside the footnote).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Write** `main.typ` section by section; `python paper/build_data.py` after each new key.
- [ ] **Step 4: Verify.** `pytest tests/test_paper.py -v` PASS; compile to
  `paper/exports/transforming-llms-fraud.pdf`; render page PNGs and read every page (layout, figures,
  no overflow into margins, references complete).
- [ ] **Step 5: Commit** "paper: the IEEE paper on turning a small language model into a fraud decision model".

### Task 8: The primer

**Files:** Create `docs/econometrics-primer.md`; Modify `README.md` (one row under docs)

Eight topics in the spec's order. Each: the idea in two or three sentences; a worked example with
small numbers (for 2, use the worked payment's numbers from `slides/data.js`, named as such); where it
appears (slide number, paper section); "if he asks" (one or two lines). Plain words, no equations
beyond one line each. Topic 3 builds the two-identical-graphs example: one feature, raw and
standardised, same fit, different-looking weights.

- [ ] **Step 1: Write** the primer.
- [ ] **Step 2: Check** each "where it appears" against the built deck and paper (slide numbers and
  section names exist).
- [ ] **Step 3: Commit** "docs: a primer on the statistics behind the methods, for the presenters".

### Task 9: Whole-project check

- [ ] **Step 1:** `.venv/Scripts/python -m pytest -m "not model"`. Expected: all PASS.
- [ ] **Step 2:** Regenerate `slides/data.js`, `paper/data.json`, the deck PDF and the paper PDF;
  `git status` shows no unexpected changes.
- [ ] **Step 3:** Update memory `final-deck.md` (new deck, paper, primer; old deck in history).
- [ ] **Step 4: Commit** any regenerated files: "final deck and paper: regenerated outputs".
