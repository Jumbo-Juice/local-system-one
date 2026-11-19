# Ideation: fraud console readability (2026-11-11)

## Subject

**Focus:** make the fraud analyst console (`fraud/viewer/`) readable to a newcomer. The owner's words: "any new person can look at and understand clearly with a bit of explanation. Even I who started the project is struggling to sort through the language."

**What the grounding found** (Observed; `js` = `fraud/viewer/console.js`):

- **The start page reads like a lab notebook.**
  - It opens with a model list: "A local model judges each TRANSFER and CASH_OUT (Qwen2.5-1.5B; later evals add…" (js:646).
  - Then comes a 15-row eval table, newest first, so the phases read 25, 23, 21, 22. It has a script-path column (js:668).
  - Then a filter of about 30 raw setup ids, and a table of 896 runs.
  - The only pinned replay is the frozen PaySim model that declines all 257 transactions it judges. Nothing frames it as the failure case.
- **The nav misleads.** "Eval" opens the newest eval, which is Phase 25 and dev-only (js:919). It is not the final test.
- **The final-test page gives the verdict before saying what it means.** "Not distinguishable" and "Mean difference … (the first minus the second)" come first; the definition comes after (js:1010-1014). No sentence on the page states the outcome in plain words.
- **One setup has up to three names**, kept by hand in SETUP_NAME, SHORT and SHORT_BASE/SHORT_RULE, plus a raw id. For example, "d1-Omni-2B reading a tabular score, amount-aware rule" (js:32) is the same setup as "d1 reads trees, rule" (js:51).
- **"Rule" means at least five different things**:
  - the rule filter;
  - the rules-only setup;
  - the amount-aware rule;
  - "Phase 14's rule";
  - "the rule fixed before the run".
- **Wrongly declining a good customer is said four ways**: "legit customers declined", "Wrong decline", "good customers declined" and "friction".
- **Sign conventions conflict.** The eval files sign differences from the cheaper side. The cost map uses "rule minus two lines" (js:1137-1139).
- **Phase numbers and file paths run through the prose**, for example "Phase 16: Phase 14's recorded scores re-priced…" (js:1283) and "Phase 12's adapter, not a final one" (js:1255).
- **The console defines none of its terms.** No glossary or help explains lambda, two lines, customer line or distinguishable.
- **GLOSSARY.md is untracked, and parts of it are wrong for this console.**
  - Window is defined as "about 500 payments" (GLOSSARY.md:47), but IEEE-CIS windows are 1,000.
  - "Decision" means the question put to the model (GLOSSARY.md:14-16), but the console's "Decision" panel shows the action.
- **Constraints**:
  - `fraud/tests/test_viewer.py` needs every setup id quoted literally in js (:83-86, :109-120).
  - It pins "Phase 12's adapter, not a final one" (:160) and "read-out, not a test" and "held-out dev chain" (:125-127).
  - RUN-GUIDE.md:534-548 describes the pages.
  - **No final-test traces exist on this machine.** The newest run traces are from Phase 14; PROGRESS.md:651 says the NUC traces were deleted after extraction. So the final test cannot be replayed.

**Method:** a grounding agent mapped the console and four frame agents generated ideas (28 in total). These were merged into 16 ideas and 4 combinations. A fresh agent then tried to refute each one by checking every quote and source, and this document records its verdicts. Verdicts: 7 survived, 9 weakened, 4 refuted or effectively refuted.

## Ranked ideas

### 1. One word per concept: fix the glossary first, then make the console speak it

- **Summary:** the console's language problem is several names for one thing, and one name for several things. So decide the words first, then rename. The order:
  1. Correct GLOSSARY.md: the Window size, and Decision vs Action.
  2. Add the missing terms with one-line definitions: the amount-aware rule, two lines (review/decline thresholds), customer line, lambda, distinguishable, read-out, dev chain.
  3. Give each meaning of "rule" its own name.
  4. Commit the glossary.
  5. Rename the console's strings to match, for example "good customers declined" everywhere.
  6. Lock the result with a test that lists the banned synonyms explicitly ("legit", "Wrong decline", "labels", "type filter", …). The test does not parse every `_Avoid_` line, because that is noisy.
- **Basis:**
  - `direct:` GLOSSARY.md:20 `_Avoid_: choice, label`, but js:369 says "Mass outside the labels" and js:554 "over the three labels".
  - `direct:` "legit customer declined" (js:344), "Wrong decline" (js:529) and "good customers declined" (js:1047) all name one cost.
  - `direct:` GLOSSARY.md:47 Window "about 500 payments", while IEEE-CIS windows are 1,000 rows.
  - `direct:` test_viewer.py:79-80 already bans "probability" in ruleCard; this widens that pattern.
- **Why it matters:** it treats the cause the owner names ("the language"), and the test stops the synonyms from drifting back. Ideas 2-4 all write sentences, and those sentences need these words.
- **Size:** M

```
/supermatt:grill-with-docs
Idea: give every concept in the fraud console (fraud/viewer/console.js) exactly one word, fix GLOSSARY.md first, rename the console strings to match, and lock it with a test of explicitly banned synonyms.
Basis: GLOSSARY.md:20 avoids "label" but console.js:369 says "Mass outside the labels" and :554 "over the three labels". One cost has three names: "legit customer declined" (console.js:344), "Wrong decline" (:529), "good customers declined" (:1047). "Rule" means the rule filter, rules-only, the amount-aware rule, "Phase 14's rule" and "the rule fixed before the run". GLOSSARY.md:47 says a Window is ~500 payments, but IEEE-CIS windows are 1,000. GLOSSARY.md:14-16 calls the model's question a "Decision", but the console's "Decision" panel (console.js:249) shows the action. GLOSSARY.md is untracked. Terms with no definition anywhere on the console: two lines, customer line, lambda, distinguishable, read-out, dev chain. fraud/tests/test_viewer.py:79-80 already bans "probability" in ruleCard; pinned strings at :79-80, :125-127 and :160 must survive or change with their tests.
Why it matters: the owner says even they struggle with the console's language; the root cause is synonyms and overloaded words, and the later readability work (plain-sentence verdicts, a questions-first start page) needs settled words.
```

### 2. Answer first: one plain sentence above every comparison, written by one helper

- **Summary:** add one `compare()` helper. It writes each comparison as a sentence, such as "A cost N USD less per window than B; the range runs from X less to Y more, so this test can't tell them apart". It uses one sign order everywhere and keeps signs only on chart axes.
  - **Where it's used:** the verdict banner, the answers chart and table, the after-test table and the cost-map cards.
  - **On eval pages:** the sentence comes first. "Distinguishable", the CI and the bootstrap definition go beneath it, following GRADE's summary-of-findings order: question, comparison, absolute effect, certainty, caveat.
  - **Caveat from the verifier:** the sentence must name its comparison. The page's primary question is final-signal-amt vs the baseline (−163, CI −462 to +238). The write-up's "Short answer" leads with final-signal-cust (−115). Unlabelled, the two would clash.
- **Basis:**
  - `direct:` js:1010-1014 puts "Not distinguishable" and "(the first minus the second), 95% CI" before "Distinguishable means the 95% window-bootstrap interval…".
  - `direct:` js:1137-1139 notes that the cost map's diff is "the rule MINUS two lines … unlike the eval files' mean_diff".
  - `direct:` js:1060 uses `money()`, an ASCII hyphen, where the banner uses `signed()`.
  - `direct:` slides/build_data.py:42 `def diff(entry, first)` already normalises the sign for the deck.
  - `external:` Guyatt et al., GRADE guidelines 12, J Clin Epidemiol 2013;66:158.
- **Why it matters:** the final-test page is the one the talk rests on (js:653), and today it never says its result in words.
- **Size:** M

```
/supermatt:grill-with-docs
Idea: in the fraud console, write every setup-vs-setup comparison as one plain sentence produced by a single compare() helper, with one sign convention, placed above the statistics (GRADE summary-of-findings order).
Basis: fraud/viewer/console.js:1010-1014 shows "Not distinguishable" and "Mean difference … (the first minus the second), 95% CI" before defining "distinguishable". console.js:1137-1139 says the cost map's diff is "the rule MINUS two lines … unlike the eval files' mean_diff, which is signed from the cheaper side". console.js:1060 formats with money() (hyphen) while the banner uses signed() (−/+). slides/build_data.py:42 diff(entry, first) already normalises the sign for the deck. fraud/RESULTS-FINAL.md:6 has a prose "Short answer", but the console never shows one. Caveat: the page's primary is final-signal-amt vs baseline (−163, CI −462 to +238, cheaper in 15 of 20); the write-up's short answer leads with final-signal-cust (−115), so each sentence must name its comparison. External: Guyatt et al., GRADE guidelines 12, J Clin Epidemiol 2013;66:158.
Why it matters: the final test is the page the talk rests on, and today no sentence on it states the outcome.
```

### 3. A start page built on the reader's questions, with the archive on its own page

- **Summary:** replace the model-list intro, the phase-shuffled eval table and the 896-row runs table with three blocks:
  1. **What it does:** payment, then rule filter, then model, then approve/review/decline, then cost.
  2. **What we found:** four or five reader questions, each with one answer sentence (from idea 2) and a link to its evidence. For example: does the language model beat the simple baseline? What does it do to good customers? Does it hold if prices change? Is it fast enough? What went wrong along the way?
  3. **Where to look.**

  The other pieces move or change:
  - Evals and runs move to `#/archive`, grouped by data set, in phase order, with no script column.
  - Eval descriptions change from "Phase N: …" to the question each eval asked, as the deck already does.
  - The frozen-model replay gets framed as "where this started".
- **Basis:**
  - `direct:` js:646 opens with "A local model judges each TRANSFER and CASH_OUT (Qwen2.5-1.5B; later evals add…".
  - `direct:` js:668 gives the columns `["eval", "what it is", "data and review fee", "script"]`.
  - `direct:` js:88 `"Phase 17: the C, D, V column audit, 8 test windows"`, while slides/index.html:249 says "Which columns does it need?".
  - `direct:` fraud/runs/README.md:13: the pinned run "declines all 257 transactions it judges".
- **Why it matters:** a newcomer's first screen decides whether the console reads as a story or a file browser, and today it is the file browser. No test pins the start page; RUN-GUIDE.md:534-548 needs updating.
- **Size:** M

```
/supermatt:grill-with-docs
Idea: rebuild the fraud console's start page around the reader's questions (what it does, what we found as 4-5 question/answer pairs, where to look), and move the eval and run tables to a separate #/archive page; describe each eval by the question it asked instead of "Phase N".
Basis: fraud/viewer/console.js:646 opens with a model list ("A local model judges each TRANSFER and CASH_OUT (Qwen2.5-1.5B; later evals add…"). console.js:668 shows a script-path column. console.js:84-93 (EVAL_WHAT) labels evals "Phase N: …", listed newest first so the phases read 25, 23, 21, 22. The deck names the same phases by question (slides/index.html:249 "Which columns does it need?"). fraud/runs/ has 896 runs in one unpaged table. The only pinned replay (fraud/runs/README.md:13) is the frozen PaySim model that "declines all 257 transactions it judges", shown unframed. No final-test traces exist locally (PROGRESS.md:651), so any replay link must point to PaySim or Phase 11-14 IEEE-CIS runs. No test pins the start page; RUN-GUIDE.md:534-548 must be updated.
Why it matters: the first screen decides whether a newcomer or a professor ever gets oriented; today it reads like the author's lab notebook.
```

### 4. Setup names built from parts; comparisons name only the part that differs

- **Summary:** describe each setup id by three parts: **model** (logreg, trees, d1-Omni-2B fine-tuned, …), **what it reads** (all columns, a tree score, …) and **how it decides** (two lines, amount-aware rule, rule + customer line). Build the full name, the short name and table columns ("Model / Reads / Decides with") from those parts.
  - **Exceptions table:** ids that don't fit the pattern go here, for example `ieee-logreg-all`, `*-dev30`, `rule-customers` and `lm-rich+trees`.
  - **Comparison labels:** a comparison then names only what changes, for example "d1, no trees: customer line vs none". The deck already labels comparisons this way by hand.
  - **Constraint:** keep every id as a quoted key, so test_viewer.py:83-86 and :109-120 still pass.
- **Basis:**
  - `direct:` js:60: "Most ids are a base plus -amt (the rule), -lines (two lines) or -cust (the customer line)."
  - `direct:` js:17 and js:21 give two different ids the same full name.
  - `direct:` js:61-68 (SHORT_BASE × SHORT_RULE) already composes short names this way, so this extends work that exists.
  - `direct:` slides/js/scenes.js:370 `'d1, no trees: line <span class="dim">vs no line</span>'`, while console.js:1011 prints both full short names joined by "vs".
- **Why it matters:** one source for each name means it can't drift, adding a setup means adding one row, and every comparison reads as the single change being tested.
- **Size:** M

```
/supermatt:grill-with-docs
Idea: describe every fraud-console setup id by three parts (model, what it reads, how it decides), generate full names, short names and table columns from them, and label each comparison by only the part that differs.
Basis: fraud/viewer/console.js:60 says "Most ids are a base plus -amt (the rule), -lines (two lines) or -cust (the customer line)". console.js:61-68 (SHORT_BASE × SHORT_RULE) already composes short names this way, but SETUP_NAME (console.js:12-35) is hand-written. console.js:17 and :21 give two ids the same full name. A setup can carry three names, e.g. "d1-Omni-2B reading a tabular score, amount-aware rule" (:32) and "d1 reads trees, rule" (:51). Ids that don't fit the pattern need an exceptions table: ieee-logreg-all, *-dev30, rule-customers, rule-paced, ieee-ft-ieee-d1omni, lm-rich+trees. The deck labels comparisons by the differing part (slides/js/scenes.js:370 "d1, no trees: line vs no line"), while console.js:1011 joins two full short names. fraud/tests/test_viewer.py:83-86 and :109-120 require every committed setup id quoted literally in console.js.
Why it matters: today the reader has to decode comma phrases like "d1, no trees, customer line minus d1, no trees, rule"; parts-based names give one name per setup and comparisons that read as the one thing tested.
```

### 5. Phase numbers and file paths go into one "Where this came from" fold per page

- **Summary:** pass phase numbers, script paths, result-file paths, eval timestamps and adapter directories through one `prov()` helper. It renders them into a closed `<details>` at the foot of each page, and headings get plain descriptions in their place.
  - Keep the pinned strings inside `prov()`, so tests :125 and :160 still pass.
  - Keep "Pre-registered in …" visible, because it signals integrity.
  - Before adding any regex test, rename the setup names that contain phases ("logreg, Phase 14's rule", js:57; "d1, Phase 12 adapter", js:63).
- **Basis:**
  - `direct:` js:1283 "Phase 16: Phase 14's recorded scores re-priced under other costs. Run ${r.run}, scores from eval ${r.scores_from_eval}."
  - `direct:` js:1255 "Phase 12's adapter, not a final one".
  - `direct:` js:1073 "Source: fraud/results/eval_final/rich_cust_readout.json and extras.json; …".
- **Why it matters:** phase numbers are the author's index, not the reader's words. They are why the returning owner has to re-read PROGRESS.md.
- **Size:** S

```
/supermatt:grill-with-docs
Idea: move phase numbers, script and result-file paths, eval timestamps and adapter dirs out of the fraud console's headings and hints into one closed "Where this came from" fold per page, rendered by a single prov() helper.
Basis: fraud/viewer/console.js:1283 "Phase 16: Phase 14's recorded scores re-priced under other costs. Run ${r.run}, scores from eval ${r.scores_from_eval}."; console.js:1255 "Phase 12's adapter, not a final one"; console.js:1251 "(the base, Phase 14's costs)"; console.js:1073 "Source: fraud/results/eval_final/rich_cust_readout.json and extras.json". Setup names that contain phases (console.js:57 "logreg, Phase 14's rule", :63 "d1, Phase 12 adapter") need renaming first. Pinned strings fraud/tests/test_viewer.py:125 and :160 can stay inside prov(). Keep "Pre-registered in …" visible as an integrity signal.
Why it matters: pages explain one thing by pointing at another phase number, which a newcomer can't follow and the owner has to look up.
```

### 6. The nav names its destinations, and "Final test" goes to the final test

- **Summary:** change the nav to "Start / Final test / All evals / Cost map / Auto-demo". "Final test" points to the final eval, which the start page already finds (js:610). Today "Runs" opens the start page, and "Eval" opens Phase 25, a dev-only page marked "Reads here are flattered". Keep `href="#/costs"`, which test :159 pins.
- **Basis:**
  - `direct:` html:13 `<nav><a href="#/">Runs</a><a href="#/eval">Eval</a>…`
  - `direct:` js:919 `evals[0]`
  - `direct:` js:1094 "Reads here are flattered: lambda was picked with these rows in."
- **Why it matters:** this is the cheapest fix in the list. It stops a newcomer, or a live demo, landing on the least trustworthy page with the first click.
- **Size:** S

```
/supermatt:grill-with-docs
Idea: rename the fraud console's nav to Start / Final test / All evals / Cost map / Auto-demo and point "Final test" at the final eval instead of the newest one.
Basis: fraud/viewer/index.html:13 nav reads "Runs" (→ #/, the start page) and "Eval" (→ #/eval). fraud/viewer/console.js:919 opens evals[0], currently Phase 25 (dev only), which warns "Reads here are flattered: lambda was picked with these rows in" (console.js:1094). The start page already finds the final test with evals.find((e) => e.script === "fraud/eval_final.py") (console.js:610). fraud/tests/test_viewer.py:159 pins href="#/costs" and h === "costs". RUN-GUIDE.md:534-548 describes the pages.
Why it matters: the first click from the nav should land on the result the talk rests on, not a dev-only page.
```

### 7. A run's cost is shown next to what approving everything, and the baseline, would have cost

- **Summary:** the run banner says "Cost X", with nothing to compare it to. The trace's end record already holds `summary.fraud_amount`, which equals the approve-all cost. Add "Approving everything would have cost Y on this window", plus the logreg cost on the same window where a trace exists. The logreg figure matters because approve-all alone flatters the failure run: 4.38M against 46.3M. No inference runs.
- **Basis:**
  - `direct:` js:500 `` `Cost ${money(s.cost)} · …` ``
  - The verifier checked that `summary.fraud_amount` equals the approve-all cost on IEEE windows amt0 (6,059.113) and amt5, and on PaySim test100 (36,363,088).
  - `direct:` js:1120: the eval pages already compare "Against approving everything".
- **Why it matters:** it turns the biggest number on the page where a newcomer spends the most time into something they can judge themselves.
- **Size:** S

```
/supermatt:grill-with-docs
Idea: on the fraud console's run page, show the run's cost next to two reference points from recorded data: what approving everything would have cost on the same window (summary.fraud_amount in the trace's end record) and the logreg baseline's cost on the same window where its trace exists.
Basis: fraud/viewer/console.js:500 banner `Cost ${money(s.cost)} · …` gives no scale. The trace end record's summary.fraud_amount equals the approve-all cost (checked: IEEE amt0 6,059.113 and amt5; PaySim test100 36,363,088). Every tx row carries amount, truth and weight. console.js:1120 eval pages already compare "Against approving everything". Approve-all alone flatters the pinned failure run (4.38M vs 46.3M), so add the baseline on the same window (dev logreg traces exist). Replay only, no inference (fraud/CLAUDE.md).
Why it matters: a raw dollar total on a 500- or 1,000-row window means nothing to a newcomer without a reference.
```

## Cut

| idea | reason |
|---|---|
| Names file shared by console and deck | Cost over value. slides/build_data.py holds no names today, so every name in the deck templates would need rebinding. It would also break test_viewer.py:83-86, :109-120 and :125, and couple fraud/viewer to slides/ |
| Vocabulary test parsed from every GLOSSARY.md `_Avoid_` line | Basis weakened. Most hits are false positives: "Dev gate", "Prompt tokens", "batches of 8", and the required "not probabilities". Reworked into #1 as an explicit banned-synonym list |
| In-console hover definitions read from GLOSSARY.md | Basis weakened. The glossary's Window definition is wrong for IEEE-CIS, its "Decision" conflicts with the console, the server has no route for the repo-root file, and the file is untracked. Fixing the glossary is folded into #1 |
| Eval descriptions as questions, lab-notebook page | Duplicate of a stronger idea (merged into #3) |
| Guided tour replacing the shuffled auto-demo | Basis refuted for its key stops. No final-test traces exist locally (PROGRESS.md:651). Pinning IEEE-CIS rows raises a redistribution question, since the repo documents a licence only for PaySim |
| One window's cost as three piles of dollars | Basis weakened. Per-window cost parts for the final test aren't in the committed results; it is feasible only on Phase 14 or PaySim windows |
| Presenter view with a global detail toggle | Cost over value. It would be a second hiding mechanism next to the existing `<details>`, and #3 and #5 remove most of the clutter it targets |
| Titles that state the finding | Basis weakened. The example overclaims: the 50-60 USD flip holds only for logreg as deployed with a flat price. A static H1 can't state one finding on a page with a scorer toggle and two views. Our World in Data titles are mostly descriptive |
| CONSORT-style funnel of setups | Basis weakened. Earlier phases answered separate questions on different data and costs, so they don't form one funnel. Only the IEEE end (lm_nuc → combos → final) is recorded, and gate_nuc isn't served |
| Comparison labels by the differing part | Duplicate of a stronger idea (merged into #4) |
| C1 one vocabulary spine (descriptor + deck file + labels) | Cost over value. It inherits the shared-names-file problems |
| C2 questions-first console | Duplicate of a stronger idea (it is #3, with #2's sentences) |
| C3 language guardrails (avoid-test + hover + provenance test) | Basis weakened. It inherits the false positives, the untracked and partly wrong glossary, and a regex that flags setup names. Its sound parts are #1 and #5 |
| C4 tour that teaches the cost | Basis refuted. There are no final-test traces to tour |
