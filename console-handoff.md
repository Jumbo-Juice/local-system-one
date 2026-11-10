# Console handoff: readable eval pages, mean lines and a cost map

For a new session. Make the fraud analyst console (`fraud/viewer/`) easier to read and add one page that shows how the result moves when the assumed costs change. The console may be shown live to a machine-learning professor, next to the deck in `slides/`.

Written 2026-11-10. Repo: this folder, branch `main`. Read `fraud/CLAUDE.md` first: its hard rules apply to everything here.

## Where things stand

- The audit of the deck and the console is done and committed with this file. Every statistic in `fraud/results/` was recomputed from the per-window costs: 843 checks, 0 mismatches. `pytest -m "not model"` passed with 436 tests.
- None of the work below is started. `fraud/viewer/console.js` holds only the audit's fixes.
- Leave these alone: `fraud/SUMMARY.md` has the owner's own uncommitted table re-formatting, and `AGENTS.md` and `GLOSSARY.md` are the owner's untracked files.

## What the owner asked for

The owner's words, 2026-11-10:

- "make the console more readable"
- "do you think we should change the cost of review. maybe a slider that changes the cost of review dynamically so we can highlight the bias that comes from different cost"
- "also in the scatter plot there should be a horizontal average line in noisy data"

## Decided by the owner

1. **The review stays at 10 USD everywhere.** No result is re-priced. The owner chose "Phase 16 cost map": one new console page over the cost sets Phase 16 already tested.
2. **All four readability fixes:** long setup names, text before numbers, no picture of the intervals, tables too dense.
3. **A horizontal average line on the per-window scatter.**

## Proposed, not confirmed

The owner was asked these four and sent the mean-line request instead of answering. They are my recommendations. Build them, say so in your report, and change them if the owner objects.

| question | proposal | why |
|---|---|---|
| Which cost assumptions | All six families, review fee open by default | The review fee flips nothing as deployed. The price per declined good customer does, so the page needs it to show any bias |
| Which view | "As deployed" and "re-picked" side by side | The gap between them is the lesson: a rule that isn't told its true costs is the one that goes wrong |
| Which scorer | Logreg by default, the adapter on a switch | Logreg is Phase 16's registered primary. The adapter is Phase 12's, not a final one, so label it that way |
| The deck | Unchanged | It already cites Phase 16 on the slide-7 backup, and the talk has no spare time |

## Task 1: the cost map page

**Data (Observed, committed):** `fraud/results/eval_cost_stress/20261103-210247.json`, written by `fraud/eval_cost_stress.py`. Phase 16, pre-registered in that script's docstring. It re-prices Phase 14's recorded scores on Phase 14's 20 test windows. No model ran. The console has no page for it today.

Shape of the file:

- `base_costs`: friction rate 0.1, friction minimum 1, review fee 10, review share 0.05.
- `families`: six lists, one entry per tested value. Each entry has `value`, `label`, `costs` and four blocks: `logreg_deployed`, `logreg_repicked`, `adapter_deployed`, `adapter_repicked`.

  | family | values |
  |---|---|
  | review fee | 1, 5, 10, 20, 30, 50, 75, 100, 200 |
  | review share | 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2 |
  | friction rate | 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1 |
  | friction minimum | 0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50 |
  | flat per good customer declined | 0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 150, 200 |
  | flat only, no percentage | 1, 2, 5, 10, 20, 30, 50, 75, 100, 200 |

- Each block: `rule_cost`, `thresholds_cost`, `diff`, `ci`, `rule_cheaper_windows`, `rule_wins`, `distinguishable`, `better`, `rule_good_declined`, `thresholds_good_declined`, `rule_frauds_stopped`, `thresholds_frauds_stopped`, and the fraud USD stopped for each. Re-picked blocks also have `lambda` and `thresholds`.
- **Sign:** `diff` is the rule minus two lines, so a plus means the rule costs more. This is not the eval files' convention, where `mean_diff` is signed from the cheaper side. Don't reuse the eval pages' table code for it without checking.
- `stops`: per family and block, `below_base` and `above_base`, each with `first_not_lower` and `first_not_distinguishably_lower`. Null means it never happens in the tested range.
- `grid`: 30 cells of friction rate by flat price, logreg only. Optional. Leave it out unless it comes cheaply.

**"As deployed"** means Phase 14's decisions are unchanged and only the prices move. **"Re-picked"** means the rule is told the new cost: lines and lambda are re-picked on the 30 dev windows.

**Numbers to check the page against (Observed, logreg, USD per window, rule minus two lines):**

| review fee | as deployed | 95% CI | re-picked | 95% CI |
|---:|---:|---|---:|---|
| 1 | -716 | -1,196 to -353 | -696 | -1,170 to -333 |
| 10 (the base) | -825 | -1,307 to -464 | -818 | -1,283 to -470 |
| 50 | -1,309 | -1,805 to -918 | -379 | -852 to -24 |
| 75 | -1,612 | -2,132 to -1,170 | -151 | -434 to 115, not distinguishable |
| 200 | -3,124 | -3,935 to -2,327 | -86 | -282 to 125, not distinguishable |

| flat per declined good customer | as deployed | 95% CI | re-picked | 95% CI |
|---:|---:|---|---:|---|
| 0 (the base) | -825 | -1,307 to -464 | -818 | -1,283 to -470 |
| 30 | -335 | -862 to 86, not distinguishable | -821 | -1,265 to -454 |
| 60 | +156 | -423 to 656, not distinguishable | -798 | -1,271 to -402 |
| 100 | +810 | 151 to 1,427, distinguishable | -824 | -1,291 to -429 |

So the review fee flips nothing as deployed, and the flat price flips the result near 50 to 60 USD. Told the cost, the rule keeps its lead at every flat price tried. The page should let the reader see both.

**What to build (Implementation choice, change what reads better):**

- A route `#/costs` and a "Costs" link in the nav of `fraud/viewer/index.html`.
- The server must serve the file. Add `eval_cost_stress` to `EVAL_DIRS` in `fraud/viewer/server.py`. It won't be listed as an eval, because `eval_list` keeps only files with `totals` or `setups`. Add a test that says so. Find the file by listing or by a small API route; don't hard-code the run id in the page.
- A selector for the cost assumption, then a **stepped** slider over that family's tested values, with the base value marked. It can't be continuous: the console never re-decides rows.
- At the chosen value, two cards side by side, "as deployed" and "re-picked": the rule's cost, two lines' cost, the difference with its interval, distinguishable or not, windows cheaper of 20, good customers declined and frauds stopped for each.
- One chart across the whole family: the difference and its interval at every tested value, both views, a zero line, the chosen value marked. Space the values evenly and label them, because they aren't evenly spaced in dollars.
- One sentence from `stops`: where the rule's mean stops being lower, and where the interval stops excluding 0.
- A switch from logreg to the adapter, labelled "Phase 12's adapter, not a final one".
- The whole family as a table under a closed `<details>`.

**Labels the page must carry:**

- Earlier test: Phase 14's 20 test windows, re-priced from recorded scores. Not the final test, and not paired with it.
- Many cost sets, not corrected for multiple comparisons: a map, not separate findings. The docstring says this.
- Every price is an assumption nobody measured.
- The rule against two lines on the same scores. It does not compare the language model with logreg.

**Don't re-price the final test.** Its traces aren't on this machine, so only means could be shown, with no intervals, and the owner chose the option without that read-out.

## Task 2: readable eval pages

All in `ieeeEval` and its helpers in `fraud/viewer/console.js`, and `console.css`. The PaySim page `evalPage` can take the same treatment where it fits.

1. **Short setup names.** Names like "d1-Omni-2B with more prompt columns, no tabular score, amount-aware rule" wrap in every table. Add short names and use them in tables and charts. Keep the full name in a tooltip and in one legend table per page, with the setup id. Match the deck, so one thing has one name:

   | id | deck's name |
   |---|---|
   | `final-logreg-amt` | Baseline: logreg, rule |
   | `final-signal-amt` | d1 reads trees, rule |
   | `final-signal-cust` | d1 reads trees, customer line |
   | `final-rich-amt` | d1, no trees, rule |
   | `final-rich-cust` | d1, no trees, customer line |
   | `ieee-approve-all` | Approve everything |

   Most ids end in `-amt` (the rule), `-lines` (two lines) or `-cust` (the customer line), so a base name plus a suffix covers them. `setupName` already builds the audit and feature names from a pattern. `python <scratch>/facts.py` is gone with the session; list the ids again from `fraud/results/eval_*/2*.json`.
2. **Numbers before text.** Each page opens with a dense paragraph. Replace it with a verdict banner built from the data (the first registered question: the two setups, the difference, the interval, distinguishable or not), then fact cards with one fact per row, then the details. `fact`, `factCard`, `noteCard` and `.banner` exist for the run page: reuse them. Keep every caveat the paragraph has now: test or dev only, the assumed costs, uncalibrated scores, don't compare across evals.
3. **A picture of the intervals.** The answers are a table only. Draw each answer as a point with its interval and a zero line, like the deck's slide 11 (`scenes.forest` in `slides/js/scenes.js`). Comparisons against approve-all are ten times larger, so give them their own panel and scale, as the deck does. Keep the table under a closed `<details>`.
4. **Less dense tables.** Larger type. The setups table shows cost and customers by default, with AUC, recall, precision and declines one click away. The per-window table starts closed.

## Task 3: the mean line on the scatter

`evalChart` draws one dot per setup per window on a log axis. Add a horizontal line at each setup's mean, in its colour, with the value at the right edge.

- Use the mean the table shows (`mean_cost`), so the line and the table agree.
- **Trap:** on the log axis the four final setups, 1,943 to 2,427, sit a few pixels apart and their lines overlap. The log axis is there for PaySim, whose costs span orders of magnitude. For the IEEE-CIS pages a linear axis from 0 reads better (Implementation choice). Keep log for PaySim, and say on the chart which it is.
- Keep the labels from colliding: sort them and push them apart.
- The dev pages have 88 chains and are the noisy ones the owner means.

## Rules that apply

- **No inference in the console.** It only replays recorded files.
- **Scores are uncalibrated.** Never show one as a probability.
- **Don't pair costs across evals.** Different windows.
- **A difference counts only if the 95% window-bootstrap interval of the mean paired difference excludes 0.** Say "not distinguishable", never "slightly better".
- **Don't add the fraud app to the game viewer,** and don't reuse `viewer/shell.js`.
- **No new packages.** Hand-written SVG, as the page has now.
- Keep `RUN-GUIDE.md` current: the console section lists the pages and the folders they read.
- Commits have no attribution lines. Commit locally. Don't push unless the owner asks.

## Traps found during the audit

- `fraud/viewer/console.js` and `fraud/tests/test_viewer.py` have CRLF line endings in the working copy. Keep them.
- `fraud/tests/test_viewer.py` checks for exact strings in `console.js`. A rewrite must keep them or update the test with a reason: `function ruleCard`, `Amount-aware`, `Customer line`, `Two lines on the score`, `uncalibrated`, `fetch(`, `function setupName`, `all.length === scored || !/^dev\d+$/.test(w)`, `.slice(-scored)`, `rich_cust_readout.json`, `extras.json`, `"final-rich-cust"`, `read-out, not a test`, `The registered result above stands as it came out`, `held-out dev chain`.
- Phase 11's result (`eval_ieee/20261102-183431.json`) lists 30 windows, 10 dev and 20 test, but each setup has 20 costs. `ieeeEval` keeps only the scored ones. Don't undo that.
- The final test's page loads `eval_final/rich_cust_readout.json` and `extras.json` for its "After the test" panel. Keep the panel and its "read-out, not a test" label.
- In Git Bash here, a Python heredoc that contains a backslash or an apostrophe can fail to parse. Write the script to a file in the scratchpad and run it.
- After a hash change the page fetches `api/runs` for about 900 traces, so wait for the title to change before reading the page.

## Check it

```bash
.venv/Scripts/python -m pytest -m "not model"
```

Then start the `fraud-console` entry of `.claude/launch.json` and look, with no console errors, at:

- `#/`: the start page.
- `#/eval/20261104-090649`: the final test, with its "After the test" panel.
- `#/eval/20261104-090217`: a dev page with 88 chains, for the mean lines.
- `#/eval/20261102-183431`: Phase 11, whose window rows must still read test0 to test19.
- `#/eval/20261005-022333`: a PaySim page, still on a log axis.
- `#/costs`: every family, both views, both scorers, against the tables above.

At 375 px wide nothing scrolls sideways (`document.documentElement.scrollWidth`). Compare every number on the cost map with the JSON, not with this file.

## Done means

- The cost map page shows all six families from the committed file, with the labels above, and its numbers match the JSON.
- The eval pages open with a verdict and facts, use short names, draw the intervals, and keep their tables one click away.
- The scatter has a mean line per setup that can be told apart.
- `pytest -m "not model"` passes, with tests for the new page's route and for the file not being listed as an eval.
- `RUN-GUIDE.md` names the new page and folder.
- A local commit with no attribution lines.
