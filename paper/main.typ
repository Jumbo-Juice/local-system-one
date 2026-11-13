// The paper: Transforming LLMs into classifiers for fraud detection. IEEE two-column, built with Typst.
// Every number comes from data.json through v() with a "path|format" key; run `python paper/build_data.py` after adding a key.
#import "vendor/charged-ieee/lib.typ": ieee

#let D = json("data.json")
#let v(k) = D.at(k)

// Citations as IEEE numbers; the list itself is at the end (references: start/end).
#let cite-n(..ns) = ns.pos().map(n => "[" + str(n) + "]").join(", ")
#let bib(n, body) = grid(columns: (2.2em, 1fr), column-gutter: 0.2em, [\[#n\]], body)

// code-and-data: start
#let code-and-data = footnote[Code and data: every number in this paper is read from #raw("paper/data.json"), which #raw("paper/build_data.py") builds from the deck's build, #raw("slides/build_data.py"). That reads the committed results: the final test #raw(v("_sources.final")), the read-out after it #raw(v("_sources.readout")), the dev runs #raw(v("_sources.nuc")), the earlier tests #raw(v("_sources.amount")), #raw(v("_sources.rule")), #raw(v("_sources.audit")) and #raw(v("_sources.tabular")), the pinned PaySim trace #raw(v("_sources.paysim_trace")), and prose-only numbers with their sources in #raw(v("_sources.manual")). Write-ups: #raw("fraud/SUMMARY.md"), #raw("fraud/RESULTS-FINAL.md"), #raw("fraud/RESULTS.md").]
// code-and-data: end

// Ruling 8: the IEEE look needs a Times face; Windows has Times New Roman, TeX Gyre Termes is not installed.
// The set rule after the template covers the body; a show-set on text also reaches the title, authors and abstract,
// which the template lays out itself. Spaces next to symbols are not text elements, hence both, and the abstract
// wrapped in its own text().
#let times = ("TeX Gyre Termes", "Times New Roman")
#show text: set text(font: times)
#show place: set text(font: times)
#show footnote.entry: set text(font: times)

#show: ieee.with(
  title: [Transforming LLMs into Classifiers for Fraud Detection],
  authors: (
    (name: "Juji Vongvilay", organization: [Group 13, Frog Watch]),
    (name: "Thu Ya Aung", organization: [Group 13, Frog Watch]),
    (name: "Chanpichsoponha Samnang", organization: [Group 13, Frog Watch]),
    (name: "Sai Thit Swan Thu", organization: [Group 13, Frog Watch]),
  ),
  abstract: text(font: times)[
    Every payment needs a decision before the money moves: approve, send to a person for review, or decline. We ask whether a small language model, run on one machine, can make that decision, and whether it is worth it in dollars. We turn the model into a classifier: the payment is written as plain facts followed by one lettered multiple-choice question, the model reads it in one forward pass, and its scores for the three answer letters become three probabilities through a softmax. A small low-rank adapter teaches it which letter to give. On simulated PaySim payments, training took the model from losing #v("manual.paysim.mean_untrained_millions") million USD per window to #v("manual.paysim.mean_trained|int") USD, close to hand-written rules. On #v("final.windows_n") windows of real IEEE-CIS payments, locked away and scored once, no setup showed a distinguishable saving against logistic regression with the same cost rule. The language model alone cost #v("final.diffs.rich_amt_baseline.mean|abs") USD per window more (#v("manual.conventions.ci_level")% interval #v("final.diffs.rich_amt_baseline.ci|ci")); the setup that read a boosted-tree score cost #v("final.diffs.signal_amt_baseline.mean|signed") (#v("final.diffs.signal_amt_baseline.ci|ci")) but mostly follows that score. What held: far fewer good customers declined with a flat price per declined customer in the rule, a 90th-percentile time of #v("final.latency.p90|int") ms per decision for the language model alone. The costs are assumed, and fairness was not measured.
  ],
  index-terms: (text(font: times)[fraud detection, language models, multiple-choice scoring, low-rank adaptation, cost-sensitive decisions],),
  bibliography: none,
)
#set text(font: times)


// ---- figures, drawn here from data.json (no package) -------------------------------------------------------

#let minus(x) = if x < 0 { "\u{2212}" + str(calc.abs(x)) } else { str(x) }

// Dots with interval lines and a zero line: one row per setup, each a paired difference against logistic regression.
#let interval-chart(rows, axis) = layout(size => {
  let w = size.width
  let label-w = 0.36 * w
  let val-w = 0.09 * w
  let plot-w = w - label-w - val-w
  let lo0 = calc.min(0.0, ..rows.map(r => r.lo))
  let hi0 = calc.max(0.0, ..rows.map(r => r.hi))
  let pad = (hi0 - lo0) * 0.05
  let (lo, hi) = (lo0 - pad, hi0 + pad)
  let x(val) = (val - lo) / (hi - lo) * plot-w
  let row-h = 21pt
  let plot-h = rows.len() * row-h
  let step = if hi - lo > 2400 { 1000 } else if hi - lo > 1200 { 500 } else { 250 }
  let ticks = range(int(calc.ceil(lo / step)), int(calc.floor(hi / step)) + 1).map(k => k * step)
  block(width: w, height: plot-h + 22pt, {
    for t in ticks {
      place(dx: label-w + x(t), dy: 0pt, line(start: (0pt, 0pt), end: (0pt, plot-h), stroke: (paint: luma(200), thickness: 0.4pt, dash: "dotted")))
      place(dx: label-w + x(t) - 15pt, dy: plot-h + 2pt, box(width: 30pt, align(center, text(6.5pt, minus(t)))))
    }
    place(dx: label-w + x(0), dy: 0pt, line(start: (0pt, 0pt), end: (0pt, plot-h), stroke: 0.8pt + black))
    place(dx: label-w, dy: plot-h, line(start: (0pt, 0pt), end: (plot-w, 0pt), stroke: 0.4pt + luma(120)))
    place(dx: label-w - 20pt, dy: plot-h + 12pt, box(width: plot-w + 40pt, align(center, text(6.5pt, axis))))
    for (i, r) in rows.enumerate() {
      let y = i * row-h + row-h / 2
      place(dx: 0pt, dy: y - row-h / 2, box(width: label-w - 6pt, height: row-h, {
        set par(first-line-indent: 0pt, justify: false, leading: 0.35em)
        align(right + horizon, text(7pt, r.label))
      }))
      place(dx: label-w + x(r.lo), dy: y, line(start: (0pt, 0pt), end: (x(r.hi) - x(r.lo), 0pt), stroke: 1.4pt + luma(110)))
      for e in (r.lo, r.hi) {
        place(dx: label-w + x(e), dy: y - 2.5pt, line(start: (0pt, 0pt), end: (0pt, 5pt), stroke: 0.8pt + luma(110)))
      }
      place(dx: label-w + x(r.mean) - 2.4pt, dy: y - 2.4pt, circle(radius: 2.4pt, fill: black))
      place(dx: label-w + plot-w + 4pt, dy: y - 4pt, text(7pt, r.said))
    }
  })
})

#let stage(head, body) = rect(width: 100%, inset: 4pt, stroke: 0.5pt + luma(90), radius: 2pt, {
  set text(7pt)
  set par(first-line-indent: 0pt, justify: false, leading: 0.4em)
  strong(head)
  linebreak()
  body
})
#let arrow = align(center + horizon, text(9pt, sym.arrow.r))

#let fig-pipeline = [#figure(
  placement: top,
  scope: "parent",
  grid(
    columns: (1fr, auto, 1fr, auto, 1.25fr, auto, 1fr, auto, 0.9fr, auto, 0.9fr, auto, 1.05fr),
    column-gutter: 3pt,
    align: horizon,
    stage[Input][A PaySim #v("worked.type") of #v("worked.amount|usd2") USD to a #v("worked.to"), hour #v("worked.hour")],
    arrow,
    stage[Plain facts][Receiver known: paid #v("worked.facts.receiver_before") times before. Amount not seen earlier this hour. Not round.],
    arrow,
    stage[Lettered question][#v("worked.prompt.question") \ A #v("worked.prompt.options.0") \ B #v("worked.prompt.options.1") \ C #v("worked.prompt.options.2")],
    arrow,
    stage[One forward pass][Qwen2.5-1.5B-Instruct #cite-n(10) reads the text once and scores every possible next token #cite-n(5)],
    arrow,
    stage[Three logits][Keep the scores of A, B, C, shifted so the top is zero: #v("worked.logits.listed.0|d2"), #v("worked.logits.listed.1|d2"), #v("worked.logits.listed.2|d2")],
    arrow,
    stage[Softmax][Three probabilities that add up to one: #v("worked.orders.listed.0|d2"), #v("worked.orders.listed.1|d2"), #v("worked.orders.listed.2|d2")],
    arrow,
    stage[Action][Untrained: the top letter. Trained: two cut-offs (PaySim) or the cost rule (IEEE-CIS)],
  ),
  caption: [The language model as a decision model, on the worked payment (PaySim, simulated; the model not yet trained). It writes nothing: one pass gives one score per letter. Logits and probabilities are for the options as listed; Table I has both orders.],
) <pipeline>]

#let fig-final = [#figure(
  placement: top,
  interval-chart(
    (
      (label: [language model + tree score, rule], mean: float(v("final.diffs.signal_amt_baseline.mean")), lo: float(v("final.diffs.signal_amt_baseline.ci.0")), hi: float(v("final.diffs.signal_amt_baseline.ci.1")), said: v("final.diffs.signal_amt_baseline.mean|signed")),
      (label: [language model + tree score, customer line], mean: float(v("final.diffs.signal_cust_baseline.mean")), lo: float(v("final.diffs.signal_cust_baseline.ci.0")), hi: float(v("final.diffs.signal_cust_baseline.ci.1")), said: v("final.diffs.signal_cust_baseline.mean|signed")),
      (label: [language model alone, rule], mean: float(v("final.diffs.rich_amt_baseline.mean")), lo: float(v("final.diffs.rich_amt_baseline.ci.0")), hi: float(v("final.diffs.rich_amt_baseline.ci.1")), said: v("final.diffs.rich_amt_baseline.mean|signed")),
      (label: [language model alone, customer line\*], mean: float(v("final.diffs.rich_cust_baseline.mean")), lo: float(v("final.diffs.rich_cust_baseline.ci.0")), hi: float(v("final.diffs.rich_cust_baseline.ci.1")), said: v("final.diffs.rich_cust_baseline.mean|signed")),
    ),
    [USD per window, setup minus logistic regression],
  ),
  caption: [The final test: each setup's cost per window minus logistic regression's, both deciding the same #v("final.windows_n") windows. Dot: the mean difference; line: its #v("manual.conventions.ci_level")% interval. Left of the zero line is cheaper. Every line crosses zero. \*Read after the test, not a test result.],
) <final>]

#let table-worked = [#figure(
  placement: top,
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, right, right, center, right, right, right),
    stroke: none,
    inset: (x: 3pt, y: 2.2pt),
    table.hline(stroke: 0.6pt),
    table.header(
      [], table.cell(colspan: 3)[as listed], table.cell(colspan: 3)[reversed], [average],
      [answer], [letter], [logit], [prob.], [letter], [logit], [prob.], [prob.],
    ),
    table.hline(stroke: 0.4pt),
    [approve], [A], [#v("worked.logits.listed.0|d2")], [#v("worked.orders.listed.0|d2")], [C], [#v("worked.logits.reversed.0|d2")], [#v("worked.orders.reversed.0|d2")], [#v("worked.scores.0|d2")],
    [review], [B], [#v("worked.logits.listed.1|d2")], [#v("worked.orders.listed.1|d2")], [B], [#v("worked.logits.reversed.1|d2")], [#v("worked.orders.reversed.1|d2")], [#v("worked.scores.1|d2")],
    [decline], [C], [#v("worked.logits.listed.2|d2")], [*#v("worked.orders.listed.2|d2")*], [A], [#v("worked.logits.reversed.2|d2")], [#v("worked.orders.reversed.2|d2")], [*#v("worked.scores.2|d2")*],
    table.hline(stroke: 0.6pt),
  ),
  caption: [The worked payment's logits (shifted so the top is zero) and softmax probabilities, with the options as listed and reversed (untrained Qwen2.5-1.5B, PaySim)],
) <worked>]

#let table-final = [#figure(
  placement: top,
  scope: "parent",
  {
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (1fr, auto, auto, auto, auto, auto, auto, auto),
    align: (left + horizon, right + horizon, center + horizon, right + horizon, center + horizon, left + horizon, right + horizon, right + horizon),
    stroke: none,
    inset: (x: 5pt, y: 2.6pt),
    table.hline(stroke: 0.6pt),
    table.header(
      [setup, each with the cost rule], [mean cost], [#v("manual.conventions.ci_level")% interval], [minus logistic \ regression], [#v("manual.conventions.ci_level")% interval \ of the difference], [verdict], [good customers \ declined], [frauds stopped \ (of #v("final.frauds|int"))],
    ),
    table.hline(stroke: 0.4pt),
    [logistic regression (the baseline)], [#v("final.setups.baseline.mean|int")], [#v("final.setups.baseline.ci|range")], [], [], [], [#v("final.setups.baseline.good_declined|int")], [#v("final.setups.baseline.frauds_stopped|int")],
    [language model + tree score], [#v("final.setups.signal_amt.mean|int")], [#v("final.setups.signal_amt.ci|range")], [#v("final.diffs.signal_amt_baseline.mean|signed")], [#v("final.diffs.signal_amt_baseline.ci|ci")], [#v("final.diffs.signal_amt_baseline|dist")], [#v("final.setups.signal_amt.good_declined|int")], [#v("final.setups.signal_amt.frauds_stopped|int")],
    [language model + tree score, customer line], [#v("final.setups.signal_cust.mean|int")], [#v("final.setups.signal_cust.ci|range")], [#v("final.diffs.signal_cust_baseline.mean|signed")], [#v("final.diffs.signal_cust_baseline.ci|ci")], [#v("final.diffs.signal_cust_baseline|dist")], [#v("final.setups.signal_cust.good_declined|int")], [#v("final.setups.signal_cust.frauds_stopped|int")],
    [language model alone], [#v("final.setups.rich_amt.mean|int")], [#v("final.setups.rich_amt.ci|range")], [#v("final.diffs.rich_amt_baseline.mean|signed")], [#v("final.diffs.rich_amt_baseline.ci|ci")], [#v("final.diffs.rich_amt_baseline|dist")], [#v("final.setups.rich_amt.good_declined|int")], [#v("final.setups.rich_amt.frauds_stopped|int")],
    [language model alone, customer line\*], [#v("final.setups.rich_cust.mean|int")], [#v("final.setups.rich_cust.ci|range")], [#v("final.diffs.rich_cust_baseline.mean|signed")], [#v("final.diffs.rich_cust_baseline.ci|ci")], [#v("final.diffs.rich_cust_baseline|dist")], [#v("final.setups.rich_cust.good_declined|int")], [#v("final.setups.rich_cust.frauds_stopped|int")],
    [approve every payment (no model)], [#v("final.setups.approve_all.mean|int")], [#v("final.setups.approve_all.ci|range")], [], [], [], [#v("final.setups.approve_all.good_declined|int")], [#v("final.setups.approve_all.frauds_stopped|int")],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [The final test: #v("final.windows_n") windows, #v("final.rows|int") real payments, #v("final.frauds|int") frauds, scored once. Costs in USD per window. The language model is d1-Omni-2B, trained. \*Read after the test, not a registered result.],
) <finaltable>]

// ---- the paper --------------------------------------------------------------------------------------------

= Introduction

Every card or mobile payment needs a decision before the money moves. There are three: _approve_ it, send it to a person for a _review_, or _decline_ it. Each can be wrong in its own way. Approve a fraud and the money is gone; decline a genuine payment and a good customer is turned away; a review finds the truth but costs a fee, and only a few payments can get one. The stakes are large: card fraud losses worldwide were #v("manual.nilson.usd_billions") billion USD in #v("manual.nilson.year") #cite-n(1).

Fraud teams usually score each payment with a statistical model, such as logistic regression or boosted trees, and turn the score into an action with cut-offs. Large language models read text the way an analyst reads a case, but as chat assistants they write their answer word by word, usually on a remote server, and do not show how sure they are. This paper asks a narrower question: *can a small language model, running on one machine, be the decision maker itself, and is it worth it in dollars?*

Our approach treats the model as a classifier rather than a writer. Each payment is written as plain facts and followed by one multiple-choice question whose three answers, lettered A, B and C, are the three actions. The model reads the text once, and we keep only its scores for the three letters. We train a small adapter so that it gives the right letter, and we judge every method by what its decisions cost, not by how often they are right.

The contributions are: (i) a recipe that turns a language model into a three-way decision model, read in one forward pass, with its known weakness, a lean towards a letter's position, measured on simulated payments and cancelled there by averaging both option orders (on real payments the model read one order, chosen on dev); (ii) a two-cycle study, on simulated and then real payments, with every choice made on a development part and a final test kept locked away, chosen in advance and scored once; (iii) an honest account of the results and of the biases and limitations we know of.#code-and-data

The findings, in short: training is what makes the language model useful at all; on the real payments no setup showed a distinguishable saving against logistic regression; the cheapest-looking language setup mostly follows a tree score it is given; and the clearest gain, far fewer good customers declined, comes from the decision rule rather than the model.

= Related Work

Card fraud detection is a classification problem with three awkward features: fraud is rare, the data drift over time, and investigators can only check a few alerts. Dal Pozzolo et al. #cite-n(2) showed that a realistic evaluation must respect all three, which is why we split by time and limit reviews. On tabular data, tree ensembles remain hard to beat: Grinsztajn et al. #cite-n(3) found that tree-based models still outperform deep networks on typical tables. Boosted trees are therefore our strongest classical reference.

Language models can classify table rows once each row is written out as text. TabLLM #cite-n(4) serialises rows into sentences and asks a language model for the class, with few or no labelled examples. Robinson and Wingate #cite-n(5) showed that presenting the options with letters, so that one forward pass scores every option through its letter, works better than scoring each full answer separately; our question follows that form. Pezeshkpour and Hruschka #cite-n(6) found that language models are sensitive to the order in which options are listed, a bias we observe directly. Closest to this work, Tan, Ma and Zhang #cite-n(7) prompted language models on structured financial data for fraud, including the two datasets used here, with much larger models than ours, mostly without training, and found that they still lag specialised classifiers.

The idea of using a language model as a fast decision function, a model that never writes but chooses among options in one pass, comes from System One models #cite-n(13). The gap we address: among the studies reviewed, none reads a small model's answer scores in one pass, trains it to answer with a letter, and tests it in dollars on real payments against a locked final test.

= Data and Protocol

== Two datasets, two cycles
The work followed CRISP-DM #cite-n(12) in two cycles. The first used PaySim #cite-n(8), a simulator of mobile-money payments: #v("manual.data.paysim_rows|int") payments over #v("manual.data.paysim_days|num") days, about one fraud in #v("manual.data.paysim_one_in|int"). We built the whole pipeline and the training recipe there. Because PaySim is simulated, it can show that a pipeline works but not that it is worth using. The second cycle used IEEE-CIS #cite-n(9): #v("manual.split.rows|int") labelled e-commerce payments from Vesta, #v("manual.split.fraud_share") fraud. Many of its columns are anonymised counters (the C columns), time gaps (D) and engineered features (V).

== Split by time, windows, dev and the final test
IEEE-CIS was split by time and never shuffled, so no model learns from the future. The _fit_ part (#v("manual.split.fit|int") payments) is where models learn their weights. The _dev_ part (#v("manual.split.dev|int")) is where we choose: for example the cut-offs, the price of a review in the rule below, and which setups go forward. The _test_ part (#v("manual.split.test|int")) is never tuned on.

A _window_ is #v("manual.data.window_rows|int") payments in a row, in time order. Every method decides the same windows, so methods are compared window by window. The _final test_ is #v("final.windows_n") windows from the test part: #v("final.rows|int") payments with #v("final.frauds|int") frauds, worth #v("final.fraud_usd|int") USD. Its windows were reserved before this work began; the setups, the comparisons and the rule for choosing a winner were written down and committed before any setup scored a final-test row, and each setup was scored once. This is pre-registration: a result cannot steer the question it answers. Anything asked after the results were seen is labelled a _read-out_, not a test result.

== Judging by dollars
Accuracy, the share of payments labelled correctly, is the wrong yardstick here. On the final test a model that approves every payment and looks at nothing is #v("final.approve_all_accuracy|pct") accurate, because fraud is rare; yet it is the most expensive setup we ran (Section V-F). We therefore price every decision. For a window $w$,
$ C(w) = sum_("fraud approved") a_i + sum_("good declined") f(a_i) + c_r dot n_r, $ <cost>
where $a_i$ is a payment's amount, $f(a) = max(r a, m)$ is the _friction_ of turning a good customer away, $c_r$ is the review fee and $n_r$ the number of reviews. We set $r =$ #v("costs.friction_rate|pct"), $m =$ #v("costs.friction_min|num") USD and $c_r =$ #v("costs.review_fee|num") USD, and at most #v("costs.review_share|pct") of a window's payments can be reviewed; a review is assumed to find the truth. These prices are assumed, not measured, and every dollar in this paper rests on them (Section VI). Next to each cost we report how many good customers were declined and how many frauds were stopped.

== Paired differences and intervals
To compare setups $A$ and $B$ we take, on every window, the difference $d_w = C_A (w) - C_B (w)$, and its mean $overline(d)$ over the windows; a negative mean means $A$ is cheaper. Pairing removes the large swings between windows that both setups share. The window, not the payment, is the unit: costs are summed per window, payments next to each other in time are not independent, and a single large fraud can dominate a window. How far $overline(d)$ could plausibly move is given by a #v("manual.conventions.ci_level")% _interval_ from the bootstrap: resample the windows with replacement #v("manual.recipe.boot_reps|int") times, take the mean difference each time, and keep the middle #v("manual.conventions.ci_level")% of those means. We call two setups _distinguishable_ only when the interval stays on one side of zero. For example, on the final test the language model alone cost #v("final.diffs.rich_amt_baseline.mean|abs") USD per window more than logistic regression, with an interval of #v("final.diffs.rich_amt_baseline.ci|ci")\; the interval crosses zero, so the two are not distinguishable, however large the mean looks.

= Methods

We follow one payment through every method: a PaySim #v("worked.type") of #v("worked.amount|usd2") USD to a #v("worked.to") at hour #v("worked.hour"), which turned out to be genuine. It was picked by a written rule, the first transfer the model judged whose top answer changed with the order of the options (Implementation choice), not by eye. Its language-model numbers are measured. No rule, logistic regression or tree ever scored it, so the numbers in Sections IV-A to IV-C are _illustrations_ with made-up weights, there only to show the arithmetic.

== The cost rule and the customer line
Every model here gives a fraud score. On IEEE-CIS a logistic fit of the score's logit on dev turns it into $p$, a number between zero and one used only to weigh costs, not claimed as a fraud probability. The rule then prices each action for this one payment and takes the cheapest:
$ "approve:" p a, quad "decline:" (1 - p) f(a), quad "review:" c_r + lambda. $ <rule>
Here $lambda$ is a price for using up one of the few reviews, picked on dev as the value with the lowest dev cost. Once the review budget is spent, the rule takes the cheaper of approve and decline.

On the worked payment, with an illustrative $p approx$ #calc.round(float(v("illustration.rule.p")), digits: 4) (the costs use its unrounded value) and $lambda =$ #v("illustration.rule.lam|num"), approving would be expected to lose #v("illustration.rule.big.approve|usd2") USD, declining #v("illustration.rule.big.decline|usd2"), and a review costs #v("illustration.rule.review|usd2"): the rule asks for a review. The same $p$ on a #v("illustration.rule.small.amount|num") USD payment gives #v("illustration.rule.small.approve|usd2"), #v("illustration.rule.small.decline|usd2") and #v("illustration.rule.review|usd2"): now it declines. Two fixed cut-offs on the score would treat both payments alike; the rule lets the amount decide. Measured on #v("rule.phase14.windows_n") earlier test windows, on the same logistic-regression scores, the rule cost #v("rule.phase14.diff.mean|signed") USD per window against two fixed cut-offs (interval #v("rule.phase14.diff.ci|ci")), distinguishable. Its price: it declined more good customers, #v("manual.phase14.good_declined_rule|int") against #v("manual.phase14.good_declined_lines|int"), because a wrong decline is priced at only a share of the amount.

The _customer line_ is one change to the rule: when it weighs a decline, it adds a flat #v("manual.customer_line.flat|num") USD for turning a good customer away. This price enters the decision only; it is never charged in the cost of Eq. @cost. It makes the rule slower to decline.

== Logistic regression
Logistic regression adds up the payment's facts, each times a weight, into a score $s = w_0 + sum_j w_j x_j$, and squashes the score into a number between zero and one, $p = 1 slash (1 + e^(-s))$. On the worked transfer, with made-up weights: the start is #v("illustration.logreg.start|d2")\; being a transfer (a yes counts as one) adds #v("illustration.logreg.rows.transfer.adds|d2")\; the log of the amount, #v("illustration.logreg.rows.log_amount.x|d2"), times #v("illustration.logreg.rows.log_amount.w|d2") adds #v("illustration.logreg.rows.log_amount.adds|d2")\; it is not at night and not a round amount, so those add nothing. The score is #v("illustration.logreg.score|d2") and $p =$ #v("illustration.logreg.p|d2"). A weight reads through the odds, $p slash (1 - p)$: adding $w$ to the score multiplies the odds of fraud by $e^w$, here $e^(#v("illustration.logreg.rows.transfer.w|d2")) approx$ #v("illustration.logreg.odds_times|d2") for being a transfer, whatever the other facts are. On IEEE-CIS the real model is fitted on the fit part on all the columns. With the rule it is our _baseline_: every other method is compared with it.

== Boosted trees
A tree asks questions about the payment and, depending on the answers, adds a correction to the score. Boosting adds many small trees one after another, each fitted to what the trees before it still get wrong. On the worked transfer, with three made-up trees: the start is #v("illustration.trees.start|d2"), low because fraud is rare; "a transfer?" yes, #v("illustration.trees.t1.yes|d2")\; "amount over #v("illustration.trees.t2.over|int") USD?" yes, #v("illustration.trees.t2.yes|d2")\; "at night?" no, #v("illustration.trees.t3.no|d2"). The score is #v("illustration.trees.score|d2"), squashed as before to $p =$ #v("illustration.trees.p|d2"). A deeper tree can ask "a transfer, and large?", a combination that one weighted sum cannot express. The real model has #v("manual.recipe.trees")\; each correction is shrunk so that no single tree decides. Its score is also given to one of the language-model setups (Section IV-G).

== The language model as a decision model
This is the part that is new. A language model reads a text as a sequence of tokens, word pieces, and at the end it gives a score to every token in its vocabulary as the possible next one. Chat assistants use those scores to write an answer one token at a time. We use them once.

The payment is written as plain facts (@pipeline). For the worked transfer: the receiver is known and was paid #v("worked.facts.receiver_before") times before; the amount was not seen earlier in the hour; it is not round. Then comes one question, "#v("worked.prompt.question")", with three lettered options: A, #v("worked.prompt.options.0")\; B, #v("worked.prompt.options.1")\; C, #v("worked.prompt.options.2"). The model reads all of this in _one forward pass_, the text going through the network once, and we keep only three of its scores: those for the letters A, B and C #cite-n(5). These raw scores are _logits_: higher means more likely, and they are not yet probabilities. For the worked payment, shifted so the top one is zero, they are #v("worked.logits.listed.0|d2"), #v("worked.logits.listed.1|d2") and #v("worked.logits.listed.2|d2") (@worked).

The _softmax_ turns logits $z_k$ into three numbers between zero and one that add up to one:
$ P(k) = e^(z_k) / (sum_j e^(z_j)). $ <softmax>
This is the softmax (multinomial logistic) form: each answer's share of the exponentiated scores. Adding the same constant to every logit leaves Eq. @softmax unchanged, which is why the logits can be shifted so the top is zero. For the worked payment, $e^0 = 1$ and $e^(#v("worked.logits.listed.0|d2")) approx$ #v("worked.exp.listed.0|d2"), so decline gets #v("worked.exp.listed.2|d2") $slash$ #v("worked.exp.sum_listed|d2") $=$ #v("worked.orders.listed.2|d2"), and approve and review #v("worked.orders.listed.0|d2") each. The softmax runs over the three letters only: whatever weight the model puts on any other token is dropped, which is the same as renormalising its next-token probabilities over A, B and C. So the language model becomes a classifier with the interface of any other: a payment in, one probability per class out, from one pass and with no word written. Writing even a one-word answer would take a pass per token and would hide how close the runner-up was; the three scores show it. These are scores in the softmax sense; nothing makes them calibrated fraud probabilities, which is why the rule refits them on dev.

The order of the options matters. Asked again with the options reversed (A decline, B review, C approve), the untrained model put #v("worked.orders.reversed.0|d2") on C, now approve: its top answer flipped from #v("worked.top.listed") to #v("worked.top.reversed"), and both times it picked the last letter (@worked). It follows the position as well as the meaning, as Pezeshkpour and Hruschka report for other models #cite-n(6). On PaySim we therefore ask both orders in the same batch and average the two, which cancels the position: approve #v("worked.scores.0|d2"), review #v("worked.scores.1|d2"), decline #v("worked.scores.2|d2"). Untrained, the model takes the top answer, #v("worked.action"), and a good customer is turned away. On the real IEEE-CIS payments the trained model read one order only, the one dev picked, so nothing there was averaged.

#fig-pipeline
#table-worked

== Two models, and why
On PaySim we used Qwen2.5-1.5B-Instruct #cite-n(10): small enough for a laptop's integrated graphics, openly licensed, and able to answer in one pass. On IEEE-CIS we used d1-Omni-2B #cite-n(14), run in our own PyTorch code at half precision. In earlier checks on PaySim dev rows, frozen and with the same prompt, it ranked fraud better than the frozen Qwen model, and it ran faster than the other candidate we tried. On IEEE-CIS the question and its three options are the same as on PaySim. The prompt opens with a short context on the patterns seen in the fit rows, written before any model scored a row, and then gives the payment's own fields, in this shape (each slot in italics is filled from the row):
#block(inset: (left: 1em, right: 0.5em, y: 2pt), {
  set text(8.5pt)
  set par(first-line-indent: 0pt, justify: false)
  [Payment of _amount_ USD, product _code_, hour _h_:00. Card: _network_ _type_. Purchaser email: _domain_. Recipient email: _domain_. Device: _type_. Match codes: M4 _value_, M6 _value_. Counters: C1, C4, C8, C13 _values_. D1: _value_.]
})
The richer prompt used on the final test adds more counters and time gaps, a distance, and what earlier payments say about the same card and address, never their labels: "Card history: _n_ earlier payments, the last one _t_ ago, ...; this amount is _x_ times its usual." The tree-score setup adds one more line, "Risk score from a tabular model: _s_ (uncalibrated; higher is more suspicious)." Every field is written out as words and numbers; the model sees no table.

== Training with LoRA
Untrained, the model is useless as a decider (Section V-A). We train it, but not all of it. An _adapter_ is a small set of extra weights added to a frozen model. LoRA #cite-n(11) makes it low-rank: a weight matrix $W$ of the model stays frozen, and next to it two thin matrices $B$ and $A$, only $r =$ #v("manual.recipe.lora_r") columns or rows wide, add a correction,
$ W' = W + B A. $
Only $B$ and $A$ learn, a tiny share of the weights, so training fits on one machine. The lesson for each example is one payment's question, and the target is a single letter: the letter of approve if the payment was genuine, of decline if it was fraud. Review is never a target; it comes from the cut-offs or the rule. The loss is a cross-entropy: minus the log of the probability given to the target. On PaySim it runs over the whole vocabulary at the answer position, with the letter's token as the target; on IEEE-CIS it runs over the three letters' scores only, divided by a fixed temperature of #v("manual.training_loss.d1_temperature"). Take the three-letter form, without the temperature. The worked payment was genuine, so its target is the letter of approve; the untrained model gave that letter #calc.round(float(v("worked.orders.listed.0")), digits: 4) as listed, a loss of $-ln$ #calc.round(float(v("worked.orders.listed.0")), digits: 4) $approx$ #calc.round(-calc.ln(float(v("worked.orders.listed.0"))), digits: 2). Each training step nudges $B$ and $A$ to lower this loss, which raises the target letter's logit against the other two. On IEEE-CIS the adapter saw #v("manual.training.fit_rows|int") fit rows, every fit fraud (#v("manual.training.fit_frauds|int")) and #v("manual.training.fit_legit|int") genuine payments, so that fraud makes up #int(calc.round(100 * float(v("manual.training.fit_frauds")) / float(v("manual.training.fit_rows"))))% of the examples, far above its share in the data; one more reason the scores are refitted on dev before the rule uses them. Training ran in #v("manual.recipe.batches|int") batches of #v("manual.recipe.batch"), learning rate #v("manual.recipe.lr"), with LoRA scale $alpha =$ #v("manual.recipe.lora_alpha"). Each setup has one adapter, from one training run. On PaySim, #v("manual.paysim_ft.examples|int") examples took #v("manual.paysim_ft.minutes") minutes on a laptop.

== From probabilities to an action
On PaySim, after training, the averaged decline score goes through two cut-offs picked on dev: below the lower one approve, between them review, above the upper one decline. On IEEE-CIS the decline score goes through the cost rule of Section IV-A, with $lambda$ picked on dev.

One setup gives the language model a tree score as one more fact: a line in its prompt with the boosted trees' risk score for the payment, marked as uncalibrated. The model is free to use it or not. The risk is that it simply copies it, and we measure how closely its ranking follows the score.

The final test compares four registered setups, each with the rule: logistic regression (the baseline); the language model reading the tree score; the same with the customer line; and the language model alone, with no tree score. Approving every payment is reported as a floor. The language model alone with the customer line was asked for after the results were seen, so it is a read-out.

= Results

== PaySim: from useless to matching the rules
Untrained, the Qwen model declined all #v("paysim.model_declined") of the #v("paysim.model_n") payments it judged in the worked window, and on test windows it lost #v("manual.paysim.mean_untrained_millions") million USD per window. After #v("manual.paysim_ft.minutes") minutes of training, its mean cost per window of #v("manual.paysim_ft.window_rows") payments was #v("manual.paysim.mean_trained|int") USD, against #v("manual.paysim.mean_logreg|int") for logistic regression and #v("manual.paysim.mean_rules|int") for hand-written rules, at PaySim's own prices (a review #v("manual.paysim_ft.review_fee") USD, a wrong decline at least #v("manual.paysim_ft.friction_min") USD). It was cheaper than logistic regression in #v("manual.paysim_ft.cheaper_than_logreg") of #v("manual.paysim_ft.windows") new windows, and it wanted the same action as the rules on #v("manual.paysim.matched|int") of #v("manual.paysim.decisions|int") payments. It learned the rule book; it never beat it. We infer that simulated PaySim fraud is close to separable by a few hand-written rules, which leaves a model little room to add value there.

== IEEE-CIS on dev
On #v("dev.windows_n") dev windows the mean cost per window fell step by step: #v("dev.setups.frozen.mean|int") USD for the frozen model, #v("dev.setups.phase12.mean|int") after training, #v("dev.setups.rich.mean|int") with the richer prompt, against #v("dev.setups.logreg.mean|int") for logistic regression and #v("dev.setups.trees.mean|int") for boosted trees, all with the rule. Training mattered most: the frozen d1-Omni-2B cost #v("dev.diffs.frozen_phase12.mean|abs") USD per window more than the trained one (interval #v("dev.diffs.frozen_phase12.ci|ci")). What the model was shown came next: the richer prompt cost #v("dev.diffs.rich_phase12.mean|signed") against the plain one (#v("dev.diffs.rich_phase12.ci|ci")), distinguishable, and it then tied logistic regression, #v("dev.diffs.rich_logreg.mean|signed") (#v("dev.diffs.rich_logreg.ci|ci")). Boosted trees stayed cheaper than the language model alone, which cost #v("dev.diffs.rich_trees.mean|abs") more (#v("dev.diffs.rich_trees.ci|ci")). The setup that read the tree score cost #v("dev.diffs.signal_signal_alone.mean|signed") against the tree score used alone with the same rule (#v("dev.diffs.signal_signal_alone.ci|ci")), not distinguishable, and its ranking followed the score with a rank correlation of #v("dev.echo.rank_corr|d2"). Training on cost, weighting each row by what a mistake on it costs, did not help: #v("cost.diffs.weighted_rich.mean|signed") (#v("cost.diffs.weighted_rich.ci|ci")). Every dev read except logistic regression's is flattered, because $lambda$ was picked with these rows in.

== The final test
@finaltable and @final give the registered result. The primary comparison, the language model reading the tree score against logistic regression, came out at #v("final.diffs.signal_amt_baseline.mean|signed") USD per window, interval #v("final.diffs.signal_amt_baseline.ci|ci"): #v("final.diffs.signal_amt_baseline|dist"). With the customer line it was #v("final.diffs.signal_cust_baseline.mean|signed") (#v("final.diffs.signal_cust_baseline.ci|ci")). The second registered question, the customer line against the same setup without it, gave #v("final.diffs.signal_cust_signal_amt.mean|signed") (#v("final.diffs.signal_cust_signal_amt.ci|ci")): #v("final.diffs.signal_cust_signal_amt|dist"). The language model alone, the fairest measure of its own judgement, cost #v("final.diffs.rich_amt_baseline.mean|signed") (#v("final.diffs.rich_amt_baseline.ci|ci")), and with the customer line, read after the test, #v("final.diffs.rich_cust_baseline.mean|signed") (#v("final.diffs.rich_cust_baseline.ci|ci")). Window by window, the tree-reading setup was cheaper than logistic regression in #v("final.diffs.signal_amt_baseline.first_cheaper") of the #v("final.windows_n") windows and the language model alone in #v("final.diffs.rich_amt_baseline.first_cheaper")\; a few dear windows outweigh many cheap ones, which is why the mean and its interval, not the count, decide. Said plainly, no setup showed a distinguishable saving against logistic regression on the final test: every interval crosses zero.

Some comparisons are distinguishable. Each comparison made with approving everything was distinguishable; the tree-reading setup, for example, was cheaper by #v("final.diffs.signal_amt_approve.mean|abs") USD per window (#v("final.diffs.signal_amt_approve.ci|ci")). And the language model alone cost #v("final.diffs.rich_amt_signal_amt.mean|abs") more than the same model reading the tree score (#v("final.diffs.rich_amt_signal_amt.ci|ci")): the tree score is what helps. Ranking tells the same story: the area under the ROC curve (the chance that a random fraud scores above a random genuine payment) was #v("final.setups.signal_amt.auc|auc") for the tree-reading setup, #v("final.setups.baseline.auc|auc") for logistic regression and #v("final.setups.rich_amt.auc|auc") for the language model alone. The language model alone ranks about as well as logistic regression yet costs more on the mean; the reason was not studied.

#fig-final
#table-final

== The customer line
The clearest gain is to customers (@finaltable). Logistic regression declined #v("final.setups.baseline.good_declined|int") good customers over the final test; the tree-reading setup with the customer line #v("final.setups.signal_cust.good_declined|int")\; the language model alone with the customer line, a read-out, #v("final.setups.rich_cust.good_declined|int"). The price is fewer frauds stopped: #v("final.setups.signal_cust.frauds_stopped|int") and #v("final.setups.rich_cust.frauds_stopped|int") against #v("final.setups.baseline.frauds_stopped|int") of #v("final.frauds|int"). Most of this gain is the customer line, not the language model: on logistic regression, in an earlier test on #v("rule.phase18.windows_n") windows, it cut good customers declined from #v("rule.phase18.base|int") to #v("rule.phase18.customers|int"). If a declined good customer truly cost #v("manual.customer_line.flat|num") USD more, an assumption and a read-out, the language model alone with the customer line would tie the baseline: #v("final.told.diff|signed") USD per window (#v("final.told.ci|ci")).

Where the money goes tells the same story. Each window's cost is the sum of the three parts of Eq. @cost, so the totals over the final test show what each setup traded. Logistic regression let #v("final.setups.baseline.fraud_let_through|int") USD of fraud through, paid #v("final.setups.baseline.good_declined_usd|int") USD of friction on declined good customers and #v("final.setups.baseline.review_fees|int") USD of review fees. The tree-reading setup with the customer line let more fraud through, #v("final.setups.signal_cust.fraud_let_through|int") USD, but paid only #v("final.setups.signal_cust.good_declined_usd|int") USD of friction, with #v("final.setups.signal_cust.review_fees|int") USD of reviews. The language model alone let #v("final.setups.rich_amt.fraud_let_through|int") USD through, with #v("final.setups.rich_amt.good_declined_usd|int") USD of friction and #v("final.setups.rich_amt.review_fees|int") USD of reviews: it lost on every part at once. Fraud let through is by far the largest part for every setup. Single large frauds weigh heavily in it: in one window the tree-reading setup with the customer line cost #v("final.dearest.cost|int") USD against the baseline's #v("final.dearest.baseline|int") (Section VI-I).

== Speed
The language model alone answered within #v("final.latency.p90|int") ms for #v("manual.conventions.latency_quantile")% of payments (half of them within #v("final.latency.p50|int") ms), one payment at a time on one desktop graphics card (an RTX 3060 Ti), against a bar of #v("final.latency.bar|int") ms set before the test. That is not the whole stack: the trees and the history features the prompt reads were not timed.

== Two flips
The first flip is accuracy. Approving every payment is #v("final.approve_all_accuracy|pct") accurate on the final test and costs #v("final.setups.approve_all.mean|int") USD per window, against #v("final.setups.baseline.mean|int") for logistic regression with the rule. Accuracy counts a tiny fraud and a huge one as one mistake each; the cost does not.

The second flip is dev against new windows. On #v("manual.tabular_dev.chains") dev windows, boosted trees looked clearly cheaper than logistic regression, both with the rule: #v("rule.phase20.dev.mean|signed") USD per window (#v("rule.phase20.dev.ci|ci")), #v("rule.phase20.dev|dist"). On #v("rule.phase20.windows_n") new windows the same model and rule gave #v("rule.phase20.diff.mean|signed") (#v("rule.phase20.diff.ci|ci")), #v("rule.phase20.diff|dist"). The language model alone did the same: a tie with logistic regression on dev, #v("dev.diffs.rich_logreg.mean|signed"), became #v("final.diffs.rich_amt_baseline.mean|signed") on the final test. We infer two reasons: dev rows sit close in time to the rows the models learned from, and $lambda$ was picked on dev scores, so dev flatters a model that fits fine detail. Only the locked final test can say what holds.

= Biases and Limitations

== Assumed costs
Every dollar here rests on prices we chose: a wrong decline at #v("costs.friction_rate|pct") of the amount, at least #v("costs.friction_min|num") USD, and a review at #v("costs.review_fee|num") USD. Change one and verdicts move. When we added a flat price for each good customer declined, on earlier test windows, the rule's saving over two fixed cut-offs stopped being distinguishable from #v("manual.stress.not_distinguishable_from") USD per customer, and its mean saving became a loss at #v("manual.stress.flat_low")–#v("manual.stress.flat_high") USD. The rule declines more customers, so any price charged per customer counts against it.

== Option-order bias
The untrained PaySim model leaned towards a letter's position: in the worked window, its top answer flipped with the order on #v("paysim.order_flips") of the #v("paysim.model_n") payments it judged, and in #v("paysim.last_letter_both") of those it took the last letter both times. Averaging both orders cancels the position #cite-n(5, 6). On IEEE-CIS the trained model read one order, chosen on dev, and its lean there was not measured.

== Optimism from choosing on dev
The cut-offs, $lambda$, the prompt and which setups went forward were all chosen on dev, so dev results flatter. The trees' dev saving shrank to a not-distinguishable one on new windows, and the language model's dev tie became a higher mean cost on the final test (Section V-F). The final test was reserved before this work began and scored once, which limits this bias to what dev chose; it does not remove it. A small difference in procedure remains: $lambda$ for the language-model setups came from more dev windows than logistic regression's.

== The language model copying the tree score
The cheapest-looking language setup ranked payments almost exactly as the tree score it reads: rank correlation #v("final.rank_corr|d2") on the final-test rows. On dev it did no better than the score alone. The score alone was not a registered final-test setup, so the test cannot show what the language model adds on top of it. We infer that this setup is mostly a tree score with a language model around it.

== Fairness promised, not measured
The proposal listed fairness among the risks to control. Fairness was not measured: who gets declined was never checked by group, for example by card type, device or region, and nothing here can say whether the customer line's gain is shared evenly.

== Columns that may not exist at decision time
Logistic regression and the tree score both use the anonymised C, D and V columns, and we could not establish that the C counters are known when a payment is decided. On #v("rule.phase17.windows_n") earlier test windows, logistic regression without those columns cost #v("rule.phase17.diff.mean|abs") USD per window more (#v("rule.phase17.diff.ci|ci")). If they are not available in time, the results here do not hold.

== PaySim is simulated
The first cycle's success, matching the hand-written rules, is on simulated data that nearly separate by rule. It shows that the pipeline and the training work, not that a language model is worth using on real payments.

== One adapter per setup
Each setup was trained once. On dev a second training seed moved the cost by #v("dev.diffs.seed1_phase12.mean|signed") USD per window (#v("dev.diffs.seed1_phase12.ci|ci")), about the size of several differences reported here.

== No clean test rows left
The final test is spent: it was scored once and cannot be re-scored without becoming a second look. The earlier test windows have been used too. With only #v("final.windows_n") windows the intervals are wide, and single large frauds weigh heavily: in one window the tree-reading setup with the customer line cost #v("final.dearest.cost|int") USD against the baseline's #v("final.dearest.baseline|int"). Answering what the language model adds, or whether any setup saves money, needs new data.

= Conclusion

A language model can be read as a classifier: one lettered question, one forward pass, one score per letter, and no word written. A small LoRA adapter took it from useless to matching hand-written rules on simulated payments. On the real payments, the trained d1-Omni-2B alone answered within #v("final.latency.p90|int") ms for #v("manual.conventions.latency_quantile")% of payments on one desktop graphics card, well inside its time bar. Scored once on a locked final test, no setup showed a distinguishable saving against logistic regression with the same cost rule; the cheapest-looking language setup mostly follows the tree score it reads, and the language model alone cost more on the mean. The gain that held, far fewer good customers declined, comes mainly from the customer line in the decision rule, which works on logistic regression too. The prices are assumed, the untrained model leaned towards the last letter, and fairness was not measured. For a team that already runs a tabular model, we infer that the cheaper lesson here is the decision rule: pricing each action for each payment, and pricing a turned-away customer on its own, changed outcomes more than any model did. The next steps are new data, a registered test of the tree score alone, measured costs, and a check of outcomes by group.

#heading(numbering: none)[References]

#[
#set text(8pt)
#set par(first-line-indent: 0pt, justify: true, spacing: 0.45em)
// references: start
#bib(1)[The Nilson Report, “Card fraud losses worldwide in 2023,” no. 1276, Dec. 2024.]
#bib(2)[A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, G. Bontempi, “Credit card fraud detection: a realistic modeling and a novel learning strategy,” _IEEE TNNLS_, 29(8), 3784–3797, 2018.]
#bib(3)[L. Grinsztajn, E. Oyallon, G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?,” _NeurIPS Datasets and Benchmarks_, 2022.]
#bib(4)[S. Hegselmann et al., “TabLLM: Few-shot classification of tabular data with large language models,” _AISTATS_, PMLR 206, 5549–5581, 2023.]
#bib(5)[J. Robinson, D. Wingate, “Leveraging large language models for multiple choice question answering,” _ICLR_, 2023.]
#bib(6)[P. Pezeshkpour, E. Hruschka, “Large language models sensitivity to the order of options in multiple-choice questions,” _Findings of NAACL_, 2006–2017, 2024.]
#bib(7)[X. Tan, Y. Ma, X. Zhang, “Understanding structured financial data with LLMs: a case study on fraud detection,” arXiv:2512.13040, 2025.]
#bib(8)[E. A. Lopez-Rojas, A. Elmir, S. Axelsson, “PaySim: a financial mobile money simulator for fraud detection,” _EMSS_, 2016.]
#bib(9)[IEEE-CIS and Vesta Corporation, “IEEE-CIS Fraud Detection,” Kaggle, 2019.]
#bib(10)[Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.]
#bib(11)[E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” _ICLR_, 2022.]
#bib(12)[R. Wirth, J. Hipp, “CRISP-DM: Towards a standard process model for data mining,” 2000.]
#bib(13)[TypeSafe, “Introducing System One models & Jev,” 2026; S. Goedecke, “Two techniques for working with System One models,” 2026.]
#bib(14)[LiquidAI, “d1-Omni-2B,” Hugging Face model card, LiquidAI/d1-Omni-2B, 2026.]
// references: end
]
