// The paper: Can a small fine-tuned LLM make fraud decisions? IEEE two-column, built with Typst.
// Every number comes from data.json through v() with a "path|format" key; run `python paper/build_data.py` after adding a key.
// The blocks between "generated: start" and "generated: end" are written by `python paper/gen_tables.py`; edit that script, not the block.
#import "vendor/charged-ieee/lib.typ": ieee

#let D = json("data.json")
#let v(k) = D.at(k)

// Citations as IEEE numbers, by key, numbered in order of first citation (REFS below); the list is at the end (references: start/end).
#let REFS = ("nilson", "robinson", "pet", "dalpozzolo", "tabllm", "tan", "lift", "fang", "tabpfn", "lmbff", "pezeshkpour", "zheng", "elkan", "bahnsen", "whitrow", "grinsztajn", "friedman", "ieeecis", "paysim", "crisp", "nosek", "dror", "kunsch", "efron", "platt", "d1omni", "guo", "lora", "qwen", "kaufman")
#let cite-k(..ks) = ks.pos().map(k => "[" + str(REFS.position(x => x == k) + 1) + "]").join(", ")
#let bibk(k, body) = grid(columns: (2.2em, 1fr), column-gutter: 0.2em, [\[#(REFS.position(x => x == k) + 1)\]], body)

// code-and-data: start
#let code-and-data = footnote[Code and results: #link("https://github.com/Jumbo-Juice/local-system-one")[github.com/Jumbo-Juice/local-system-one]\; the data download steps are in its #raw("REPRODUCE.md"). Every number in this paper is read from #raw("paper/data.json"), which #raw("paper/build_data.py") builds from the committed results: the final test #raw(v("_sources.final")), the trees' read-out on it #raw(v("rev._sources.readout")), the analyses of saved final-test outputs #raw(v("rev._sources.readouts")), the dev experiments #raw(v("rev._sources.parity")), #raw(v("rev._sources.noc")), #raw(v("rev._sources.order")), #raw(v("rev._sources.custline")) and #raw(v("_sources.nuc")), the window audit #raw(v("rev._sources.audit")), the earlier tests #raw(v("_sources.amount")), #raw(v("_sources.rule")), #raw(v("_sources.audit")) and #raw(v("_sources.tabular")), the pinned PaySim trace #raw(v("_sources.paysim_trace")), and prose-only numbers with their sources in #raw(v("_sources.manual")). Registrations: #raw("fraud/prereg/"), the eval scripts' docstrings, and #raw("REVISION_LOG.md").]
// code-and-data: end

#let times = ("TeX Gyre Termes", "Times New Roman")
#show text: set text(font: times)
#show place: set text(font: times)
#show footnote.entry: set text(font: times)

#show: ieee.with(
  title: [Can a Small Fine-Tuned LLM Make Fraud Decisions? A Locked, Cost-Based Evaluation],
  authors: (
    (name: "Juji Vongvilay", organization: [Siam University]),
    (name: "Thu Ya Aung", organization: [Siam University]),
    (name: "Chanpichsoponha Samnang", organization: [Siam University]),
    (name: "Sai Thit Swan Thu", organization: [Siam University]),
  ),
  abstract: text(font: times)[
    A fraud team must approve, review or decline every payment, and each wrong call has a price. We ask whether a small language model, fine-tuned with low-rank adapters and read as a single-pass three-way classifier, lowers that price against logistic regression and boosted trees on real IEEE-CIS payments, and whether any difference comes from the model, the information it sees, or the decision rule. Every choice was made on development data; a final test was registered in advance and scored once. The decision rule mattered more than any model: on earlier test windows, pricing each action per payment, a Bayes minimum-risk rule with a review budget, cost #v("rule.phase14.diff.mean|abs") USD per #v("manual.data.window_rows|int")-payment window less than two fixed cut-offs, while no model was distinguishably cheaper than logistic regression. The fine-tuned model showed no detectable saving over logistic regression; effects below about #v("rev.cmp.p23_primary.mde|int") USD per window cannot be ruled out (read-out). With no unscored test data left, boosted trees were read out on the same windows: cheapest, but not distinguishably so. On development data, at equal information the language model was cheaper than logistic regression and not distinguishable from boosted trees, so its deficit is mostly information. All prices are assumed.
  ],
  index-terms: (text(font: times)[fraud detection, language models, verbalizers, low-rank adaptation, cost-sensitive decisions, pre-registration],),
  bibliography: none,
)
#set text(font: times)


// ---- figures, drawn here from data.json (no package) -------------------------------------------------------

#let minus(x) = if x < 0 { "\u{2212}" + str(calc.abs(x)) } else { str(x) }

// Dots with interval lines and a zero line: one row per comparison.
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
      place(dx: label-w + x(r.mean) - 2.4pt, dy: y - 2.4pt, circle(radius: 2.4pt, fill: if r.at("open", default: false) { white } else { black }, stroke: 0.8pt + black))
      place(dx: label-w + plot-w + 4pt, dy: y - 4pt, text(7pt, r.said))
    }
  })
})

// A scatter of (x, y) series joined by lines; each point may carry a short note.
#let scatter(series, xlab, ylab, height: 150pt) = layout(size => {
  let w = size.width
  let left = 30pt
  let bottom = 22pt
  let pw = w - left - 6pt
  let ph = height - bottom
  let xs = series.map(s => s.pts.map(p => p.x)).flatten()
  let ys = series.map(s => s.pts.map(p => p.y)).flatten()
  let (x0, x1) = (0.0, calc.max(..xs) * 1.05)
  let (y0, y1) = (calc.min(..ys) * 0.97, calc.max(..ys) * 1.02)
  let X(x) = left + (x - x0) / (x1 - x0) * pw
  let Y(y) = ph - (y - y0) / (y1 - y0) * ph
  let xstep = if x1 > 3000 { 1000 } else { 500 }
  let ystep = if y1 - y0 > 600 { 200 } else { 100 }
  block(width: w, height: height, {
    place(dx: left, dy: ph, line(start: (0pt, 0pt), end: (pw, 0pt), stroke: 0.4pt + luma(120)))
    place(dx: left, dy: 0pt, line(start: (0pt, 0pt), end: (0pt, ph), stroke: 0.4pt + luma(120)))
    for k in range(0, int(x1 / xstep) + 1) {
      let t = k * xstep
      place(dx: X(t) - 12pt, dy: ph + 2pt, box(width: 24pt, align(center, text(6pt, str(t)))))
    }
    for k in range(int(calc.ceil(y0 / ystep)), int(y1 / ystep) + 1) {
      let t = k * ystep
      place(dx: 0pt, dy: Y(t) - 3.5pt, box(width: left - 3pt, align(right, text(6pt, str(t)))))
    }
    place(dx: left, dy: ph + 11pt, box(width: pw, align(center, text(6.5pt, xlab))))
    place(dx: left + 4pt, dy: 1pt, text(6.5pt, ylab))
    for s in series {
      for (a, b) in s.pts.zip(s.pts.slice(1)) {
        place(line(start: (X(a.x), Y(a.y)), end: (X(b.x), Y(b.y)), stroke: (paint: s.color, thickness: 0.9pt, dash: s.dash)))
      }
      for p in s.pts {
        place(dx: X(p.x) - 2pt, dy: Y(p.y) - 2pt, if s.square { square(size: 4pt, fill: s.color) } else { circle(radius: 2pt, fill: s.color) })
        if p.note != none { place(dx: if p.x > 0.7 * x1 { X(p.x) - 52pt } else { X(p.x) + 3pt }, dy: Y(p.y) + s.shift, text(5.5pt, fill: s.color, p.note)) }
      }
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
    stage[One forward pass][The model reads the text once and scores every possible next token #cite-k("robinson")],
    arrow,
    stage[Verbalizer][Keep the scores of the letters A, B, C, shifted so the top is zero: #v("worked.logits.listed.0|d2"), #v("worked.logits.listed.1|d2"), #v("worked.logits.listed.2|d2")],
    arrow,
    stage[Softmax][Three scores that add up to one: #v("worked.orders.listed.0|d2"), #v("worked.orders.listed.1|d2"), #v("worked.orders.listed.2|d2")],
    arrow,
    stage[Action][Untrained: the top letter. Trained: Platt scaling, then the cost rule],
  ),
  caption: [The language model as a single-pass classifier, on one simulated PaySim payment (untrained Qwen2.5-1.5B, options as listed). It writes nothing: one pass gives one score per letter.],
) <pipeline>]

#let fig-final = [#figure(
  placement: top,
  interval-chart(
    (
      (label: [LM + tree score], mean: float(v("final.diffs.signal_amt_baseline.mean")), lo: float(v("final.diffs.signal_amt_baseline.ci.0")), hi: float(v("final.diffs.signal_amt_baseline.ci.1")), said: v("final.diffs.signal_amt_baseline.mean|signed")),
      (label: [LM + tree score, customer line], mean: float(v("final.diffs.signal_cust_baseline.mean")), lo: float(v("final.diffs.signal_cust_baseline.ci.0")), hi: float(v("final.diffs.signal_cust_baseline.ci.1")), said: v("final.diffs.signal_cust_baseline.mean|signed")),
      (label: [LM alone], mean: float(v("final.diffs.rich_amt_baseline.mean")), lo: float(v("final.diffs.rich_amt_baseline.ci.0")), hi: float(v("final.diffs.rich_amt_baseline.ci.1")), said: v("final.diffs.rich_amt_baseline.mean|signed")),
      (label: [GBDT alone (read-out)], mean: float(v("rev.cmp.ro_secondary.mean")), lo: float(v("rev.cmp.ro_secondary.ci.0")), hi: float(v("rev.cmp.ro_secondary.ci.1")), said: v("rev.cmp.ro_secondary.mean|signed"), open: true),
    ),
    [USD per window, setup minus logistic regression],
  ),
  caption: [The final test: each setup's cost per window minus logistic regression's on the same #v("final.windows_n") windows. Dot: mean paired difference; line: its #v("manual.conventions.ci_level")% interval; left of zero is cheaper. Filled dots are registered results; the open dot is the trees' read-out on the same, already-used windows. Every line crosses zero.],
) <final>]

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
    [logistic regression (LR, the baseline)], [#v("final.setups.baseline.mean|int")], [#v("final.setups.baseline.ci|range")], [], [], [], [#v("final.setups.baseline.good_declined|int")], [#v("final.setups.baseline.frauds_stopped|int")],
    [LM + tree score], [#v("final.setups.signal_amt.mean|int")], [#v("final.setups.signal_amt.ci|range")], [#v("final.diffs.signal_amt_baseline.mean|signed")], [#v("final.diffs.signal_amt_baseline.ci|ci")], [#v("final.diffs.signal_amt_baseline|dist")], [#v("final.setups.signal_amt.good_declined|int")], [#v("final.setups.signal_amt.frauds_stopped|int")],
    [LM + tree score, customer line], [#v("final.setups.signal_cust.mean|int")], [#v("final.setups.signal_cust.ci|range")], [#v("final.diffs.signal_cust_baseline.mean|signed")], [#v("final.diffs.signal_cust_baseline.ci|ci")], [#v("final.diffs.signal_cust_baseline|dist")], [#v("final.setups.signal_cust.good_declined|int")], [#v("final.setups.signal_cust.frauds_stopped|int")],
    [LM alone], [#v("final.setups.rich_amt.mean|int")], [#v("final.setups.rich_amt.ci|range")], [#v("final.diffs.rich_amt_baseline.mean|signed")], [#v("final.diffs.rich_amt_baseline.ci|ci")], [#v("final.diffs.rich_amt_baseline|dist")], [#v("final.setups.rich_amt.good_declined|int")], [#v("final.setups.rich_amt.frauds_stopped|int")],
    [LM alone, customer line\*], [#v("final.setups.rich_cust.mean|int")], [#v("final.setups.rich_cust.ci|range")], [#v("final.diffs.rich_cust_baseline.mean|signed")], [#v("final.diffs.rich_cust_baseline.ci|ci")], [#v("final.diffs.rich_cust_baseline|dist")], [#v("final.setups.rich_cust.good_declined|int")], [#v("final.setups.rich_cust.frauds_stopped|int")],
    [GBDT alone\*], [#v("rev.gbdt.mean|int")], [#v("rev.gbdt.ci|range")], [#v("rev.cmp.ro_secondary.mean|signed")], [#v("rev.cmp.ro_secondary.ci|ci")], [#v("rev.cmp.ro_secondary|dist")], [#v("rev.gbdt.good_declined|int")], [#v("rev.gbdt.frauds_stopped|int")],
    [approve every payment (no model)], [#v("final.setups.approve_all.mean|int")], [#v("final.setups.approve_all.ci|range")], [], [], [], [#v("final.setups.approve_all.good_declined|int")], [#v("final.setups.approve_all.frauds_stopped|int")],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [The final test (held-out, pre-registered): #v("final.windows_n") windows, #v("final.rows|int") real payments, #v("final.frauds|int") frauds, each setup scored once. Costs in USD per window. LM is the fine-tuned d1-Omni-2B. \*Read-out: computed after the registered results were seen, on the same windows; not a test result.],
) <finaltable>]

// generated: start
#let table-sources = [#figure(
  placement: top,
  {
  set text(7pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (1fr, auto, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([dev comparison, USD per window], [mean], [95% interval], [verdict]),
    table.hline(stroke: 0.4pt),
    [LM alone − LR, every column], [#v("dev.diffs.rich_logreg.mean|signed")], [#v("dev.diffs.rich_logreg.ci|ci")], [#v("dev.diffs.rich_logreg|dist")],
    [LM alone − LR, at the prompt's fields], [#v("rev.parity.cmp.lm-rich__par-logreg.mean|signed")], [#v("rev.parity.cmp.lm-rich__par-logreg.ci|ci")], [#v("rev.parity.cmp.lm-rich__par-logreg|dist")],
    [LM alone − GBDT, at the prompt's fields], [#v("rev.parity.cmp.lm-rich__par-trees.mean|signed")], [#v("rev.parity.cmp.lm-rich__par-trees.ci|ci")], [#v("rev.parity.cmp.lm-rich__par-trees|dist")],
    [GBDT: prompt's fields − every column], [#v("rev.parity.cmp.par-trees__tab-all.mean|signed")], [#v("rev.parity.cmp.par-trees__tab-all.ci|ci")], [#v("rev.parity.cmp.par-trees__tab-all|dist")],
    [LM alone − LR, both without C], [#v("rev.noc.cmp.lm-rich-noc__noc-logreg.mean|signed")], [#v("rev.noc.cmp.lm-rich-noc__noc-logreg.ci|ci")], [#v("rev.noc.cmp.lm-rich-noc__noc-logreg|dist")],
    [LM alone: forward − scored option order], [#v("rev.order.lm-rich.fwd_rev.mean|signed")], [#v("rev.order.lm-rich.fwd_rev.ci|ci")], [#v("rev.order.lm-rich.fwd_rev|dist")],
    [LM, second − first training seed], [#v("rev.seed.mean|signed")], [#v("rev.seed.ci|ci")], [not distinguishable],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Where the difference comes from (dev, #v("manual.tabular_dev.chains") windows; every read involving a language-model setup is flattered by $lambda$ picked on these windows). The first row is the registered setups' dev comparison; the rest are the revision's dev experiments (Appendices E–G).],
) <table-sources>]

#let table-robust = [#figure(
  placement: none,
  {
  set text(6.5pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([a − b], [label], [mean], [percentile (registered)], [block 2], [block 4], [BCa], [paired SD], [MDE]),
    table.hline(stroke: 0.4pt),
    [LM + tree − LR], [#v("rev.cmp.p23_primary.label")], [#v("rev.cmp.p23_primary.mean|signed")], [#v("rev.cmp.p23_primary.ci|ci")], [#v("rev.cmp.p23_primary.block2|ci")], [#v("rev.cmp.p23_primary.block4|ci")], [#v("rev.cmp.p23_primary.bca|ci")], [#v("rev.cmp.p23_primary.sd|int")], [#v("rev.cmp.p23_primary.mde|int")],
    [LM alone − LR], [#v("rev.cmp.p23_third.label")], [#v("rev.cmp.p23_third.mean|signed")], [#v("rev.cmp.p23_third.ci|ci")], [#v("rev.cmp.p23_third.block2|ci")], [#v("rev.cmp.p23_third.block4|ci")], [#v("rev.cmp.p23_third.bca|ci")], [#v("rev.cmp.p23_third.sd|int")], [#v("rev.cmp.p23_third.mde|int")],
    [cust. line − LM + tree], [#v("rev.cmp.p23_second.label")], [#v("rev.cmp.p23_second.mean|signed")], [#v("rev.cmp.p23_second.ci|ci")], [#v("rev.cmp.p23_second.block2|ci")], [#v("rev.cmp.p23_second.block4|ci")], [#v("rev.cmp.p23_second.bca|ci")], [#v("rev.cmp.p23_second.sd|int")], [#v("rev.cmp.p23_second.mde|int")],
    [LM + tree, cust. − LR], [#v("rev.cmp.p23_cust_vs_lr.label")], [#v("rev.cmp.p23_cust_vs_lr.mean|signed")], [#v("rev.cmp.p23_cust_vs_lr.ci|ci")], [#v("rev.cmp.p23_cust_vs_lr.block2|ci")], [#v("rev.cmp.p23_cust_vs_lr.block4|ci")], [#v("rev.cmp.p23_cust_vs_lr.bca|ci")], [#v("rev.cmp.p23_cust_vs_lr.sd|int")], [#v("rev.cmp.p23_cust_vs_lr.mde|int")],
    [LM alone − LM + tree], [#v("rev.cmp.p23_rich_vs_signal.label")], [#v("rev.cmp.p23_rich_vs_signal.mean|signed")], [#v("rev.cmp.p23_rich_vs_signal.ci|ci")], [#v("rev.cmp.p23_rich_vs_signal.block2|ci")], [#v("rev.cmp.p23_rich_vs_signal.block4|ci")], [#v("rev.cmp.p23_rich_vs_signal.bca|ci")], [#v("rev.cmp.p23_rich_vs_signal.sd|int")], [#v("rev.cmp.p23_rich_vs_signal.mde|int")],
    [LM + tree − approve all], [#v("rev.cmp.p23_signal_vs_all.label")], [#v("rev.cmp.p23_signal_vs_all.mean|signed")], [#v("rev.cmp.p23_signal_vs_all.ci|ci")], [#v("rev.cmp.p23_signal_vs_all.block2|ci")], [#v("rev.cmp.p23_signal_vs_all.block4|ci")], [#v("rev.cmp.p23_signal_vs_all.bca|ci")], [#v("rev.cmp.p23_signal_vs_all.sd|int")], [#v("rev.cmp.p23_signal_vs_all.mde|int")],
    [LM + tree − GBDT\*], [#v("rev.cmp.ro_primary.label")], [#v("rev.cmp.ro_primary.mean|signed")], [#v("rev.cmp.ro_primary.ci|ci")], [#v("rev.cmp.ro_primary.block2|ci")], [#v("rev.cmp.ro_primary.block4|ci")], [#v("rev.cmp.ro_primary.bca|ci")], [#v("rev.cmp.ro_primary.sd|int")], [#v("rev.cmp.ro_primary.mde|int")],
    [GBDT − LR\*], [#v("rev.cmp.ro_secondary.label")], [#v("rev.cmp.ro_secondary.mean|signed")], [#v("rev.cmp.ro_secondary.ci|ci")], [#v("rev.cmp.ro_secondary.block2|ci")], [#v("rev.cmp.ro_secondary.block4|ci")], [#v("rev.cmp.ro_secondary.bca|ci")], [#v("rev.cmp.ro_secondary.sd|int")], [#v("rev.cmp.ro_secondary.mde|int")],
    [LM alone − GBDT\*], [#v("rev.cmp.ro_rich_vs_gbdt.label")], [#v("rev.cmp.ro_rich_vs_gbdt.mean|signed")], [#v("rev.cmp.ro_rich_vs_gbdt.ci|ci")], [#v("rev.cmp.ro_rich_vs_gbdt.block2|ci")], [#v("rev.cmp.ro_rich_vs_gbdt.block4|ci")], [#v("rev.cmp.ro_rich_vs_gbdt.bca|ci")], [#v("rev.cmp.ro_rich_vs_gbdt.sd|int")], [#v("rev.cmp.ro_rich_vs_gbdt.mde|int")],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Every final-test comparison, USD per window. The registered percentile interval is the headline; the block bootstraps (blocks of 2 and 4 windows in time order) and BCa intervals, the paired SD and the MDE are read-outs. The block-4 interval of GBDT − LR excludes zero only by a margin within Monte Carlo error, so it is borderline. \*GBDT is a read-out on the same windows.],
) <table-robust>]

#let table-ranking = [#figure(
  placement: none,
  {
  set text(6.5pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([setup], [ROC-AUC], [95% CI], [PR-AUC], [95% CI], [fraud let through], [good declined], [review fees]),
    table.hline(stroke: 0.4pt),
    [LR], [#v("rev.ranking.lr.roc_auc|auc")], [#v("rev.ranking.lr.roc_auc_ci|aucrange")], [#v("rev.ranking.lr.pr_auc|auc")], [#v("rev.ranking.lr.pr_auc_ci|aucrange")], [#v("rev.by_action.lr.fraud_let_through|int")], [#v("rev.by_action.lr.good_declined|int")], [#v("rev.by_action.lr.review_fees|int")],
    [LM + tree], [#v("rev.ranking.lmtree.roc_auc|auc")], [#v("rev.ranking.lmtree.roc_auc_ci|aucrange")], [#v("rev.ranking.lmtree.pr_auc|auc")], [#v("rev.ranking.lmtree.pr_auc_ci|aucrange")], [#v("rev.by_action.lmtree.fraud_let_through|int")], [#v("rev.by_action.lmtree.good_declined|int")], [#v("rev.by_action.lmtree.review_fees|int")],
    [LM + tree, cust.], [#v("rev.ranking.lmtree_cust.roc_auc|auc")], [#v("rev.ranking.lmtree_cust.roc_auc_ci|aucrange")], [#v("rev.ranking.lmtree_cust.pr_auc|auc")], [#v("rev.ranking.lmtree_cust.pr_auc_ci|aucrange")], [#v("rev.by_action.lmtree_cust.fraud_let_through|int")], [#v("rev.by_action.lmtree_cust.good_declined|int")], [#v("rev.by_action.lmtree_cust.review_fees|int")],
    [LM alone], [#v("rev.ranking.lm.roc_auc|auc")], [#v("rev.ranking.lm.roc_auc_ci|aucrange")], [#v("rev.ranking.lm.pr_auc|auc")], [#v("rev.ranking.lm.pr_auc_ci|aucrange")], [#v("rev.by_action.lm.fraud_let_through|int")], [#v("rev.by_action.lm.good_declined|int")], [#v("rev.by_action.lm.review_fees|int")],
    [GBDT\*], [#v("rev.ranking.gbdt.roc_auc|auc")], [#v("rev.ranking.gbdt.roc_auc_ci|aucrange")], [#v("rev.ranking.gbdt.pr_auc|auc")], [#v("rev.ranking.gbdt.pr_auc_ci|aucrange")], [#v("rev.by_action.gbdt.fraud_let_through|int")], [#v("rev.by_action.gbdt.good_declined|int")], [#v("rev.by_action.gbdt.review_fees|int")],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Ranking and where the money goes on the final test (read-out). Intervals: window bootstrap, #v("manual.recipe.boot_reps|int") resamples. Costs: USD per window by part of Eq. 1. \*Read-out setup.],
) <table-ranking>]

#let ranking-drop = block(inset: (y: 3pt))[#set text(8pt)
#set par(first-line-indent: 0pt)
LM alone − LR on the final test is #v("rev.drop_amount.0.mean|signed") USD per window (#v("rev.drop_amount.0.ci|ci"), held-out). Removing the largest 1 and 2 frauds by amount (#v("rev.drop_amount.2.amounts.0|usd2") and #v("rev.drop_amount.2.amounts.1|usd2") USD), which both setups reviewed, leaves #v("rev.drop_amount.1.mean|signed") and #v("rev.drop_amount.2.mean|signed") (read-out). Post-hoc read-out, not registered: removing the 1 and 2 frauds with the largest cost gap (#v("rev.drop_gap.2.amounts.0|usd2") and #v("rev.drop_gap.2.amounts.1|usd2") USD, approved by LM alone, reviewed by LR) leaves #v("rev.drop_gap.1.mean|signed") (#v("rev.drop_gap.1.ci|ci")) and #v("rev.drop_gap.2.mean|signed") (#v("rev.drop_gap.2.ci|ci")).]

#let rel-panel(title, pts) = layout(size => {
  let w = size.width
  let s = w - 14pt
  let top = calc.max(0.6, ..pts.map(p => calc.max(p.x, p.y)))
  let X(x) = 12pt + calc.sqrt(x / top) * s
  let Y(y) = s - calc.sqrt(y / top) * s
  block(width: w, height: s + 18pt, {
    place(dx: 12pt, dy: 0pt, rect(width: s, height: s, stroke: 0.4pt + luma(150)))
    place(line(start: (X(0.0), Y(0.0)), end: (X(top), Y(top)), stroke: (paint: luma(170), thickness: 0.5pt, dash: "dashed")))
    for p in pts { place(dx: X(p.x) - 1.6pt, dy: Y(p.y) - 1.6pt, circle(radius: 1.6pt, fill: black)) }
    place(dx: 12pt, dy: s + 2pt, box(width: s, align(center, text(6pt, title))))
  })
})
#let fig-reliability = [#figure(
  placement: none,
  grid(columns: (1fr,) * 5, column-gutter: 10pt, rel-panel([LR], ((x: float(v("rev.reliability.lr.0.mean_p")), y: float(v("rev.reliability.lr.0.fraud_rate"))), (x: float(v("rev.reliability.lr.1.mean_p")), y: float(v("rev.reliability.lr.1.fraud_rate"))), (x: float(v("rev.reliability.lr.2.mean_p")), y: float(v("rev.reliability.lr.2.fraud_rate"))), (x: float(v("rev.reliability.lr.3.mean_p")), y: float(v("rev.reliability.lr.3.fraud_rate"))), (x: float(v("rev.reliability.lr.4.mean_p")), y: float(v("rev.reliability.lr.4.fraud_rate"))), (x: float(v("rev.reliability.lr.5.mean_p")), y: float(v("rev.reliability.lr.5.fraud_rate"))), (x: float(v("rev.reliability.lr.6.mean_p")), y: float(v("rev.reliability.lr.6.fraud_rate"))), (x: float(v("rev.reliability.lr.7.mean_p")), y: float(v("rev.reliability.lr.7.fraud_rate"))), (x: float(v("rev.reliability.lr.8.mean_p")), y: float(v("rev.reliability.lr.8.fraud_rate"))), (x: float(v("rev.reliability.lr.9.mean_p")), y: float(v("rev.reliability.lr.9.fraud_rate"))))), rel-panel([LM + tree], ((x: float(v("rev.reliability.lmtree.0.mean_p")), y: float(v("rev.reliability.lmtree.0.fraud_rate"))), (x: float(v("rev.reliability.lmtree.1.mean_p")), y: float(v("rev.reliability.lmtree.1.fraud_rate"))), (x: float(v("rev.reliability.lmtree.2.mean_p")), y: float(v("rev.reliability.lmtree.2.fraud_rate"))), (x: float(v("rev.reliability.lmtree.3.mean_p")), y: float(v("rev.reliability.lmtree.3.fraud_rate"))), (x: float(v("rev.reliability.lmtree.4.mean_p")), y: float(v("rev.reliability.lmtree.4.fraud_rate"))), (x: float(v("rev.reliability.lmtree.5.mean_p")), y: float(v("rev.reliability.lmtree.5.fraud_rate"))), (x: float(v("rev.reliability.lmtree.6.mean_p")), y: float(v("rev.reliability.lmtree.6.fraud_rate"))), (x: float(v("rev.reliability.lmtree.7.mean_p")), y: float(v("rev.reliability.lmtree.7.fraud_rate"))), (x: float(v("rev.reliability.lmtree.8.mean_p")), y: float(v("rev.reliability.lmtree.8.fraud_rate"))), (x: float(v("rev.reliability.lmtree.9.mean_p")), y: float(v("rev.reliability.lmtree.9.fraud_rate"))))), rel-panel([LM + tree, cust.], ((x: float(v("rev.reliability.lmtree_cust.0.mean_p")), y: float(v("rev.reliability.lmtree_cust.0.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.1.mean_p")), y: float(v("rev.reliability.lmtree_cust.1.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.2.mean_p")), y: float(v("rev.reliability.lmtree_cust.2.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.3.mean_p")), y: float(v("rev.reliability.lmtree_cust.3.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.4.mean_p")), y: float(v("rev.reliability.lmtree_cust.4.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.5.mean_p")), y: float(v("rev.reliability.lmtree_cust.5.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.6.mean_p")), y: float(v("rev.reliability.lmtree_cust.6.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.7.mean_p")), y: float(v("rev.reliability.lmtree_cust.7.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.8.mean_p")), y: float(v("rev.reliability.lmtree_cust.8.fraud_rate"))), (x: float(v("rev.reliability.lmtree_cust.9.mean_p")), y: float(v("rev.reliability.lmtree_cust.9.fraud_rate"))))), rel-panel([LM alone], ((x: float(v("rev.reliability.lm.0.mean_p")), y: float(v("rev.reliability.lm.0.fraud_rate"))), (x: float(v("rev.reliability.lm.1.mean_p")), y: float(v("rev.reliability.lm.1.fraud_rate"))), (x: float(v("rev.reliability.lm.2.mean_p")), y: float(v("rev.reliability.lm.2.fraud_rate"))), (x: float(v("rev.reliability.lm.3.mean_p")), y: float(v("rev.reliability.lm.3.fraud_rate"))), (x: float(v("rev.reliability.lm.4.mean_p")), y: float(v("rev.reliability.lm.4.fraud_rate"))), (x: float(v("rev.reliability.lm.5.mean_p")), y: float(v("rev.reliability.lm.5.fraud_rate"))), (x: float(v("rev.reliability.lm.6.mean_p")), y: float(v("rev.reliability.lm.6.fraud_rate"))), (x: float(v("rev.reliability.lm.7.mean_p")), y: float(v("rev.reliability.lm.7.fraud_rate"))), (x: float(v("rev.reliability.lm.8.mean_p")), y: float(v("rev.reliability.lm.8.fraud_rate"))), (x: float(v("rev.reliability.lm.9.mean_p")), y: float(v("rev.reliability.lm.9.fraud_rate"))))), rel-panel([GBDT\*], ((x: float(v("rev.reliability.gbdt.0.mean_p")), y: float(v("rev.reliability.gbdt.0.fraud_rate"))), (x: float(v("rev.reliability.gbdt.1.mean_p")), y: float(v("rev.reliability.gbdt.1.fraud_rate"))), (x: float(v("rev.reliability.gbdt.2.mean_p")), y: float(v("rev.reliability.gbdt.2.fraud_rate"))), (x: float(v("rev.reliability.gbdt.3.mean_p")), y: float(v("rev.reliability.gbdt.3.fraud_rate"))), (x: float(v("rev.reliability.gbdt.4.mean_p")), y: float(v("rev.reliability.gbdt.4.fraud_rate"))), (x: float(v("rev.reliability.gbdt.5.mean_p")), y: float(v("rev.reliability.gbdt.5.fraud_rate"))), (x: float(v("rev.reliability.gbdt.6.mean_p")), y: float(v("rev.reliability.gbdt.6.fraud_rate"))), (x: float(v("rev.reliability.gbdt.7.mean_p")), y: float(v("rev.reliability.gbdt.7.fraud_rate"))), (x: float(v("rev.reliability.gbdt.8.mean_p")), y: float(v("rev.reliability.gbdt.8.fraud_rate"))), (x: float(v("rev.reliability.gbdt.9.mean_p")), y: float(v("rev.reliability.gbdt.9.fraud_rate")))))),
  caption: [Reliability of the rule's $p$ (Platt-scaled score) on the final test, per setup (read-out): ten equal-count bins, mean $p$ (across) against the observed fraud rate (up), both on a square-root scale; dashed: perfect calibration. \*GBDT is a read-out setup.],
) <fig-reliability>]

#let gcell(who, d, dist) = table.cell(fill: if who == "gbdt" { luma(225) } else if who == "lmtree" { rgb("#cfe0f3") } else { white },
  align(center, { set text(6pt); [#(if who == "gbdt" { "GBDT" } else if who == "lmtree" { "LM+tree" } else { who })] ; linebreak(); if dist == "yes" { strong(d) } else { d } }))
#let fig-grid = [#figure(
  placement: none,
  {
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto,) + (1fr,) * 9,
    stroke: 0.3pt + white,
    inset: 3pt,
    table.header(table.cell(rowspan: 1)[#text(6pt)[friction \ rate]], [#text(6pt)[#v("rev.grid.cells.0.c_r") / #v("rev.grid.cells.0.customer")]], [#text(6pt)[#v("rev.grid.cells.1.c_r") / #v("rev.grid.cells.1.customer")]], [#text(6pt)[#v("rev.grid.cells.2.c_r") / #v("rev.grid.cells.2.customer")]], [#text(6pt)[#v("rev.grid.cells.3.c_r") / #v("rev.grid.cells.3.customer")]], [#text(6pt)[#v("rev.grid.cells.4.c_r") / #v("rev.grid.cells.4.customer")]], [#text(6pt)[#v("rev.grid.cells.5.c_r") / #v("rev.grid.cells.5.customer")]], [#text(6pt)[#v("rev.grid.cells.6.c_r") / #v("rev.grid.cells.6.customer")]], [#text(6pt)[#v("rev.grid.cells.7.c_r") / #v("rev.grid.cells.7.customer")]], [#text(6pt)[#v("rev.grid.cells.8.c_r") / #v("rev.grid.cells.8.customer")]]),
    [#v("rev.grid.cells.0.r|pct")], gcell(v("rev.grid.cells.0.cheapest"), v("rev.grid.cells.0.vs_lr|signed"), v("rev.grid.cells.0.dist")), gcell(v("rev.grid.cells.1.cheapest"), v("rev.grid.cells.1.vs_lr|signed"), v("rev.grid.cells.1.dist")), gcell(v("rev.grid.cells.2.cheapest"), v("rev.grid.cells.2.vs_lr|signed"), v("rev.grid.cells.2.dist")), gcell(v("rev.grid.cells.3.cheapest"), v("rev.grid.cells.3.vs_lr|signed"), v("rev.grid.cells.3.dist")), gcell(v("rev.grid.cells.4.cheapest"), v("rev.grid.cells.4.vs_lr|signed"), v("rev.grid.cells.4.dist")), gcell(v("rev.grid.cells.5.cheapest"), v("rev.grid.cells.5.vs_lr|signed"), v("rev.grid.cells.5.dist")), gcell(v("rev.grid.cells.6.cheapest"), v("rev.grid.cells.6.vs_lr|signed"), v("rev.grid.cells.6.dist")), gcell(v("rev.grid.cells.7.cheapest"), v("rev.grid.cells.7.vs_lr|signed"), v("rev.grid.cells.7.dist")), gcell(v("rev.grid.cells.8.cheapest"), v("rev.grid.cells.8.vs_lr|signed"), v("rev.grid.cells.8.dist")), [#v("rev.grid.cells.9.r|pct")], gcell(v("rev.grid.cells.9.cheapest"), v("rev.grid.cells.9.vs_lr|signed"), v("rev.grid.cells.9.dist")), gcell(v("rev.grid.cells.10.cheapest"), v("rev.grid.cells.10.vs_lr|signed"), v("rev.grid.cells.10.dist")), gcell(v("rev.grid.cells.11.cheapest"), v("rev.grid.cells.11.vs_lr|signed"), v("rev.grid.cells.11.dist")), gcell(v("rev.grid.cells.12.cheapest"), v("rev.grid.cells.12.vs_lr|signed"), v("rev.grid.cells.12.dist")), gcell(v("rev.grid.cells.13.cheapest"), v("rev.grid.cells.13.vs_lr|signed"), v("rev.grid.cells.13.dist")), gcell(v("rev.grid.cells.14.cheapest"), v("rev.grid.cells.14.vs_lr|signed"), v("rev.grid.cells.14.dist")), gcell(v("rev.grid.cells.15.cheapest"), v("rev.grid.cells.15.vs_lr|signed"), v("rev.grid.cells.15.dist")), gcell(v("rev.grid.cells.16.cheapest"), v("rev.grid.cells.16.vs_lr|signed"), v("rev.grid.cells.16.dist")), gcell(v("rev.grid.cells.17.cheapest"), v("rev.grid.cells.17.vs_lr|signed"), v("rev.grid.cells.17.dist")), [#v("rev.grid.cells.18.r|pct")], gcell(v("rev.grid.cells.18.cheapest"), v("rev.grid.cells.18.vs_lr|signed"), v("rev.grid.cells.18.dist")), gcell(v("rev.grid.cells.19.cheapest"), v("rev.grid.cells.19.vs_lr|signed"), v("rev.grid.cells.19.dist")), gcell(v("rev.grid.cells.20.cheapest"), v("rev.grid.cells.20.vs_lr|signed"), v("rev.grid.cells.20.dist")), gcell(v("rev.grid.cells.21.cheapest"), v("rev.grid.cells.21.vs_lr|signed"), v("rev.grid.cells.21.dist")), gcell(v("rev.grid.cells.22.cheapest"), v("rev.grid.cells.22.vs_lr|signed"), v("rev.grid.cells.22.dist")), gcell(v("rev.grid.cells.23.cheapest"), v("rev.grid.cells.23.vs_lr|signed"), v("rev.grid.cells.23.dist")), gcell(v("rev.grid.cells.24.cheapest"), v("rev.grid.cells.24.vs_lr|signed"), v("rev.grid.cells.24.dist")), gcell(v("rev.grid.cells.25.cheapest"), v("rev.grid.cells.25.vs_lr|signed"), v("rev.grid.cells.25.dist")), gcell(v("rev.grid.cells.26.cheapest"), v("rev.grid.cells.26.vs_lr|signed"), v("rev.grid.cells.26.dist")),
  )
  },
  caption: [Price grid on the final test (read-out). Rows: friction rate $r$; columns: review fee $c_r$ / charged price per good customer declined, in USD. Each cell names the cheapest of LR, LM + tree, LM alone and GBDT alone, with $lambda$ re-picked on dev under that cell's prices, and its mean difference from LR in USD per window; bold: distinguishable from LR. The base cell (#v("costs.friction_rate|pct"), #v("costs.review_fee|num") / 0) reproduces the saved costs exactly.],
) <fig-grid>]

#let fig-tradeoff = [#figure(
  placement: top,
  scatter(((name: [LR], color: luma(60), dash: "solid", square: false, shift: 3pt, pts: ((x: float(v("rev.custline.logreg.points.0.declined")), y: float(v("rev.custline.logreg.points.0.stopped")), note: [#v("rev.custline.logreg.points.0.price") USD: #v("rev.custline.logreg.points.0.cost|int")]), (x: float(v("rev.custline.logreg.points.1.declined")), y: float(v("rev.custline.logreg.points.1.stopped")), note: none), (x: float(v("rev.custline.logreg.points.2.declined")), y: float(v("rev.custline.logreg.points.2.stopped")), note: [#v("rev.custline.logreg.points.2.price") USD: #v("rev.custline.logreg.points.2.cost|int")]), (x: float(v("rev.custline.logreg.points.3.declined")), y: float(v("rev.custline.logreg.points.3.stopped")), note: none), (x: float(v("rev.custline.logreg.points.4.declined")), y: float(v("rev.custline.logreg.points.4.stopped")), note: none), (x: float(v("rev.custline.logreg.points.5.declined")), y: float(v("rev.custline.logreg.points.5.stopped")), note: none), (x: float(v("rev.custline.logreg.points.6.declined")), y: float(v("rev.custline.logreg.points.6.stopped")), note: none), (x: float(v("rev.custline.logreg.points.7.declined")), y: float(v("rev.custline.logreg.points.7.stopped")), note: none), (x: float(v("rev.custline.logreg.points.8.declined")), y: float(v("rev.custline.logreg.points.8.stopped")), note: none), (x: float(v("rev.custline.logreg.points.9.declined")), y: float(v("rev.custline.logreg.points.9.stopped")), note: none), (x: float(v("rev.custline.logreg.points.10.declined")), y: float(v("rev.custline.logreg.points.10.stopped")), note: none), (x: float(v("rev.custline.logreg.points.11.declined")), y: float(v("rev.custline.logreg.points.11.stopped")), note: none), (x: float(v("rev.custline.logreg.points.12.declined")), y: float(v("rev.custline.logreg.points.12.stopped")), note: [#v("rev.custline.logreg.points.12.price") USD: #v("rev.custline.logreg.points.12.cost|int")]))), (name: [LM + tree], color: rgb("#2f6db5"), dash: "dashed", square: true, shift: -9pt, pts: ((x: float(v("rev.custline.lm-signal.points.0.declined")), y: float(v("rev.custline.lm-signal.points.0.stopped")), note: [#v("rev.custline.lm-signal.points.0.price") USD: #v("rev.custline.lm-signal.points.0.cost|int")]), (x: float(v("rev.custline.lm-signal.points.1.declined")), y: float(v("rev.custline.lm-signal.points.1.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.2.declined")), y: float(v("rev.custline.lm-signal.points.2.stopped")), note: [#v("rev.custline.lm-signal.points.2.price") USD: #v("rev.custline.lm-signal.points.2.cost|int")]), (x: float(v("rev.custline.lm-signal.points.3.declined")), y: float(v("rev.custline.lm-signal.points.3.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.4.declined")), y: float(v("rev.custline.lm-signal.points.4.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.5.declined")), y: float(v("rev.custline.lm-signal.points.5.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.6.declined")), y: float(v("rev.custline.lm-signal.points.6.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.7.declined")), y: float(v("rev.custline.lm-signal.points.7.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.8.declined")), y: float(v("rev.custline.lm-signal.points.8.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.9.declined")), y: float(v("rev.custline.lm-signal.points.9.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.10.declined")), y: float(v("rev.custline.lm-signal.points.10.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.11.declined")), y: float(v("rev.custline.lm-signal.points.11.stopped")), note: none), (x: float(v("rev.custline.lm-signal.points.12.declined")), y: float(v("rev.custline.lm-signal.points.12.stopped")), note: none)))), [good customers declined (#v("manual.tabular_dev.chains") dev windows)], [frauds stopped (of #v("rev.custline.logreg.frauds|int"))]),
  caption: [The customer line moves the operating point (dev). Each point is a price per good customer turned away, from #v("rev.custline.logreg.points.0.price") to #v("rev.custline.logreg.points.12.price") USD in steps of #v("rev.custline.logreg.points.1.price"), added to the rule's decision only; notes give the price and the mean cost per window. Grey circles: LR; blue squares: LM + tree. Fewer good customers declined comes with fewer frauds stopped and equal or higher cost.],
) <tradeoff>]

#let table-disp-gd = [#figure(
  placement: none,
  {
  set text(6pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([group], [n], [LR], [LM + tree], [LM + tree, cust.], [LM alone], [GBDT\*]),
    table.hline(stroke: 0.4pt),
    [credit card], [#v("rev.disparity.card6.credit.n_good|int")], [#v("rev.disparity.card6.credit.lr.gd") (#v("rev.disparity.card6.credit.lr.gd_ci"))], [#v("rev.disparity.card6.credit.lmtree.gd") (#v("rev.disparity.card6.credit.lmtree.gd_ci"))], [#v("rev.disparity.card6.credit.lmtree_cust.gd") (#v("rev.disparity.card6.credit.lmtree_cust.gd_ci"))], [#v("rev.disparity.card6.credit.lm.gd") (#v("rev.disparity.card6.credit.lm.gd_ci"))], [#v("rev.disparity.card6.credit.gbdt.gd") (#v("rev.disparity.card6.credit.gbdt.gd_ci"))],
    [debit card], [#v("rev.disparity.card6.debit.n_good|int")], [#v("rev.disparity.card6.debit.lr.gd") (#v("rev.disparity.card6.debit.lr.gd_ci"))], [#v("rev.disparity.card6.debit.lmtree.gd") (#v("rev.disparity.card6.debit.lmtree.gd_ci"))], [#v("rev.disparity.card6.debit.lmtree_cust.gd") (#v("rev.disparity.card6.debit.lmtree_cust.gd_ci"))], [#v("rev.disparity.card6.debit.lm.gd") (#v("rev.disparity.card6.debit.lm.gd_ci"))], [#v("rev.disparity.card6.debit.gbdt.gd") (#v("rev.disparity.card6.debit.gbdt.gd_ci"))],
    [desktop], [#v("rev.disparity.DeviceType.desktop.n_good|int")], [#v("rev.disparity.DeviceType.desktop.lr.gd") (#v("rev.disparity.DeviceType.desktop.lr.gd_ci"))], [#v("rev.disparity.DeviceType.desktop.lmtree.gd") (#v("rev.disparity.DeviceType.desktop.lmtree.gd_ci"))], [#v("rev.disparity.DeviceType.desktop.lmtree_cust.gd") (#v("rev.disparity.DeviceType.desktop.lmtree_cust.gd_ci"))], [#v("rev.disparity.DeviceType.desktop.lm.gd") (#v("rev.disparity.DeviceType.desktop.lm.gd_ci"))], [#v("rev.disparity.DeviceType.desktop.gbdt.gd") (#v("rev.disparity.DeviceType.desktop.gbdt.gd_ci"))],
    [mobile], [#v("rev.disparity.DeviceType.mobile.n_good|int")], [#v("rev.disparity.DeviceType.mobile.lr.gd") (#v("rev.disparity.DeviceType.mobile.lr.gd_ci"))], [#v("rev.disparity.DeviceType.mobile.lmtree.gd") (#v("rev.disparity.DeviceType.mobile.lmtree.gd_ci"))], [#v("rev.disparity.DeviceType.mobile.lmtree_cust.gd") (#v("rev.disparity.DeviceType.mobile.lmtree_cust.gd_ci"))], [#v("rev.disparity.DeviceType.mobile.lm.gd") (#v("rev.disparity.DeviceType.mobile.lm.gd_ci"))], [#v("rev.disparity.DeviceType.mobile.gbdt.gd") (#v("rev.disparity.DeviceType.mobile.gbdt.gd_ci"))],
    [no device recorded], [#v("rev.disparity.DeviceType.none.n_good|int")], [#v("rev.disparity.DeviceType.none.lr.gd") (#v("rev.disparity.DeviceType.none.lr.gd_ci"))], [#v("rev.disparity.DeviceType.none.lmtree.gd") (#v("rev.disparity.DeviceType.none.lmtree.gd_ci"))], [#v("rev.disparity.DeviceType.none.lmtree_cust.gd") (#v("rev.disparity.DeviceType.none.lmtree_cust.gd_ci"))], [#v("rev.disparity.DeviceType.none.lm.gd") (#v("rev.disparity.DeviceType.none.lm.gd_ci"))], [#v("rev.disparity.DeviceType.none.gbdt.gd") (#v("rev.disparity.DeviceType.none.gbdt.gd_ci"))],
    [gmail.com], [#v("rev.disparity.P_emaildomain.gmail_com.n_good|int")], [#v("rev.disparity.P_emaildomain.gmail_com.lr.gd") (#v("rev.disparity.P_emaildomain.gmail_com.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lmtree.gd") (#v("rev.disparity.P_emaildomain.gmail_com.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.gmail_com.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lm.gd") (#v("rev.disparity.P_emaildomain.gmail_com.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.gbdt.gd") (#v("rev.disparity.P_emaildomain.gmail_com.gbdt.gd_ci"))],
    [yahoo.com], [#v("rev.disparity.P_emaildomain.yahoo_com.n_good|int")], [#v("rev.disparity.P_emaildomain.yahoo_com.lr.gd") (#v("rev.disparity.P_emaildomain.yahoo_com.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lmtree.gd") (#v("rev.disparity.P_emaildomain.yahoo_com.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.yahoo_com.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lm.gd") (#v("rev.disparity.P_emaildomain.yahoo_com.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.gbdt.gd") (#v("rev.disparity.P_emaildomain.yahoo_com.gbdt.gd_ci"))],
    [hotmail.com], [#v("rev.disparity.P_emaildomain.hotmail_com.n_good|int")], [#v("rev.disparity.P_emaildomain.hotmail_com.lr.gd") (#v("rev.disparity.P_emaildomain.hotmail_com.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lmtree.gd") (#v("rev.disparity.P_emaildomain.hotmail_com.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.hotmail_com.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lm.gd") (#v("rev.disparity.P_emaildomain.hotmail_com.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.gbdt.gd") (#v("rev.disparity.P_emaildomain.hotmail_com.gbdt.gd_ci"))],
    [anonymous.com], [#v("rev.disparity.P_emaildomain.anonymous_com.n_good|int")], [#v("rev.disparity.P_emaildomain.anonymous_com.lr.gd") (#v("rev.disparity.P_emaildomain.anonymous_com.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lmtree.gd") (#v("rev.disparity.P_emaildomain.anonymous_com.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.anonymous_com.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lm.gd") (#v("rev.disparity.P_emaildomain.anonymous_com.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.gbdt.gd") (#v("rev.disparity.P_emaildomain.anonymous_com.gbdt.gd_ci"))],
    [aol.com], [#v("rev.disparity.P_emaildomain.aol_com.n_good|int")], [#v("rev.disparity.P_emaildomain.aol_com.lr.gd") (#v("rev.disparity.P_emaildomain.aol_com.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lmtree.gd") (#v("rev.disparity.P_emaildomain.aol_com.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.aol_com.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lm.gd") (#v("rev.disparity.P_emaildomain.aol_com.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.gbdt.gd") (#v("rev.disparity.P_emaildomain.aol_com.gbdt.gd_ci"))],
    [other domains], [#v("rev.disparity.P_emaildomain.other.n_good|int")], [#v("rev.disparity.P_emaildomain.other.lr.gd") (#v("rev.disparity.P_emaildomain.other.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.other.lmtree.gd") (#v("rev.disparity.P_emaildomain.other.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.other.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.other.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.other.lm.gd") (#v("rev.disparity.P_emaildomain.other.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.other.gbdt.gd") (#v("rev.disparity.P_emaildomain.other.gbdt.gd_ci"))],
    [no email], [#v("rev.disparity.P_emaildomain.none.n_good|int")], [#v("rev.disparity.P_emaildomain.none.lr.gd") (#v("rev.disparity.P_emaildomain.none.lr.gd_ci"))], [#v("rev.disparity.P_emaildomain.none.lmtree.gd") (#v("rev.disparity.P_emaildomain.none.lmtree.gd_ci"))], [#v("rev.disparity.P_emaildomain.none.lmtree_cust.gd") (#v("rev.disparity.P_emaildomain.none.lmtree_cust.gd_ci"))], [#v("rev.disparity.P_emaildomain.none.lm.gd") (#v("rev.disparity.P_emaildomain.none.lm.gd_ci"))], [#v("rev.disparity.P_emaildomain.none.gbdt.gd") (#v("rev.disparity.P_emaildomain.none.gbdt.gd_ci"))],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Share of good customers declined, % with 95% window-bootstrap interval, by card type, device type and purchaser email domain on the final test (read-out). A disparity check on proxies, not a fairness audit. \*GBDT is a read-out setup.],
) <table-disp-gd>]

#let table-disp-fm = [#figure(
  placement: none,
  {
  set text(6pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([group], [n], [LR], [LM + tree], [LM + tree, cust.], [LM alone], [GBDT\*]),
    table.hline(stroke: 0.4pt),
    [credit card], [#v("rev.disparity.card6.credit.n_fraud|int")], [#v("rev.disparity.card6.credit.lr.fm") (#v("rev.disparity.card6.credit.lr.fm_ci"))], [#v("rev.disparity.card6.credit.lmtree.fm") (#v("rev.disparity.card6.credit.lmtree.fm_ci"))], [#v("rev.disparity.card6.credit.lmtree_cust.fm") (#v("rev.disparity.card6.credit.lmtree_cust.fm_ci"))], [#v("rev.disparity.card6.credit.lm.fm") (#v("rev.disparity.card6.credit.lm.fm_ci"))], [#v("rev.disparity.card6.credit.gbdt.fm") (#v("rev.disparity.card6.credit.gbdt.fm_ci"))],
    [debit card], [#v("rev.disparity.card6.debit.n_fraud|int")], [#v("rev.disparity.card6.debit.lr.fm") (#v("rev.disparity.card6.debit.lr.fm_ci"))], [#v("rev.disparity.card6.debit.lmtree.fm") (#v("rev.disparity.card6.debit.lmtree.fm_ci"))], [#v("rev.disparity.card6.debit.lmtree_cust.fm") (#v("rev.disparity.card6.debit.lmtree_cust.fm_ci"))], [#v("rev.disparity.card6.debit.lm.fm") (#v("rev.disparity.card6.debit.lm.fm_ci"))], [#v("rev.disparity.card6.debit.gbdt.fm") (#v("rev.disparity.card6.debit.gbdt.fm_ci"))],
    [desktop], [#v("rev.disparity.DeviceType.desktop.n_fraud|int")], [#v("rev.disparity.DeviceType.desktop.lr.fm") (#v("rev.disparity.DeviceType.desktop.lr.fm_ci"))], [#v("rev.disparity.DeviceType.desktop.lmtree.fm") (#v("rev.disparity.DeviceType.desktop.lmtree.fm_ci"))], [#v("rev.disparity.DeviceType.desktop.lmtree_cust.fm") (#v("rev.disparity.DeviceType.desktop.lmtree_cust.fm_ci"))], [#v("rev.disparity.DeviceType.desktop.lm.fm") (#v("rev.disparity.DeviceType.desktop.lm.fm_ci"))], [#v("rev.disparity.DeviceType.desktop.gbdt.fm") (#v("rev.disparity.DeviceType.desktop.gbdt.fm_ci"))],
    [mobile], [#v("rev.disparity.DeviceType.mobile.n_fraud|int")], [#v("rev.disparity.DeviceType.mobile.lr.fm") (#v("rev.disparity.DeviceType.mobile.lr.fm_ci"))], [#v("rev.disparity.DeviceType.mobile.lmtree.fm") (#v("rev.disparity.DeviceType.mobile.lmtree.fm_ci"))], [#v("rev.disparity.DeviceType.mobile.lmtree_cust.fm") (#v("rev.disparity.DeviceType.mobile.lmtree_cust.fm_ci"))], [#v("rev.disparity.DeviceType.mobile.lm.fm") (#v("rev.disparity.DeviceType.mobile.lm.fm_ci"))], [#v("rev.disparity.DeviceType.mobile.gbdt.fm") (#v("rev.disparity.DeviceType.mobile.gbdt.fm_ci"))],
    [no device recorded], [#v("rev.disparity.DeviceType.none.n_fraud|int")], [#v("rev.disparity.DeviceType.none.lr.fm") (#v("rev.disparity.DeviceType.none.lr.fm_ci"))], [#v("rev.disparity.DeviceType.none.lmtree.fm") (#v("rev.disparity.DeviceType.none.lmtree.fm_ci"))], [#v("rev.disparity.DeviceType.none.lmtree_cust.fm") (#v("rev.disparity.DeviceType.none.lmtree_cust.fm_ci"))], [#v("rev.disparity.DeviceType.none.lm.fm") (#v("rev.disparity.DeviceType.none.lm.fm_ci"))], [#v("rev.disparity.DeviceType.none.gbdt.fm") (#v("rev.disparity.DeviceType.none.gbdt.fm_ci"))],
    [gmail.com], [#v("rev.disparity.P_emaildomain.gmail_com.n_fraud|int")], [#v("rev.disparity.P_emaildomain.gmail_com.lr.fm") (#v("rev.disparity.P_emaildomain.gmail_com.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lmtree.fm") (#v("rev.disparity.P_emaildomain.gmail_com.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.gmail_com.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.lm.fm") (#v("rev.disparity.P_emaildomain.gmail_com.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.gmail_com.gbdt.fm") (#v("rev.disparity.P_emaildomain.gmail_com.gbdt.fm_ci"))],
    [yahoo.com], [#v("rev.disparity.P_emaildomain.yahoo_com.n_fraud|int")], [#v("rev.disparity.P_emaildomain.yahoo_com.lr.fm") (#v("rev.disparity.P_emaildomain.yahoo_com.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lmtree.fm") (#v("rev.disparity.P_emaildomain.yahoo_com.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.yahoo_com.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.lm.fm") (#v("rev.disparity.P_emaildomain.yahoo_com.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.yahoo_com.gbdt.fm") (#v("rev.disparity.P_emaildomain.yahoo_com.gbdt.fm_ci"))],
    [hotmail.com], [#v("rev.disparity.P_emaildomain.hotmail_com.n_fraud|int")], [#v("rev.disparity.P_emaildomain.hotmail_com.lr.fm") (#v("rev.disparity.P_emaildomain.hotmail_com.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lmtree.fm") (#v("rev.disparity.P_emaildomain.hotmail_com.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.hotmail_com.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.lm.fm") (#v("rev.disparity.P_emaildomain.hotmail_com.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.hotmail_com.gbdt.fm") (#v("rev.disparity.P_emaildomain.hotmail_com.gbdt.fm_ci"))],
    [anonymous.com], [#v("rev.disparity.P_emaildomain.anonymous_com.n_fraud|int")], [#v("rev.disparity.P_emaildomain.anonymous_com.lr.fm") (#v("rev.disparity.P_emaildomain.anonymous_com.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lmtree.fm") (#v("rev.disparity.P_emaildomain.anonymous_com.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.anonymous_com.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.lm.fm") (#v("rev.disparity.P_emaildomain.anonymous_com.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.anonymous_com.gbdt.fm") (#v("rev.disparity.P_emaildomain.anonymous_com.gbdt.fm_ci"))],
    [aol.com], [#v("rev.disparity.P_emaildomain.aol_com.n_fraud|int")], [#v("rev.disparity.P_emaildomain.aol_com.lr.fm") (#v("rev.disparity.P_emaildomain.aol_com.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lmtree.fm") (#v("rev.disparity.P_emaildomain.aol_com.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.aol_com.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.lm.fm") (#v("rev.disparity.P_emaildomain.aol_com.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.aol_com.gbdt.fm") (#v("rev.disparity.P_emaildomain.aol_com.gbdt.fm_ci"))],
    [other domains], [#v("rev.disparity.P_emaildomain.other.n_fraud|int")], [#v("rev.disparity.P_emaildomain.other.lr.fm") (#v("rev.disparity.P_emaildomain.other.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.other.lmtree.fm") (#v("rev.disparity.P_emaildomain.other.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.other.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.other.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.other.lm.fm") (#v("rev.disparity.P_emaildomain.other.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.other.gbdt.fm") (#v("rev.disparity.P_emaildomain.other.gbdt.fm_ci"))],
    [no email], [#v("rev.disparity.P_emaildomain.none.n_fraud|int")], [#v("rev.disparity.P_emaildomain.none.lr.fm") (#v("rev.disparity.P_emaildomain.none.lr.fm_ci"))], [#v("rev.disparity.P_emaildomain.none.lmtree.fm") (#v("rev.disparity.P_emaildomain.none.lmtree.fm_ci"))], [#v("rev.disparity.P_emaildomain.none.lmtree_cust.fm") (#v("rev.disparity.P_emaildomain.none.lmtree_cust.fm_ci"))], [#v("rev.disparity.P_emaildomain.none.lm.fm") (#v("rev.disparity.P_emaildomain.none.lm.fm_ci"))], [#v("rev.disparity.P_emaildomain.none.gbdt.fm") (#v("rev.disparity.P_emaildomain.none.gbdt.fm_ci"))],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Share of frauds missed, % with 95% window-bootstrap interval, by card type, device type and purchaser email domain on the final test (read-out). A disparity check on proxies, not a fairness audit. \*GBDT is a read-out setup.],
) <table-disp-fm>]
#let table-disparity = [#table-disp-gd #table-disp-fm]

#let table-parity = [#figure(
  placement: none,
  {
  set text(7pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (1fr, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([setup or difference], [USD / window], [AUC, or 95% interval]),
    table.hline(stroke: 0.4pt),
    [GBDT, all columns], [#v("rev.parity.setups.tab-all.mean|int")], [#v("rev.parity.setups.tab-all.auc|auc")],
    [LM + tree], [#v("rev.parity.setups.lm-signal.mean|int")], [#v("rev.parity.setups.lm-signal.auc|auc")],
    [GBDT, prompt's fields], [#v("rev.parity.setups.par-trees.mean|int")], [#v("rev.parity.setups.par-trees.auc|auc")],
    [LM alone], [#v("rev.parity.setups.lm-rich.mean|int")], [#v("rev.parity.setups.lm-rich.auc|auc")],
    [LR, all columns], [#v("rev.parity.setups.tab-logreg.mean|int")], [#v("rev.parity.setups.tab-logreg.auc|auc")],
    [LR, prompt's fields], [#v("rev.parity.setups.par-logreg.mean|int")], [#v("rev.parity.setups.par-logreg.auc|auc")],
    [LM alone − LR, prompt's fields], [#v("rev.parity.cmp.lm-rich__par-logreg.mean|signed")], [#v("rev.parity.cmp.lm-rich__par-logreg.ci|ci"), #v("rev.parity.cmp.lm-rich__par-logreg|dist")],
    [LM alone − GBDT, prompt's fields], [#v("rev.parity.cmp.lm-rich__par-trees.mean|signed")], [#v("rev.parity.cmp.lm-rich__par-trees.ci|ci"), #v("rev.parity.cmp.lm-rich__par-trees|dist")],
    [LM + tree − GBDT, prompt's fields], [#v("rev.parity.cmp.lm-signal__par-trees.mean|signed")], [#v("rev.parity.cmp.lm-signal__par-trees.ci|ci"), #v("rev.parity.cmp.lm-signal__par-trees|dist")],
    [LR: prompt's fields − all columns], [#v("rev.parity.cmp.par-logreg__tab-logreg.mean|signed")], [#v("rev.parity.cmp.par-logreg__tab-logreg.ci|ci"), #v("rev.parity.cmp.par-logreg__tab-logreg|dist")],
    [GBDT: prompt's fields − all columns], [#v("rev.parity.cmp.par-trees__tab-all.mean|signed")], [#v("rev.parity.cmp.par-trees__tab-all.ci|ci"), #v("rev.parity.cmp.par-trees__tab-all|dist")],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Feature parity on #v("manual.tabular_dev.chains") dev windows (dev): LR and GBDT refitted on exactly the richer prompt's fields and its card and address history. $p$ on #v("rev.protocol.dev_windows_30") dev windows and $lambda$ on all #v("rev.protocol.dev_windows_all") for every row, so LR here differs slightly from the paper's baseline.],
) <table-parity>]

#let table-noc = [#figure(
  placement: none,
  {
  set text(7pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (1fr, auto, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([setup], [with C], [without C], [difference (95% interval)]),
    table.hline(stroke: 0.4pt),
    [GBDT], [#v("rev.noc.setups.tab-all.mean|int")], [#v("rev.noc.setups.noc-trees.mean|int")], [#v("rev.noc.cmp.noc-trees__tab-all.mean|signed") (#v("rev.noc.cmp.noc-trees__tab-all.ci|ci"))],
    [LM alone], [#v("rev.noc.setups.lm-rich.mean|int")], [#v("rev.noc.setups.lm-rich-noc.mean|int")], [#v("rev.noc.cmp.lm-rich-noc__lm-rich.mean|signed") (#v("rev.noc.cmp.lm-rich-noc__lm-rich.ci|ci"))],
    [LR], [#v("rev.noc.setups.tab-logreg.mean|int")], [#v("rev.noc.setups.noc-logreg.mean|int")], [#v("rev.noc.cmp.noc-logreg__tab-logreg.mean|signed") (#v("rev.noc.cmp.noc-logreg__tab-logreg.ci|ci"))],
    [GBDT − LR, no C], [], [], [#v("rev.noc.cmp.noc-trees__noc-logreg.mean|signed") (#v("rev.noc.cmp.noc-trees__noc-logreg.ci|ci"))],
    [LM alone − LR, no C], [], [], [#v("rev.noc.cmp.lm-rich-noc__noc-logreg.mean|signed") (#v("rev.noc.cmp.lm-rich-noc__noc-logreg.ci|ci"))],
    [LM alone − GBDT, no C], [], [], [#v("rev.noc.cmp.lm-rich-noc__noc-trees.mean|signed") (#v("rev.noc.cmp.lm-rich-noc__noc-trees.ci|ci"))],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Without any C column, #v("manual.tabular_dev.chains") dev windows (dev): LR and GBDT refitted without C1–C14; one LoRA retrain of LM alone with no C column in its prompt or context, same rows and settings. Every setup gets distinguishably dearer, and the ranking of LM alone and LR changes. LM + tree was not run without C.],
) <table-noc>]

#let table-order = [#figure(
  placement: none,
  {
  set text(7pt)
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (1fr, auto, auto, auto, auto),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header([setup], [top answer flips], [as scored], [forward], [forward − as scored]),
    table.hline(stroke: 0.4pt),
    [LM alone], [#v("rev.order.lm-rich.flip_pct")%], [#v("rev.order.lm-rich.rev|int")], [#v("rev.order.lm-rich.fwd|int")], [#v("rev.order.lm-rich.fwd_rev.mean|signed") (#v("rev.order.lm-rich.fwd_rev.ci|ci"))],
    [LM + tree], [#v("rev.order.lm-signal.flip_pct")%], [#v("rev.order.lm-signal.rev|int")], [#v("rev.order.lm-signal.fwd|int")], [#v("rev.order.lm-signal.fwd_rev.mean|signed") (#v("rev.order.lm-signal.fwd_rev.ci|ci"))],
    table.hline(stroke: 0.6pt),
  )
  },
  caption: [Option-order bias on #v("manual.tabular_dev.chains") dev windows (dev). As scored: A decline, B review, C approve; forward: A approve, B review, C decline. Averaging both orders for LM alone, triggered by the registered rule, changed its cost by #v("rev.order.lm-rich.avg_rev.mean|signed") (#v("rev.order.lm-rich.avg_rev.ci|ci")).],
) <table-order>]

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
      [answer], [letter], [logit], [score], [letter], [logit], [score], [score],
    ),
    table.hline(stroke: 0.4pt),
    [approve], [A], [#v("worked.logits.listed.0|d2")], [#v("worked.orders.listed.0|d2")], [C], [#v("worked.logits.reversed.0|d2")], [#v("worked.orders.reversed.0|d2")], [#v("worked.scores.0|d2")],
    [review], [B], [#v("worked.logits.listed.1|d2")], [#v("worked.orders.listed.1|d2")], [B], [#v("worked.logits.reversed.1|d2")], [#v("worked.orders.reversed.1|d2")], [#v("worked.scores.1|d2")],
    [decline], [C], [#v("worked.logits.listed.2|d2")], [*#v("worked.orders.listed.2|d2")*], [A], [#v("worked.logits.reversed.2|d2")], [#v("worked.orders.reversed.2|d2")], [*#v("worked.scores.2|d2")*],
    table.hline(stroke: 0.6pt),
  ),
  caption: [Option-order bias on PaySim: the worked payment's logits (shifted so the top is zero) and softmax scores with the options as listed and reversed (untrained Qwen2.5-1.5B). The top answer follows the last letter.],
) <worked>]

#let prompts = {
  set par(first-line-indent: 0pt, justify: false)
  [The plain prompt's state, the richer prompt's state, and the tree-score line, each filled for one fit row chosen by a written rule (the first fit row, in time order, with earlier card payments and a recorded device). The full text the model reads, here for the richer prompt, adds the context block, the question and the options in the order scored, inside the model's chat template; the model's answer slot follows the last line.]
  std.v(4pt)
  [*Plain state.*]
  block(inset: 4pt, stroke: 0.4pt + luma(160), width: 100%, raw(v("rev.prompt.state_plain"), block: true))
  [*Tree-score line* (LM + tree only).]
  block(inset: 4pt, stroke: 0.4pt + luma(160), width: 100%, raw(v("rev.prompt.signal_line"), block: true))
  [*Full richer prompt.*]
  block(inset: 4pt, stroke: 0.4pt + luma(160), width: 100%, { set text(6.5pt); raw(v("rev.prompt.full_rich"), block: true) })
}
// generated: end

// ---- the paper --------------------------------------------------------------------------------------------

= Introduction

Every card payment needs a decision before the money moves: _approve_ it, send it to a person for _review_, or _decline_ it. Each can be wrong in its own way. Approve a fraud and the money is gone; decline a genuine payment and a good customer is turned away; a review finds the truth but costs a fee, and only a few payments can get one. Card fraud losses worldwide were #v("manual.nilson.usd_billions") billion USD in #v("manual.nilson.year") #cite-k("nilson").

Fraud teams usually score each payment with a tabular model, such as logistic regression (LR) or gradient-boosted decision trees (GBDT), and turn the score into an action. Language models can also classify: written as text with lettered options, a payment can be scored by reading the model's next-token scores for the letters in one forward pass #cite-k("robinson"), and a small adapter can train it to answer with the right letter, as in prompt-based fine-tuning with a verbalizer #cite-k("pet"). We apply that existing technique; it is not our contribution.#footnote[The idea of a language model as a fast decision function, one that chooses among options in one pass instead of writing, came to this project from System One models: TypeSafe, “Introducing System One models & Jev,” 2026, and S. Goedecke, “Two techniques for working with System One models,” 2026.]

*Research question.* Does a small fine-tuned language model, used as a single-pass three-way classifier, lower decision cost under assumed prices against LR and GBDT on real payments; and where does any difference come from: the model, the information it sees, or the decision rule?

*Contributions.* (i) A locked, pre-registered, dollar-costed evaluation protocol for three-way fraud decisions, and its results on IEEE-CIS. (ii) Evidence that the decision rule changes outcomes more than the model does. (iii) An account of the biases and limitations of the evaluation itself, including option-order bias, optimism from choosing on development data, columns of uncertain timing, and the absence of unscored test data.#code-and-data

In short: pricing each action per payment changed costs far more than any choice of model; the fine-tuned model showed no detectable saving over LR; trees read out on the same windows were cheapest but not distinguishably so; and the model's gap to the tabular baselines on development data is mostly a gap in information.

= Related Work

== Language models on tabular data
Fraud detection on cards is a classification problem with rare positives, drift over time and a limited budget for checking alerts; a realistic evaluation must respect all three #cite-k("dalpozzolo"). TabLLM serialises table rows into sentences and classifies them with a language model, zero-shot or fine-tuned on a few labelled rows #cite-k("tabllm"). Tan, Ma and Zhang prompt larger open models on structured fraud data, including IEEE-CIS and PaySim, and find that direct prompting lags tabular baselines unless retrieval and feature selection are added #cite-k("tan"). LIFT fine-tunes language models on serialised rows for non-language tasks #cite-k("lift"), and a survey covers the wider area #cite-k("fang"). We did not test the main alternatives: a tabular foundation model such as TabPFN #cite-k("tabpfn"), language-model embeddings of each row fed into GBDT #cite-k("fang"), or large language models used zero-shot #cite-k("tan").

== Prompt-based fine-tuning and verbalizers
Pattern-exploiting training (PET) rewrites inputs as cloze questions and maps each class to a word, the _verbalizer_, so the language-model head itself becomes the classifier #cite-k("pet")\; LM-BFF extends this with automatically found prompts and demonstrations #cite-k("lmbff"). Robinson and Wingate show that presenting options with letters, so that one pass scores every option through its letter, beats scoring each full answer #cite-k("robinson"). Our classifier is this letter-verbalizer form with LoRA training. Models lean towards option positions or letters: Pezeshkpour and Hruschka find large swings when options are reordered #cite-k("pezeshkpour"), and Zheng et al. trace the bias to the option identifiers #cite-k("zheng"). We measure this _option-order bias_ directly.

== Cost-sensitive fraud evaluation
When errors have different costs, the decision that minimises expected cost uses the class probability and the costs, not the most likely class #cite-k("elkan"). Correa Bahnsen et al. apply this Bayes minimum-risk rule to card fraud with amount-dependent costs #cite-k("bahnsen"). Our cost rule is that standard rule, with a third action, a review, under a budget. Aggregating a card's past payments into features is standard in fraud detection #cite-k("whitrow")\; our richer prompt states such aggregates in words.

== Tree ensembles on tables
Tree ensembles remain hard to beat on medium-sized tabular data #cite-k("grinsztajn")\; gradient boosting #cite-k("friedman") is therefore our strongest classical reference.

*The gap.* Among the studies reviewed, none reports a locked, dollar-costed, pre-registered comparison of fine-tuned small language models against linear and tree baselines on real payments.

= Data and Protocol

== Data and split
The evaluation uses IEEE-CIS #cite-k("ieeecis"): #v("manual.split.rows|int") labelled e-commerce payments, #v("manual.split.fraud_share") fraud, many of whose columns are anonymised counters (C), time gaps (D) and engineered features (V). It was split by time and never shuffled: the _fit_ part (#v("manual.split.fit|int") payments) trains the models, the _development_ (dev) part (#v("manual.split.dev|int")) is where every choice is made, and the _test_ part (#v("manual.split.test|int")) is never tuned on. PaySim #cite-k("paysim"), a simulator of mobile-money payments, served only to develop the pipeline (Section V-A). The work followed CRISP-DM #cite-k("crisp") in these two cycles.

A _window_ is #v("manual.data.window_rows|int") consecutive payments. Every setup decides the same windows, so setups are compared window by window. The _final test_ is #v("final.windows_n") test windows: #v("final.rows|int") payments with #v("final.frauds|int") frauds, worth #v("final.fraud_usd|int") USD.

== Pre-registration and labels
Before any setup scored a final-test row, its windows, setups, comparisons and decision rule were written down and committed, and each setup was then scored once #cite-k("nosek"). Every number from the final test or from this revision's analyses carries one of three labels: _dev_ (development data, where choices were made), _held-out_ (the registered final test), or _read-out_ (computed after the registered results were seen, a post-hoc analysis that cannot test anything). Numbers from earlier test windows, used before the final test, are marked as such. An audit of every earlier evaluation found that all #v("rev.audit.windows_scored|int") possible test windows had already been scored by some setup and only #v("rev.audit.never_scored_rows|int") test rows had not; no clean window remained for a new held-out test (Section V-D).

== Judging by dollars
Accuracy is the wrong yardstick: approving every payment is #v("final.approve_all_accuracy|pct") accurate on the final test (held-out) and the dearest setup we ran. We price every decision. For a window $w$,
$ C(w) = sum_("fraud approved") a_i + sum_("good declined") f(a_i) + c_r dot n_r, $ <cost>
where $a_i$ is a payment's amount, $f(a) = max(r a, m)$ the _friction_ of turning a good customer away, $c_r$ the review fee and $n_r$ the number of reviews. We set $r =$ #v("costs.friction_rate|pct"), $m =$ #v("costs.friction_min|num") USD, $c_r =$ #v("costs.review_fee|num") USD, and allow at most #v("costs.review_share|pct") of a window to be reviewed; a review is assumed to find the truth. These prices are assumed, not measured (Section V-C).

== Paired differences, intervals and power
To compare setups $A$ and $B$ we take each window's difference $d_w = C_A (w) - C_B (w)$ and its mean $overline(d)$; negative means $A$ is cheaper. The window, not the payment, is the unit: neighbouring payments are not independent and one large fraud can dominate a window. The registered interval is the #v("manual.conventions.ci_level")% percentile interval of $overline(d)$ from a paired window bootstrap with #v("manual.recipe.boot_reps|int") resamples; two setups are _distinguishable_ only when it excludes zero #cite-k("dror"). As robustness checks, read out after the test, we add moving-block bootstraps over windows in time order #cite-k("kunsch") and bias-corrected and accelerated (BCa) intervals #cite-k("efron") (Appendix A). The _minimum detectable effect_ (MDE) of a comparison is $#v("rev.protocol.mde_factor") s slash sqrt(n)$, with $s$ the paired standard deviation and $n$ the windows: the difference a test of this size detects with #v("rev.protocol.mde_power_pct")% power at the #v("rev.protocol.mde_alpha_pct")% level, two-sided. Two training seeds of the same language-model recipe differed by #v("rev.seed.mean|signed") USD per window on dev (#v("rev.seed.ci|ci")), so we treat differences of about #v("rev.seed.floor|int") USD per window as noise.

= Methods

== The cost rule: Bayes minimum risk with a review budget
Every model gives a fraud score. A logistic fit of the score's logit on dev, _Platt scaling_ #cite-k("platt"), turns it into $p$, used only to weigh costs. The rule then prices each action for this one payment and takes the cheapest #cite-k("elkan", "bahnsen"):
$ "approve:" p a, quad "decline:" (1 - p) f(a), quad "review:" c_r + lambda. $ <rule>
Here $lambda$ prices one of the few reviews and is picked on dev as the value with the lowest dev cost; once the budget is spent, the rule takes the cheaper of approve and decline. On the PaySim payment of @pipeline, with an illustrative $p approx$ #calc.round(float(v("illustration.rule.p")), digits: 4) and $lambda =$ #v("illustration.rule.lam|num"), approving would be expected to lose #v("illustration.rule.big.approve|usd2") USD, declining #v("illustration.rule.big.decline|usd2"), and a review costs #v("illustration.rule.review|usd2"): the rule asks for a review. The same $p$ on a #v("illustration.rule.small.amount|num") USD payment gives #v("illustration.rule.small.approve|usd2"), #v("illustration.rule.small.decline|usd2") and #v("illustration.rule.review|usd2"): now it declines. Two fixed cut-offs on the score, the usual alternative, would treat both payments alike.

The _customer line_ changes the rule only: when it weighs a decline it adds a flat #v("manual.customer_line.flat|num") USD for turning a good customer away. The price is never charged in Eq. @cost; it moves the rule's _operating point_ towards fewer declines.

== The baselines
LR is fitted on the fit part on all #v("rev.protocol.lr_features|int") features built from every non-identifier column. GBDT is #v("manual.recipe.trees") #cite-k("friedman") on the same columns. Neither sees the card-history aggregates the richer prompt states (Section IV-D).

== The language model as a classifier
A language model gives a score to every token as the possible next one. The payment is written as plain facts, followed by one question with three lettered options (@pipeline); the model reads it in one forward pass, and we keep only its scores for the letters A, B and C, the verbalizer #cite-k("robinson", "pet"). A softmax over these three, $P(k) = e^(z_k) slash sum_j e^(z_j)$, gives three numbers that add to one: a classifier with one score per class, from one pass and with no word written. For the PaySim payment the three letter scores, shifted so the top is zero, were #v("worked.logits.listed.0|d2"), #v("worked.logits.listed.1|d2") and #v("worked.logits.listed.2|d2")\; since $e^0 = 1$ and $e^(#v("worked.logits.listed.0|d2")) approx$ #v("worked.exp.listed.0|d2"), decline gets #v("worked.exp.listed.2|d2") $slash$ #v("worked.exp.sum_listed|d2") $=$ #v("worked.orders.listed.2|d2"). Whatever weight the model puts on any other token is dropped. These scores are not calibrated fraud probabilities, which is why the rule refits them on dev.

On IEEE-CIS we used d1-Omni-2B #cite-k("d1omni"), which reads the three option scores divided by a fixed temperature of #v("manual.training_loss.d1_temperature"). That temperature ships with the model in its own calibration file; we did not fit it, and the vendor's method is not documented to us. It is a single-parameter rescaling of the kind used for calibration #cite-k("guo").

== Prompts
On IEEE-CIS the question and options are the same as on PaySim; the prompt opens with a short hand-written context listing patterns seen in the fit rows, then states the payment's own fields: amount, product, hour, card network and type, the two email domains, device type, two match codes, four counters and one time gap. The richer prompt adds six more counters, five more time gaps, a distance, and what earlier payments say about the same card and address (count, time since the last, payments in the last day and week, amount against the card's usual, other cards at the address), never their labels. The tree-score setup adds one line with the GBDT's risk score, marked uncalibrated. Appendix H gives all of them verbatim, filled for one row.

== Training with LoRA
LoRA #cite-k("lora") freezes the model and trains two thin matrices beside each adapted weight, $W' = W + B A$, of rank #v("manual.recipe.lora_r"). Each example is one payment's question; the target is the letter of approve for a genuine payment and of decline for a fraud; review is never a target. Options appeared in a seeded random order during training. On IEEE-CIS the adapter saw #v("manual.training.fit_rows|int") fit rows: every fit fraud (#v("manual.training.fit_frauds|int")) and #v("manual.training.fit_legit|int") genuine payments, in #v("manual.recipe.batches|int") batches of #v("manual.recipe.batch") at learning rate #v("manual.recipe.lr"). The loss is a cross-entropy. On PaySim it ran over the whole vocabulary at the answer position; on IEEE-CIS over the three option scores only, divided by the temperature, because that is exactly the function the decision reads: our reader for d1-Omni-2B produces only those option scores, so there is no full-vocabulary softmax to train. Each setup has one adapter from one training run.

== Why two models
PaySim used Qwen2.5-1.5B-Instruct #cite-k("qwen"). The project then tried a second model family at the owner's request; in frozen checks on PaySim dev rows, d1-Omni-2B ranked fraud better than frozen Qwen and ran faster than the other candidate, so every IEEE-CIS experiment used it.

== Setups
The final test registered four setups, each with the cost rule: LR (the baseline); the language model reading the tree score (LM + tree); the same with the customer line; and the language model alone (LM alone), with the richer prompt and no tree score. Approving every payment is a floor. Two read-outs were added afterwards: LM alone with the customer line, and GBDT alone (Section V-D). $lambda$ for the language-model setups and for GBDT was picked on more dev windows than LR's (Section VII-C).

#fig-pipeline
#table-worked

= Results

== Pipeline development on PaySim
On simulated PaySim payments the untrained Qwen model was useless as a decider, and training fixed that: after LoRA training its mean cost per window was #v("manual.paysim.mean_trained|int") USD against #v("manual.paysim.mean_rules|int") for hand-written if-then rules on type, amount and receiver, and it chose the same action as the rules on #v("manual.paysim.matched|int") of #v("manual.paysim.decisions|int") payments. It learned the rule book; it never beat it. PaySim also showed option-order bias plainly: on the worked payment of @pipeline the untrained model's top answer flipped from #v("worked.top.listed") to #v("worked.top.reversed") when the options were reversed, both times the last letter (@worked). PaySim is simulated and nearly separable by rule, so it shows that the pipeline and the training work, not that a language model is worth using.

== The decision rule matters more than the model
On #v("rule.phase14.windows_n") earlier test windows, on the same LR scores, the cost rule cost #v("rule.phase14.diff.mean|signed") USD per window against two fixed cut-offs (#v("rule.phase14.diff.ci|ci")), distinguishable; its price was more good customers declined, #v("manual.phase14.good_declined_rule|int") against #v("manual.phase14.good_declined_lines|int"). No change of model came close: on the final test every model differs from LR by less, and none distinguishably (@finaltable).

The customer line is a cheap knob that moves the operating point, not a gain. @tradeoff sweeps its price from #v("rev.custline.logreg.points.0.price") to #v("rev.custline.logreg.points.12.price") USD on dev for LR and LM + tree: good customers declined fall steeply, from #v("rev.custline.logreg.points.0.declined|int") to #v("rev.custline.logreg.points.2.declined|int") for LR at #v("rev.custline.logreg.points.2.price") USD, while frauds stopped fall too and cost is flat or rises (dev). On the final test (held-out), the registered customer line cut good customers declined from #v("final.setups.signal_amt.good_declined|int") to #v("final.setups.signal_cust.good_declined|int") for LM + tree at #v("final.diffs.signal_cust_signal_amt.mean|signed") USD per window (#v("final.diffs.signal_cust_signal_amt.ci|ci")), #v("final.diffs.signal_cust_signal_amt|dist").

#fig-tradeoff

== Sensitivity to the assumed prices
Every dollar rests on assumed prices. We varied the friction rate, the review fee and a charged price per good customer declined over #v("rev.grid.n") combinations, re-picked $lambda$ on dev for each, and applied it to the saved final-test scores (read-out; Appendix C). GBDT alone was the cheapest setup in #v("rev.grid.gbdt_cheapest") of #v("rev.grid.n") combinations and LM + tree in #v("rev.grid.lmtree_cheapest")\; the cheapest setup was distinguishable from LR in #v("rev.grid.dist_n"), each time GBDT. In the cells where LM + tree was cheapest it was not distinguishable from LR. Which setup wins depends on the prices; that no language setup showed a distinguishable saving over LR did not change where it was tested.

== The locked final test
@finaltable and @final give the registered result. The primary comparison, LM + tree against LR, came out at #v("final.diffs.signal_amt_baseline.mean|signed") USD per window (#v("final.diffs.signal_amt_baseline.ci|ci")): #v("final.diffs.signal_amt_baseline|dist"). LM alone cost #v("final.diffs.rich_amt_baseline.mean|signed") (#v("final.diffs.rich_amt_baseline.ci|ci")). No setup showed a distinguishable saving against LR. That is no saving detectable at this sample size, not evidence of none: from the paired spread of the #v("final.windows_n") windows, effects below about #v("rev.cmp.p23_primary.mde|int") USD per window (LM + tree against LR) and #v("rev.cmp.p23_third.mde|int") (LM alone against LR) cannot be ruled out (read-out). Every block-bootstrap and BCa interval agrees with the registered verdicts; for the trees' read-out against LR, blocks of four give #v("rev.cmp.ro_secondary.block4|ci"), borderline at the Monte Carlo margin (Appendix A). Two comparisons are distinguishable: against approving everything, LM + tree saved #v("final.diffs.signal_amt_approve.mean|abs") USD per window (#v("final.diffs.signal_amt_approve.ci|ci")); and LM alone cost #v("final.diffs.rich_amt_signal_amt.mean|abs") more than the same model reading the tree score (#v("final.diffs.rich_amt_signal_amt.ci|ci")): the tree score is what helps.

*No clean test data for GBDT.* The final test did not register GBDT alone, and the audit (Section III-B) left no unscored window for a new held-out test. We therefore scored the saved GBDT once on the same #v("final.windows_n") windows, with its dev-fitted rule fixed in advance, as a read-out. It cost #v("rev.gbdt.mean|int") USD per window, the lowest of any setup: #v("rev.cmp.ro_secondary.mean|signed") against LR (#v("rev.cmp.ro_secondary.ci|ci")), #v("rev.cmp.ro_secondary|dist"), with its $lambda$ picked on more dev windows than LR's, which flatters it; and LM + tree cost #v("rev.cmp.ro_primary.mean|signed") against it (#v("rev.cmp.ro_primary.ci|ci")), #v("rev.cmp.ro_primary|dist"). LM alone was distinguishably dearer than GBDT, #v("rev.cmp.ro_rich_vs_gbdt.mean|signed") (#v("rev.cmp.ro_rich_vs_gbdt.ci|ci")). These windows were read before, so this is not a fresh test, and the MDE of the LM + tree comparison is about #v("rev.cmp.ro_primary.mde|int") USD per window.

#fig-final
#table-final
#table-sources

== Where the difference comes from
@table-sources gathers the dev evidence. *Information, not modelling (dev).* LR and GBDT saw every column but no history; the language model saw a few columns plus the card history. We refitted both baselines on exactly the fields of the richer prompt and its history (Appendix E). Cut down this way they lost #v("rev.parity.cmp.par-logreg__tab-logreg.mean|signed") (LR) and #v("rev.parity.cmp.par-trees__tab-all.mean|signed") (GBDT) USD per window, both distinguishable: most of the gap between LM alone, at #v("rev.parity.setups.lm-rich.mean|int"), and full GBDT, at #v("rev.parity.setups.tab-all.mean|int"). At equal information, LM alone was cheaper than LR, #v("rev.parity.cmp.lm-rich__par-logreg.mean|signed") (#v("rev.parity.cmp.lm-rich__par-logreg.ci|ci")), and was not distinguishable from GBDT, #v("rev.parity.cmp.lm-rich__par-trees.mean|signed") (#v("rev.parity.cmp.lm-rich__par-trees.ci|ci")). This favours the language model as a modeller; its deficit on the final test is mostly what its prompt leaves out.

*Ranking against cost (read-out).* LM alone ranked about as well as LR by ROC-AUC, #v("rev.ranking.lm.roc_auc|auc") (#v("rev.ranking.lm.roc_auc_ci|aucrange")) against #v("rev.ranking.lr.roc_auc|auc") (#v("rev.ranking.lr.roc_auc_ci|aucrange")), yet cost more. Its precision-recall area was lower, #v("rev.ranking.lm.pr_auc|auc") against #v("rev.ranking.lr.pr_auc|auc"): it ranks the top of the list worse, where the review budget and the declines are spent. It let more fraud through, #v("rev.by_action.lm.fraud_let_through|int") against #v("rev.by_action.lr.fraud_let_through|int") USD per window, and paid more in friction and reviews. The two largest frauds were reviewed by both setups, so removing them leaves the gap unchanged at #v("rev.drop_amount.2.mean|signed")\; in a further post-hoc check, not registered, removing the two frauds LM alone priced worst still leaves #v("rev.drop_gap.2.mean|signed") (#v("rev.drop_gap.2.ci|ci")). The gap is spread over many payments (Appendix B).

*Columns of uncertain timing (dev).* We could not establish that the C counters are known when a payment is decided #cite-k("kaufman"). Without any C column, with one retrain of LM alone, every setup got dearer on dev: LR #v("rev.noc.cmp.noc-logreg__tab-logreg.mean|signed"), GBDT #v("rev.noc.cmp.noc-trees__tab-all.mean|signed"), LM alone #v("rev.noc.cmp.lm-rich-noc__lm-rich.mean|signed"), all distinguishable (Appendix F). The ranking changed: with C columns LM alone and LR tie; without them LM alone is dearer than LR, #v("rev.noc.cmp.lm-rich-noc__noc-logreg.mean|signed") (#v("rev.noc.cmp.lm-rich-noc__noc-logreg.ci|ci")), just above the noise floor. GBDT stays cheapest. The language model leans on these columns most.

*Option order (dev).* The final setups were scored with options A decline, B review, C approve. Scored again in the forward order, LM alone changed its top answer on #v("rev.order.lm-rich.flip_pct")% of dev payments and LM + tree on #v("rev.order.lm-signal.flip_pct")%; the forward order cost LM alone #v("rev.order.lm-rich.fwd_rev.mean|signed") USD per window (#v("rev.order.lm-rich.fwd_rev.ci|ci")), a small but distinguishable bias, and averaging both orders moved it by #v("rev.order.lm-rich.avg_rev.mean|signed") (#v("rev.order.lm-rich.avg_rev.ci|ci")), #v("rev.order.lm-rich.avg_rev|dist") (Appendix G). The order used was not the flattering one.

== Accuracy against cost; dev against held-out
Accuracy and cost disagree: approving everything is #v("final.approve_all_accuracy|pct") accurate and costs #v("final.setups.approve_all.mean|int") USD per window, against #v("final.setups.baseline.mean|int") for LR with the rule. Dev and held-out disagree too: on dev, GBDT looked clearly cheaper than LR, #v("rule.phase20.dev.mean|signed") (#v("rule.phase20.dev.ci|ci")); on #v("rule.phase20.windows_n") earlier test windows the same model and rule gave #v("rule.phase20.diff.mean|signed") (#v("rule.phase20.diff.ci|ci")), #v("rule.phase20.diff|dist")\; and LM alone's dev tie with LR, #v("dev.diffs.rich_logreg.mean|signed"), became #v("final.diffs.rich_amt_baseline.mean|signed") on the final test. Dev rows sit close in time to the fit rows, and $lambda$ was picked on them; only the locked test can say what holds.

= Discussion

== Why the rule moved more than the model
The setups differ mostly in how they rank payments, and their ranking differs little: ROC-AUC from #v("rev.ranking.lr.roc_auc|auc") to #v("rev.ranking.lmtree.roc_auc|auc") on the final test (read-out). The rule changes something else: which action each ranked payment receives, weighed by its amount. We infer the mechanism: a ranking better by a few hundredths of AUC moves a few payments across a line, while pricing each action moves every large payment to the cheaper action at once. The rule's saving over fixed cut-offs, #v("rule.phase14.diff.mean|abs") USD per window on earlier test windows, was larger than any model's difference from LR on the final test, at most #v("final.diffs.rich_amt_baseline.mean|abs") (held-out); the two come from different windows.

== Reading a null result
No setup was distinguishable from LR, but the test was small. The held-out interval for LM + tree against LR rules out a saving larger than #v("final.diffs.signal_amt_baseline.ci.0|abs") USD per window and a loss larger than #v("final.diffs.signal_amt_baseline.ci.1|abs"), and smaller effects of either sign cannot be seen. A decision worth a few hundred dollars per #v("manual.data.window_rows|int") payments would need far more windows than were available.

== What the parity result means
At equal information the fine-tuned model was not distinguishable from boosted trees on dev (Section V-E), which is not evidence that they are equal. Its handicap is the prompt: a text budget that holds a few dozen fields where the tabular models read hundreds of columns. Giving the model the tree score closed most of the gap, but that setup then follows the score it reads (Section VII-F). For a team that already runs trees, the language model adds cost and latency without a detectable saving.

== What the protocol bought
Locking the test mattered. Dev results flipped twice on new windows (Section V-F), and the audit showed that every test window had been used by some evaluation, so a later question can only be answered by a read-out or by new data. Labelling each number as dev, held-out or read-out keeps those apart.

= Limitations

== Assumed prices
Every dollar rests on assumed prices. The price grid (Section V-C) shows that which setup is cheapest moves with them; with a charged price per good customer declined, the cost rule declines fewer customers, and on earlier test windows its saving over fixed cut-offs stopped being distinguishable from #v("manual.stress.not_distinguishable_from") USD per customer.

== Generalisability
One merchant's payments, one period, one region; no second real dataset. Nothing here says how the comparison would come out elsewhere.

== Optimism from choosing on dev
Cut-offs, $lambda$, prompts and the setups that went forward were chosen on dev, so dev results flatter (Section V-F). $lambda$ for the language-model setups and GBDT came from all #v("rev.protocol.dev_windows_all") dev windows, LR's from #v("rev.protocol.dev_windows_30")\; this flatters the others against LR. Every dev comparison involving a language-model setup is flattered for the same reason.

== The hand-written context block
The prompt's opening context, patterns observed in the fit rows, was written by hand. It is an untested human feature-engineering step: no setup was run without it.

== Columns of uncertain timing
If the C counters are not known at decision time, the results here overstate every setup, and the language model most (Section V-E). The no-C check covers LR, GBDT and LM alone on dev only; LM + tree, LR with two cut-offs and LR with the customer line were not run without C.

== The tree-score setup follows its score
LM + tree ranked payments almost exactly as the tree score it reads (rank correlation #v("final.rank_corr|d2") on the final-test rows) and on dev did no better than the score alone. Its result is mostly the trees'.

== Noise, power and spent data
Each setup was trained once, and two seeds differ by about #v("rev.seed.floor|int") USD per window (dev). With #v("final.windows_n") windows the intervals are wide and single large frauds weigh heavily. The test part is spent: the GBDT result is a read-out on reused windows, and answering whether any setup saves money needs new data.

== Disparity on proxies
We checked, as a read-out, the rate of good customers declined and of frauds missed by card type, device type and email domain (Appendix D). This is a disparity check on proxies, not a fairness audit. Groups differ far more than setups: frauds missed reach about half on debit cards and on payments with no device recorded, for every setup; the customer line lowers good customers declined in every group.

== PaySim and the classification head
PaySim is simulated; the first cycle shows only that the pipeline works. We did not compare the verbalizer with a classification head trained on the same model.

= Future Work
Compute and running cost per decision, which this evaluation did not price: the language model alone took #v("final.latency.p90|int") ms at the 90th percentile per payment on the final test (held-out), on one desktop graphics card, against far less for LR. An ablation against a classification head. New real datasets and a fresh, registered test. Measured, not assumed, prices.

= Conclusion
Read as a classifier, one lettered question and one forward pass, a small language model can be trained to decide payments. What changed outcomes most was the decision rule: on earlier test windows, pricing each action for each payment cost far less than fixed cut-offs, and a price for a turned-away customer moved the operating point, a trade-off of fewer good customers declined for fewer frauds stopped. On the locked, registered final test the fine-tuned model showed no detectable saving over logistic regression. Boosted trees, read out afterwards on the same windows, were cheapest without being distinguishably so. On dev, at equal information, the language model was not distinguishable from boosted trees; its deficit is mostly information. For a team that already runs a tabular model, the cheaper lesson is the decision rule.

#heading(numbering: none)[Acknowledgement]
This work began as a course project at Siam University.

#heading(numbering: none)[References]

#[
#set text(8pt)
#set par(first-line-indent: 0pt, justify: true, spacing: 0.45em)
// references: start
#bibk("nilson")[The Nilson Report, “Card fraud losses worldwide in 2023,” no. 1276, Dec. 2024.]
#bibk("robinson")[J. Robinson and D. Wingate, “Leveraging large language models for multiple choice question answering,” in _Proc. ICLR_, 2023.]
#bibk("pet")[T. Schick and H. Schütze, “Exploiting cloze questions for few shot text classification and natural language inference,” in _Proc. EACL_, 2021, pp. 255–269.]
#bibk("dalpozzolo")[A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, “Credit card fraud detection: A realistic modeling and a novel learning strategy,” _IEEE Trans. Neural Netw. Learn. Syst._, vol. 29, no. 8, pp. 3784–3797, 2018.]
#bibk("tabllm")[S. Hegselmann, A. Buendia, H. Lang, M. Agrawal, X. Jiang, and D. Sontag, “TabLLM: Few-shot classification of tabular data with large language models,” in _Proc. AISTATS_, PMLR 206, 2023, pp. 5549–5581.]
#bibk("tan")[X. Tan, Y. Ma, and X. Zhang, “Understanding structured financial data with LLMs: A case study on fraud detection,” in _Proc. ACL_, 2026, pp. 23375–23390.]
#bibk("lift")[T. Dinh, Y. Zeng, R. Zhang, Z. Lin, M. Gira, S. Rajput, J. Sohn, D. Papailiopoulos, and K. Lee, “LIFT: Language-interfaced fine-tuning for non-language machine learning tasks,” in _Proc. NeurIPS_, 2022.]
#bibk("fang")[X. Fang, W. Xu, F. A. Tan, Z. Hu, J. Zhang, Y. Qi, S. H. Sengamedu, and C. Faloutsos, “Large language models (LLMs) on tabular data: Prediction, generation, and understanding — A survey,” _Trans. Mach. Learn. Res._, 2024.]
#bibk("tabpfn")[N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, “TabPFN: A transformer that solves small tabular classification problems in a second,” in _Proc. ICLR_, 2023.]
#bibk("lmbff")[T. Gao, A. Fisch, and D. Chen, “Making pre-trained language models better few-shot learners,” in _Proc. ACL-IJCNLP_, 2021, pp. 3816–3830.]
#bibk("pezeshkpour")[P. Pezeshkpour and E. Hruschka, “Large language models sensitivity to the order of options in multiple-choice questions,” in _Findings of NAACL_, 2024, pp. 2006–2017.]
#bibk("zheng")[C. Zheng, H. Zhou, F. Meng, J. Zhou, and M. Huang, “Large language models are not robust multiple choice selectors,” in _Proc. ICLR_, 2024.]
#bibk("elkan")[C. Elkan, “The foundations of cost-sensitive learning,” in _Proc. IJCAI_, 2001, pp. 973–977.]
#bibk("bahnsen")[A. Correa Bahnsen, A. Stojanovic, D. Aouada, and B. Ottersten, “Cost sensitive credit card fraud detection using Bayes minimum risk,” in _Proc. ICMLA_, 2013, pp. 333–338.]
#bibk("whitrow")[C. Whitrow, D. J. Hand, P. Juszczak, D. Weston, and N. M. Adams, “Transaction aggregation as a strategy for credit card fraud detection,” _Data Min. Knowl. Discov._, vol. 18, no. 1, pp. 30–55, 2009.]
#bibk("grinsztajn")[L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?,” in _Proc. NeurIPS Datasets and Benchmarks_, 2022.]
#bibk("friedman")[J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” _Ann. Statist._, vol. 29, no. 5, pp. 1189–1232, 2001.]
#bibk("ieeecis")[IEEE Computational Intelligence Society and Vesta Corporation, “IEEE-CIS Fraud Detection,” Kaggle competition, 2019.]
#bibk("paysim")[E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, “PaySim: A financial mobile money simulator for fraud detection,” in _Proc. EMSS_, 2016.]
#bibk("crisp")[R. Wirth and J. Hipp, “CRISP-DM: Towards a standard process model for data mining,” in _Proc. 4th Int. Conf. Practical Applications of Knowledge Discovery and Data Mining_, 2000, pp. 29–39.]
#bibk("nosek")[B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, “The preregistration revolution,” _Proc. Natl. Acad. Sci. USA_, vol. 115, no. 11, pp. 2600–2606, 2018.]
#bibk("dror")[R. Dror, G. Baumer, S. Shlomov, and R. Reichart, “The hitchhiker's guide to testing statistical significance in natural language processing,” in _Proc. ACL_, 2018, pp. 1383–1392.]
#bibk("kunsch")[H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” _Ann. Statist._, vol. 17, no. 3, pp. 1217–1241, 1989.]
#bibk("efron")[B. Efron, “Better bootstrap confidence intervals,” _J. Amer. Statist. Assoc._, vol. 82, no. 397, pp. 171–185, 1987.]
#bibk("platt")[J. C. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in _Advances in Large Margin Classifiers_, A. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA, USA: MIT Press, 2000, pp. 61–74.]
#bibk("d1omni")[Liquid AI, “d1-Omni-2B,” Hugging Face model card, LiquidAI/d1-Omni-2B, 2026.]
#bibk("guo")[C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in _Proc. ICML_, PMLR 70, 2017, pp. 1321–1330.]
#bibk("lora")[E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in _Proc. ICLR_, 2022.]
#bibk("qwen")[Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.]
#bibk("kaufman")[S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” _ACM Trans. Knowl. Discov. Data_, vol. 6, no. 4, pp. 1–21, 2012.]
// references: end
]

// ---- appendices -------------------------------------------------------------------------------------------

#set page(columns: 1)
#let appendix(t) = heading(numbering: none, t)
#appendix[Appendices]

The analyses here were registered in their scripts before they ran; only the percentile intervals in Appendix A belong to the locked test. Each number is labelled dev or read-out.

#appendix[Appendix A. Bootstrap robustness]
#table-robust

#appendix[Appendix B. Ranking and the AUC–cost gap]
#table-ranking
#ranking-drop
#fig-reliability

#appendix[Appendix C. Price grid]
#fig-grid

#appendix[Appendix D. Disparity check on proxies]
#table-disparity

#appendix[Appendix E. Feature parity]
#table-parity

#appendix[Appendix F. Without the C columns]
#table-noc

#appendix[Appendix G. Option-order bias on IEEE-CIS]
#table-order

#appendix[Appendix H. Prompts verbatim]
#prompts
