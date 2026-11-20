"""Write the generated tables and figures into paper/main.typ, between "// generated: start" and "// generated: end".

    python paper/gen_tables.py && python paper/build_data.py

Each cell is a literal v("rev...") key, so paper/build_data.py finds every number and data.json stays the only source; this
script only writes Typst layout, never a number. Run it when the layout of a generated block changes, then rebuild data.json.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.typ"
SETUPS = (("lr", "LR"), ("lmtree", "LM + tree"), ("lmtree_cust", "LM + tree, cust."), ("lm", "LM alone"), ("gbdt", "GBDT\\*"))
CMP = (("p23_primary", "LM + tree − LR"), ("p23_third", "LM alone − LR"), ("p23_second", "cust. line − LM + tree"), ("p23_cust_vs_lr", "LM + tree, cust. − LR"),
       ("p23_rich_vs_signal", "LM alone − LM + tree"), ("p23_signal_vs_all", "LM + tree − approve all"), ("ro_primary", "LM + tree − GBDT\\*"),
       ("ro_secondary", "GBDT − LR\\*"), ("ro_rich_vs_gbdt", "LM alone − GBDT\\*"))
DISP = (("card6", (("credit", "credit card"), ("debit", "debit card"))),
        ("DeviceType", (("desktop", "desktop"), ("mobile", "mobile"), ("none", "no device recorded"))),
        ("P_emaildomain", (("gmail_com", "gmail.com"), ("yahoo_com", "yahoo.com"), ("hotmail_com", "hotmail.com"), ("anonymous_com", "anonymous.com"),
                           ("aol_com", "aol.com"), ("other", "other domains"), ("none", "no email"))))


def v(key: str) -> str:
    return f'#v("{key}")'


def small_table(cols: str, header: list[str], rows: list[list[str]], caption: str, label: str, size: str = "7pt", wide: bool = False) -> str:
    h = ", ".join(f"[{c}]" for c in header)
    body = "\n".join("    " + ", ".join(f"[{c}]" for c in r) + "," for r in rows)
    return f"""#let {label} = [#figure(
  placement: none,
  {{
  set text({size})
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: {cols},
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    table.hline(stroke: 0.6pt),
    table.header({h}),
    table.hline(stroke: 0.4pt),
{body}
    table.hline(stroke: 0.6pt),
  )
  }},
  caption: [{caption}],
) <{label}>]
"""


def robust() -> str:
    rows = [[n, v(f"rev.cmp.{k}.label"), v(f"rev.cmp.{k}.mean|signed"), v(f"rev.cmp.{k}.ci|ci"), v(f"rev.cmp.{k}.block2|ci"), v(f"rev.cmp.{k}.block4|ci"),
             v(f"rev.cmp.{k}.bca|ci"), v(f"rev.cmp.{k}.sd|int"), v(f"rev.cmp.{k}.mde|int")] for k, n in CMP]
    return small_table("(auto, auto, auto, auto, auto, auto, auto, auto, auto)", ["a − b", "label", "mean", "percentile (registered)", "block 2", "block 4", "BCa", "paired SD", "MDE"], rows,
                       "Every final-test comparison, USD per window. The registered percentile interval is the headline; the block bootstraps (blocks of 2 and 4 windows in time order) and BCa intervals, the paired SD and the MDE are read-outs. The block-4 interval of GBDT − LR excludes zero only by a margin within Monte Carlo error, so it is borderline. \\*GBDT is a read-out on the same windows.",
                       "table-robust", "6.5pt", True)


def ranking() -> str:
    rows = [[n, v(f"rev.ranking.{k}.roc_auc|auc"), v(f"rev.ranking.{k}.roc_auc_ci|aucrange"), v(f"rev.ranking.{k}.pr_auc|auc"), v(f"rev.ranking.{k}.pr_auc_ci|aucrange"),
             v(f"rev.by_action.{k}.fraud_let_through|int"), v(f"rev.by_action.{k}.good_declined|int"), v(f"rev.by_action.{k}.review_fees|int")] for k, n in SETUPS]
    t = small_table("(auto, auto, auto, auto, auto, auto, auto, auto)", ["setup", "ROC-AUC", "95% CI", "PR-AUC", "95% CI", "fraud let through", "good declined", "review fees"], rows,
                    "Ranking and where the money goes on the final test (read-out). Intervals: window bootstrap, #v(\"manual.recipe.boot_reps|int\") resamples. Costs: USD per window by part of Eq. 1. \\*Read-out setup.",
                    "table-ranking", "6.5pt", True)
    drop = f"""#let ranking-drop = block(inset: (y: 3pt))[#set text(8pt)
#set par(first-line-indent: 0pt)
LM alone − LR on the final test is {v("rev.drop_amount.0.mean|signed")} USD per window ({v("rev.drop_amount.0.ci|ci")}). Removing the largest 1 and 2 frauds by amount ({v("rev.drop_amount.2.amounts.0|usd2")} and {v("rev.drop_amount.2.amounts.1|usd2")} USD), which both setups reviewed, leaves {v("rev.drop_amount.1.mean|signed")} and {v("rev.drop_amount.2.mean|signed")}. Post-hoc and not registered: removing the 1 and 2 frauds with the largest cost gap ({v("rev.drop_gap.2.amounts.0|usd2")} and {v("rev.drop_gap.2.amounts.1|usd2")} USD, approved by LM alone, reviewed by LR) leaves {v("rev.drop_gap.1.mean|signed")} ({v("rev.drop_gap.1.ci|ci")}) and {v("rev.drop_gap.2.mean|signed")} ({v("rev.drop_gap.2.ci|ci")}).]
"""
    return t + "\n" + drop


def reliability() -> str:
    panels = []
    for k, n in SETUPS:
        pts = ", ".join(f'(x: float({v(f"rev.reliability.{k}.{i}.mean_p")[1:]}), y: float({v(f"rev.reliability.{k}.{i}.fraud_rate")[1:]}))' for i in range(10))
        panels.append(f"rel-panel([{n}], ({pts}))")
    return f"""#let rel-panel(title, pts) = layout(size => {{
  let w = size.width
  let s = w - 14pt
  let top = calc.max(0.6, ..pts.map(p => calc.max(p.x, p.y)))
  let X(x) = 12pt + calc.sqrt(x / top) * s
  let Y(y) = s - calc.sqrt(y / top) * s
  block(width: w, height: s + 18pt, {{
    place(dx: 12pt, dy: 0pt, rect(width: s, height: s, stroke: 0.4pt + luma(150)))
    place(line(start: (X(0.0), Y(0.0)), end: (X(top), Y(top)), stroke: (paint: luma(170), thickness: 0.5pt, dash: "dashed")))
    for p in pts {{ place(dx: X(p.x) - 1.6pt, dy: Y(p.y) - 1.6pt, circle(radius: 1.6pt, fill: black)) }}
    place(dx: 12pt, dy: s + 2pt, box(width: s, align(center, text(6pt, title))))
  }})
}})
#let fig-reliability = [#figure(
  placement: none,
  grid(columns: (1fr,) * 5, column-gutter: 10pt, {", ".join(panels)}),
  caption: [Reliability of the rule's $p$ (Platt-scaled score) on the final test, per setup (read-out): ten equal-count bins, mean $p$ (across) against the observed fraud rate (up), both on a square-root scale; dashed: perfect calibration. \\*GBDT is a read-out setup.],
) <fig-reliability>]
"""


def grid() -> str:
    cells = []
    for r in range(3):
        row = []
        for cr in range(3):
            for cu in range(3):
                i = r * 9 + cr * 3 + cu
                row.append(f'gcell(v("rev.grid.cells.{i}.cheapest"), v("rev.grid.cells.{i}.vs_lr|signed"), v("rev.grid.cells.{i}.dist"))')
        cells.append(f'[#v("rev.grid.cells.{r * 9}.r|pct")], ' + ", ".join(row))
    heads = []
    for cr in range(3):
        for cu in range(3):
            heads.append(f'[#v("rev.grid.cells.{cr * 3 + cu}.c_r") / #v("rev.grid.cells.{cr * 3 + cu}.customer")]')
    return f"""#let gcell(who, d, dist) = table.cell(fill: if who == "gbdt" {{ luma(225) }} else if who == "lmtree" {{ rgb("#cfe0f3") }} else {{ white }},
  align(center, {{ set text(6pt); [#(if who == "gbdt" {{ "GBDT" }} else if who == "lmtree" {{ "LM+tree" }} else {{ who }})] ; linebreak(); if dist == "yes" {{ strong(d) }} else {{ d }} }}))
#let fig-grid = [#figure(
  placement: none,
  {{
  set par(justify: false, first-line-indent: 0pt)
  table(
    columns: (auto,) + (1fr,) * 9,
    stroke: 0.3pt + white,
    inset: 3pt,
    table.header(table.cell(rowspan: 1)[#text(6pt)[friction \\ rate]], {", ".join(f"[#text(6pt)[{h[1:-1]}]]" for h in heads)}),
    {", ".join(cells)},
  )
  }},
  caption: [Price grid on the final test (read-out). Rows: friction rate $r$; columns: review fee $c_r$ / charged price per good customer declined, in USD. Each cell names the cheapest of LR, LM + tree, LM alone and GBDT alone, with $lambda$ re-picked on dev under that cell's prices, and its mean difference from LR in USD per window; bold: distinguishable from LR. The base cell ({v("costs.friction_rate|pct")}, {v("costs.review_fee|num")} / 0) reproduces the saved costs exactly.],
) <fig-grid>]
"""


def tradeoff() -> str:
    series = []
    for k, name, color, dash, square, shift in (("logreg", "LR", "luma(60)", "solid", "false", "3pt"), ("lm-signal", "LM + tree", 'rgb("#2f6db5")', "dashed", "true", "-9pt")):
        pts = []
        for i in range(13):
            note = f'[{v(f"rev.custline.{k}.points.{i}.price")} USD: {v(f"rev.custline.{k}.points.{i}.cost|int")}]' if i in (0, 2) or (i == 12 and k == "logreg") else "none"
            pts.append(f'(x: float({v(f"rev.custline.{k}.points.{i}.declined")[1:]}), y: float({v(f"rev.custline.{k}.points.{i}.stopped")[1:]}), note: {note})')
        series.append(f"(name: [{name}], color: {color}, dash: \"{dash}\", square: {square}, shift: {shift}, pts: ({', '.join(pts)}))")
    return f"""#let fig-tradeoff = [#figure(
  placement: top,
  scatter(({", ".join(series)}), [good customers declined (#v("manual.tabular_dev.chains") dev windows)], [frauds stopped (of #v("rev.custline.logreg.frauds|int"))]),
  caption: [The customer line moves the operating point (dev). Each point is a price per good customer turned away, from #v(\"rev.custline.logreg.points.0.price\") to #v(\"rev.custline.logreg.points.12.price\") USD in steps of #v(\"rev.custline.logreg.points.1.price\"), added to the rule's decision only; notes give the price and the mean cost per window. Grey circles: LR; blue squares: LM + tree. Fewer good customers declined comes with fewer frauds stopped and equal or higher cost.],
) <tradeoff>]
"""


def sources() -> str:
    rows = [["LM alone − LR, every column", v("dev.diffs.rich_logreg.mean|signed"), v("dev.diffs.rich_logreg.ci|ci"), v("dev.diffs.rich_logreg|dist")],
            ["LM alone − LR, at the prompt's fields", v("rev.parity.cmp.lm-rich__par-logreg.mean|signed"), v("rev.parity.cmp.lm-rich__par-logreg.ci|ci"), v("rev.parity.cmp.lm-rich__par-logreg|dist")],
            ["LM alone − GBDT, at the prompt's fields", v("rev.parity.cmp.lm-rich__par-trees.mean|signed"), v("rev.parity.cmp.lm-rich__par-trees.ci|ci"), v("rev.parity.cmp.lm-rich__par-trees|dist")],
            ["GBDT: prompt's fields − every column", v("rev.parity.cmp.par-trees__tab-all.mean|signed"), v("rev.parity.cmp.par-trees__tab-all.ci|ci"), v("rev.parity.cmp.par-trees__tab-all|dist")],
            ["LM alone − LR, both without C", v("rev.noc.cmp.lm-rich-noc__noc-logreg.mean|signed"), v("rev.noc.cmp.lm-rich-noc__noc-logreg.ci|ci"), v("rev.noc.cmp.lm-rich-noc__noc-logreg|dist")],
            ["LM alone: forward − scored option order", v("rev.order.lm-rich.fwd_rev.mean|signed"), v("rev.order.lm-rich.fwd_rev.ci|ci"), v("rev.order.lm-rich.fwd_rev|dist")],
            ["LM, second − first training seed", v("rev.seed.mean|signed"), v("rev.seed.ci|ci"), "not distinguishable"]]
    t = small_table("(1fr, auto, auto, auto)", ["dev comparison, USD per window", "mean", "95% interval", "verdict"], rows,
                    "Where the difference comes from (dev, " + v("manual.tabular_dev.chains") + " windows; every read involving a language-model setup is flattered by $lambda$ picked on these windows). The first row is the registered setups' dev comparison; the rest are the revision's dev experiments (Appendices E–G).",
                    "table-sources", "7pt")
    return t.replace("placement: none,", "placement: top,", 1)


def disparity() -> str:
    out = []
    for metric, title in (("gd", "good customers declined"), ("fm", "frauds missed")):
        rows = []
        for dim, groups in DISP:
            for g, label in groups:
                n = "n_good" if metric == "gd" else "n_fraud"
                rows.append([label, v(f"rev.disparity.{dim}.{g}.{n}|int")] + [f'{v(f"rev.disparity.{dim}.{g}.{k}.{metric}")} ({v(f"rev.disparity.{dim}.{g}.{k}.{metric}_ci")})' for k, _ in SETUPS])
        out.append(small_table("(auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr)", ["group", "n"] + [n for _, n in SETUPS], rows,
                               f"Share of {title}, % with 95% window-bootstrap interval, by card type, device type and purchaser email domain on the final test (read-out). A disparity check on proxies, not a fairness audit. \\*GBDT is a read-out setup.",
                               f"table-disp-{metric}", "6pt", True))
    return "\n".join(out) + "#let table-disparity = [#table-disp-gd #table-disp-fm]\n"


def parity() -> str:
    names = (("tab-all", "GBDT, all columns"), ("lm-signal", "LM + tree"), ("par-trees", "GBDT, prompt's fields"), ("lm-rich", "LM alone"), ("tab-logreg", "LR, all columns"), ("par-logreg", "LR, prompt's fields"))
    rows = [[n, v(f"rev.parity.setups.{k}.mean|int"), v(f"rev.parity.setups.{k}.auc|auc")] for k, n in names]
    cmps = (("lm-rich__par-logreg", "LM alone − LR, prompt's fields"), ("lm-rich__par-trees", "LM alone − GBDT, prompt's fields"), ("lm-signal__par-trees", "LM + tree − GBDT, prompt's fields"),
            ("par-logreg__tab-logreg", "LR: prompt's fields − all columns"), ("par-trees__tab-all", "GBDT: prompt's fields − all columns"))
    rows += [[n, v(f"rev.parity.cmp.{k}.mean|signed"), f'{v(f"rev.parity.cmp.{k}.ci|ci")}, {v(f"rev.parity.cmp.{k}|dist")}'] for k, n in cmps]
    return small_table("(1fr, auto, auto)", ["setup or difference", "USD / window", "AUC, or 95% interval"], rows,
                       "Feature parity on " + v("manual.tabular_dev.chains") + " dev windows (dev): LR and GBDT refitted on exactly the richer prompt's fields and its card and address history. $p$ on " + v("rev.protocol.dev_windows_30") + " dev windows and $lambda$ on all " + v("rev.protocol.dev_windows_all") + " for every row, so LR here differs slightly from the paper's baseline.",
                       "table-parity")


def noc() -> str:
    rows = [[n, v(f"rev.noc.setups.{a}.mean|int"), v(f"rev.noc.setups.{b}.mean|int"), f'{v(f"rev.noc.cmp.{b}__{a}.mean|signed")} ({v(f"rev.noc.cmp.{b}__{a}.ci|ci")})']
            for a, b, n in (("tab-all", "noc-trees", "GBDT"), ("lm-rich", "lm-rich-noc", "LM alone"), ("tab-logreg", "noc-logreg", "LR"))]
    rows += [[n, "", "", f'{v(f"rev.noc.cmp.{k}.mean|signed")} ({v(f"rev.noc.cmp.{k}.ci|ci")})'] for k, n in (("noc-trees__noc-logreg", "GBDT − LR, no C"), ("lm-rich-noc__noc-logreg", "LM alone − LR, no C"), ("lm-rich-noc__noc-trees", "LM alone − GBDT, no C"))]
    return small_table("(1fr, auto, auto, auto)", ["setup", "with C", "without C", "difference (95% interval)"], rows,
                       "Without any C column, " + v("manual.tabular_dev.chains") + " dev windows (dev): LR and GBDT refitted without C1–C14; one LoRA retrain of LM alone with no C column in its prompt or context, same rows and settings. Every setup gets distinguishably dearer, and the ranking of LM alone and LR changes. LM + tree was not run without C.",
                       "table-noc")


def order() -> str:
    rows = [[n, f'{v(f"rev.order.{k}.flip_pct")}%', v(f"rev.order.{k}.rev|int"), v(f"rev.order.{k}.fwd|int"), f'{v(f"rev.order.{k}.fwd_rev.mean|signed")} ({v(f"rev.order.{k}.fwd_rev.ci|ci")})']
            for k, n in (("lm-rich", "LM alone"), ("lm-signal", "LM + tree"))]
    t = small_table("(1fr, auto, auto, auto, auto)", ["setup", "top answer flips", "as scored", "forward", "forward − as scored"], rows,
                    "Option-order bias on " + v("manual.tabular_dev.chains") + " dev windows (dev). As scored: A decline, B review, C approve; forward: A approve, B review, C decline. Averaging both orders for LM alone, triggered by the registered rule, changed its cost by " + v("rev.order.lm-rich.avg_rev.mean|signed") + " (" + v("rev.order.lm-rich.avg_rev.ci|ci") + ").",
                    "table-order")
    return t


WORKED = """#let table-worked = [#figure(
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
"""

PROMPTS = """#let prompts = {
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
"""


def main() -> None:
    gen = "\n".join([sources(), robust(), ranking(), reliability(), grid(), tradeoff(), disparity(), parity(), noc(), order(), WORKED, PROMPTS])
    typ = MAIN.read_text(encoding="utf-8")
    new, n = re.subn(r"// generated: start\n.*?// generated: end\n", lambda m: "// generated: start\n" + gen + "// generated: end\n", typ, flags=re.S)
    assert n == 1, "markers not found"
    MAIN.write_text(new, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
