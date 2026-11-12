/* The fraud analyst console: replays recorded traces (fraud/runs/*.jsonl) and shows the eval.
   Replay only: nothing here runs a model. The replay clock is the sum of the recorded decision
   times, so at 1x each transaction appears when its decision finished in the recorded run. */
"use strict";

const app = document.getElementById("app");
const ACTIONS = ["approve", "review", "decline"];
const BY = {
  model: "the model", filter: "type filter (rule)", rules: "rules-only table",
  logreg: "logistic regression", random: "random", all: "approve-all", trees: "boosted trees", lm: "the language model, scored in a batch",
};
const SETUP_NAME = {
  hybrid: "hybrid (filter + model)", "hybrid-1.5b": "hybrid (filter + Qwen2.5-1.5B)", rules: "rules-only",
  "hybrid-ft": "hybrid (filter + fine-tuned 1.5B)", "hybrid-d3ft": "hybrid (filter + fine-tuned d3-nano)",
  "hybrid-d1ft": "hybrid (filter + fine-tuned d1-Omni-2B)",
  logreg: "filter + logistic regression", random: "filter + random", "approve-all": "approve-all",
  "ieee-approve-all": "approve-all", "ieee-logreg-all": "logistic regression, all columns, two lines", "ieee-logreg-all-amt": "logistic regression, all columns, amount-aware rule",
  "ieee-logreg-same": "logistic regression, the prompt's fields", "ieee-ft-ieee-full": "d1-Omni-2B fine-tuned on every fit fraud, two lines", "ieee-ft-full-amt": "d1-Omni-2B fine-tuned on every fit fraud, amount-aware rule",
  "ieee-blend-amt": "logistic regression and d1-Omni-2B blended, amount-aware rule", "ieee-ft-ieee-d1omni": "d1-Omni-2B fine-tuned on IEEE-CIS", "ieee-ft-d1omni": "d1-Omni-2B fine-tuned on PaySim", "ieee-frozen-qwen1.5b": "frozen Qwen2.5-1.5B",
  "rule-base": "logistic regression, Phase 14's rule", "rule-customers": "logistic regression, the rule with a customer line", "rule-paced": "logistic regression, the rule with a paced review price", "rule-combined": "logistic regression, the rule with three changes", "rule-longrun": "logistic regression, the rule with a long-run prior", "rule-lines": "logistic regression, two lines",
  "tab-logreg-amt": "logistic regression, all columns, amount-aware rule", "tab-logreg-lines": "logistic regression, all columns, two lines", "tab-all-amt": "boosted trees, all columns, amount-aware rule", "tab-all-lines": "boosted trees, all columns, two lines", "tab-all-cust": "boosted trees, all columns, the rule with a customer line",
  "tab-rich-amt": "boosted trees, extra features, amount-aware rule", "tab-rich-lines": "boosted trees, extra features, two lines", "tab-nocdv-amt": "boosted trees without C, D, V, amount-aware rule", "tab-nocdv-lines": "boosted trees without C, D, V, two lines", "tab-logreg-nocdv-amt": "logistic regression without C, D, V, amount-aware rule", "tab-logreg-nocdv-lines": "logistic regression without C, D, V, two lines",
  "signal-alone-amt": "the signal alone (trees at 400), amount-aware rule", "signal-alone-lines": "the signal alone (trees at 400), two lines",
  "lm-phase12-amt": "d1-Omni-2B fine-tuned, Phase 12's adapter, amount-aware rule", "lm-phase12-lines": "d1-Omni-2B fine-tuned, Phase 12's adapter, two lines", "lm-seed1-amt": "d1-Omni-2B fine-tuned, seed 1, amount-aware rule", "lm-seed1-lines": "d1-Omni-2B fine-tuned, seed 1, two lines",
  "lm-rich-amt": "d1-Omni-2B fine-tuned, more prompt columns, amount-aware rule", "lm-rich-lines": "d1-Omni-2B fine-tuned, more prompt columns, two lines", "lm-signal-amt": "d1-Omni-2B fine-tuned, reads a tabular score, amount-aware rule", "lm-signal-lines": "d1-Omni-2B fine-tuned, reads a tabular score, two lines",
  "lm-frozen-amt": "frozen d1-Omni-2B, amount-aware rule", "lm-frozen-lines": "frozen d1-Omni-2B, two lines",
  "lm-weighted-amt": "d1-Omni-2B trained on cost, more prompt columns, amount-aware rule", "lm-weighted-lines": "d1-Omni-2B trained on cost, more prompt columns, two lines", "lm-weighted-cust": "d1-Omni-2B trained on cost, more prompt columns, the rule with a customer line",
  "lm-rich-cust": "d1-Omni-2B fine-tuned, more prompt columns, the rule with a customer line", "lm-signal-cust": "d1-Omni-2B fine-tuned, reads a tabular score, the rule with a customer line",
  "lm-two-seeds-amt": "two seeds of d1-Omni-2B averaged, amount-aware rule", "lm-two-seeds-cust": "two seeds of d1-Omni-2B averaged, the rule with a customer line", "lm-three-amt": "three d1-Omni-2B adapters averaged, amount-aware rule", "lm-three-cust": "three d1-Omni-2B adapters averaged, the rule with a customer line",
  "lm-rich-signal-amt": "d1-Omni-2B, more columns and tabular score averaged, amount-aware rule", "lm-rich-signal-cust": "d1-Omni-2B, more columns and tabular score averaged, the rule with a customer line",
  "lm-rich+trees-amt": "d1-Omni-2B, more columns, fitted with the tabular score, amount-aware rule", "lm-rich+trees-cust": "d1-Omni-2B, more columns, fitted with the tabular score, the rule with a customer line", "lm-signal+trees-amt": "d1-Omni-2B reading the score, fitted with it again, amount-aware rule", "lm-signal+trees-cust": "d1-Omni-2B reading the score, fitted with it again, the rule with a customer line",
  "final-logreg-amt": "logistic regression, all columns, Phase 14's rule (the baseline)", "final-signal-amt": "d1-Omni-2B reading a tabular score, amount-aware rule", "final-signal-cust": "d1-Omni-2B reading a tabular score, the rule with a customer line", "final-rich-amt": "d1-Omni-2B with more prompt columns, no tabular score, amount-aware rule",
  "final-rich-cust": "d1-Omni-2B with more prompt columns, no tabular score, the rule with a customer line (read-out)",
  "ieee-logreg-all-dev30": "logistic regression, all columns, two lines re-picked on 30 dev windows", "ieee-ft-full-dev30": "d1-Omni-2B fine-tuned on every fit fraud, two lines re-picked on 30 dev windows",
};
// Setups named by a pattern rather than listed: Phase 17's column audit (what is left out of logreg) and Phase 19's feature sets.
const RULE_NAME = { amt: "amount-aware rule", lines: "two lines" };
const FEATURE_SET = { all: "all columns", rich: "all columns and extra features", uid: "extra features, interactions and a card id",
  nocdv: "without C, D, V", "nocdv-rich": "without C, D, V, with extra features" };
function setupName(s) {
  if (SETUP_NAME[s]) return SETUP_NAME[s];
  let m = /^audit-(all|no[A-Z]+)-(amt|lines)$/.exec(s);
  if (m) return `logistic regression, ${m[1] === "all" ? "all columns" : `without ${m[1].slice(2).split("").join(", ")}`}, ${RULE_NAME[m[2]]}`;
  m = /^feat-(.+)-(amt|lines)$/.exec(s);
  if (m && FEATURE_SET[m[1]]) return `logistic regression, ${FEATURE_SET[m[1]]}, ${RULE_NAME[m[2]]}`;
  return s;
}
// Short names for tables and charts. The final five match the deck (slides/), so one thing has one name. The full name
// (setupName) stays in a tooltip and in each page's legend table, with the setup id.
const SHORT = {
  "final-logreg-amt": "Baseline: logreg, rule", "final-signal-amt": "d1 reads trees, rule", "final-signal-cust": "d1 reads trees, customer line",
  "final-rich-amt": "d1, no trees, rule", "final-rich-cust": "d1, no trees, customer line", "ieee-approve-all": "Approve everything",
  "ieee-logreg-all": "logreg, two lines", "ieee-logreg-all-amt": "logreg, rule", "ieee-logreg-same": "logreg, prompt's fields",
  "ieee-ft-ieee-full": "d1 tuned on all fraud, two lines", "ieee-ft-full-amt": "d1 tuned on all fraud, rule", "ieee-blend-amt": "logreg and d1 blended, rule",
  "ieee-ft-ieee-d1omni": "d1 tuned on IEEE-CIS", "ieee-ft-d1omni": "d1 tuned on PaySim", "ieee-frozen-qwen1.5b": "Qwen 1.5B, frozen",
  "ieee-logreg-all-dev30": "logreg, two lines from 30 dev", "ieee-ft-full-dev30": "d1 tuned on all fraud, lines from 30 dev",
  "rule-base": "logreg, Phase 14's rule", "rule-customers": "rule with a customer line", "rule-paced": "rule, paced review price",
  "rule-combined": "rule, three changes", "rule-longrun": "rule, long-run prior", "rule-lines": "logreg, two lines",
};
// Most ids are a base plus -amt (the rule), -lines (two lines) or -cust (the customer line).
const SHORT_BASE = {
  "tab-logreg": "logreg", "tab-all": "trees", "tab-rich": "trees, extra features", "tab-nocdv": "trees, no C/D/V", "tab-logreg-nocdv": "logreg, no C/D/V",
  "signal-alone": "trees at 400 alone", "lm-phase12": "d1, Phase 12 adapter", "lm-seed1": "d1, seed 1", "lm-rich": "d1, no trees", "lm-signal": "d1 reads trees",
  "lm-frozen": "d1, frozen", "lm-weighted": "d1, no trees, trained on cost", "lm-two-seeds": "d1, two seeds averaged", "lm-three": "d1, three adapters averaged",
  "lm-rich-signal": "d1, both prompts averaged", "lm-rich+trees": "d1, no trees, fitted with trees", "lm-signal+trees": "d1 reads trees, fitted with trees",
  "feat-all": "logreg", "feat-rich": "logreg, extra features", "feat-uid": "logreg, card id", "feat-nocdv": "logreg, no C/D/V", "feat-nocdv-rich": "logreg, no C/D/V, extra features",
};
const SHORT_RULE = { amt: "rule", lines: "two lines", cust: "customer line" };
function shortName(s) {
  if (SHORT[s]) return SHORT[s];
  let m = /^audit-(all|no[A-Z]+)-(amt|lines)$/.exec(s);
  if (m) return `logreg, ${m[1] === "all" ? "all columns" : `no ${m[1].slice(2).split("").join("/")}`}, ${SHORT_RULE[m[2]]}`;
  m = /^(.+)-(amt|lines|cust)$/.exec(s);
  if (m && SHORT_BASE[m[1]]) return `${SHORT_BASE[m[1]]}, ${SHORT_RULE[m[2]]}`;
  return setupName(s);
}
function setupLegend(ids) {  // one table per page: the short name, the full name and the id
  return el("details", { class: "panel more", style: "margin-top:12px" }, el("summary", {}, "Setup names: short, full and id"),
    el("div", { class: "scroll-x" }, el("table", { class: "data" },
      el("thead", {}, el("tr", {}, ...["short name", "full name", "setup id"].map((t) => el("th", {}, t)))),
      el("tbody", {}, ...ids.map((s) => el("tr", {}, el("td", {}, shortName(s)), el("td", {}, setupName(s)), el("td", { class: "mono" }, s)))))));
}
// What each eval is, by the script that wrote it (fraud/ROADMAP.md). Dev evals never score a test row.
const EVAL_WHAT = {
  "fraud/eval.py": "Phase 3: frozen Qwen2.5-1.5B on PaySim, 20 test windows", "fraud/eval_ft.py": "Phase 5: fine-tuned Qwen2.5-1.5B on PaySim, 20 test windows",
  "fraud/eval_d3ft.py": "Phase 7: fine-tuned d3-nano on PaySim, 20 test windows", "fraud/eval_d1ft.py": "Phase 9: fine-tuned d1-Omni-2B on PaySim, 20 test windows",
  "fraud/eval_ieee.py": "Phase 11: the first IEEE-CIS fine-tune, 20 test windows", "fraud/eval_ieee_full.py": "Phase 12: fine-tuned on every fit fraud, 20 test windows",
  "fraud/eval_ieee_amount.py": "Phase 14: the amount-aware rule, 20 test windows", "fraud/eval_column_audit.py": "Phase 17: the C, D, V column audit, 8 test windows",
  "fraud/eval_rule.py": "Phase 18: changes to the rule, 10 test windows", "fraud/eval_features.py": "Phase 19: more features for logreg, 10 test windows",
  "fraud/eval_tabular.py": "Phase 20: boosted trees, 10 test windows", "fraud/eval_lm_nuc.py": "Phase 21: the language model's options, dev only",
  "fraud/eval_combos.py": "Phase 22: choosing the final candidates, dev only", "fraud/eval_final.py": "Phase 23: the final test, 20 windows, run once",
  "fraud/eval_lm_cost.py": "Phase 25: training on cost, dev only",
};
const evalWhat = (script) => EVAL_WHAT[script || "fraud/eval.py"] || "";
const SIGNALS = [
  ["dest_in_before", "Receiver: received money before", (v) => (v === 0 ? "never (new account)" : times(v))],
  ["dest_transfer_in_before", "Receiver: received a TRANSFER before", times],
  ["dest_out_before", "Receiver: sent money before", times],
  ["orig_out_before", "Sender: sent money before", times],
  ["orig_in_before", "Sender: received money before", times],
  ["same_amount_step_before", "Same amount earlier this hour", times],
  ["round_amount", "Round amount (multiple of 1,000)", (v) => (v ? "yes" : "no")],
];

// ---------------------------------------------------------------- helpers

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : String(k));
  return e;
}
const svgEl = (tag, attrs) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
};
function times(n) { return n === 0 ? "never" : n === 1 ? "once" : `${n} times`; }
const money = (x) => (x === null || x === undefined ? "–" : Math.round(x).toLocaleString("en-US"));
const amount = (x) => x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (x, d = 0) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)}%`);
const fix = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const ms = (x) => (x >= 100 ? `${x.toFixed(0)} ms` : x >= 1 ? `${x.toFixed(1)} ms` : x >= 0.001 ? `${x.toFixed(3)} ms` : "<0.001 ms");
const secs = (x) => `${(x / 1000).toFixed(1)} s`;
const chip = (action, text) => el("span", { class: `chip ${action}` }, text || action);
const signed = (x) => { const v = Math.round(x); return v === 0 ? "0" : `${v < 0 ? "−" : "+"}${Math.abs(v).toLocaleString("en-US")}`; };
function niceStep(range, n) {  // a round step that cuts range into about n parts
  const raw = range / n, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
}
const kUSD = (t) => (Math.abs(t) >= 1000 ? `${t / 1000}k` : `${t}`);

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function getTrace(file) {
  const r = await fetch(`runs/${encodeURIComponent(file)}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`runs/${file}: ${r.status}`);
  const recs = (await r.text()).split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const header = recs[0];
  const ticks = recs.filter((x) => x.type === "tx");
  if (/^(tab-(all|rich|nocdv)|signal-alone)/.test(header.setup)) for (const r of ticks) r.by = "trees"; // Phase 20 and 21 record trees' scores under the "logreg" kind
  // D3 traces (fraud/d3.py) store each option order as {approve, review, decline} and may hold one order only
  for (const t of ticks) if (t.orders && t.orders.length && !Array.isArray(t.orders[0])) t.orders = t.orders.map((o) => ACTIONS.map((a) => o[a]));
  const last = recs[recs.length - 1];
  return { header, ticks, end: last.type === "end" ? last : null };
}

const tip = el("div", { class: "tip", role: "status" });
document.body.append(tip);
function showTip(evt, lines) {
  tip.replaceChildren(...lines.map((l, i) => el("div", i ? {} : { style: "font-weight:600" }, l)));
  tip.style.display = "block";
  const x = Math.min(evt.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 12, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
const hideTip = () => { tip.style.display = "none"; };

// ---------------------------------------------------------------- the replay

let current = null; // the active Replay (one at a time)
let demo = null; // the auto-demo state, if running

class Replay {
  constructor(file, trace, opts = {}) {
    this.file = file;
    this.h = trace.header;
    this.ticks = trace.ticks;
    this.end = trace.end;
    this.onEnd = opts.onEnd || null;
    this.cum = [];
    let t = 0;
    for (const r of this.ticks) { t += r.ms; this.cum.push(t); }
    this.total = t;
    this.t = 0;
    this.speed = 1;
    this.playing = false;
    this.shown = 0;
    this.follow = true;
    this.sel = null;
    this.last = null;
    this.raf = null;
    this.prefix();
  }

  prefix() {
    // Running totals after each transaction: cost split by kind, confusion, fraud stopped.
    const z = () => ({ cost: 0, missed: 0, friction: 0, reviews: 0, stopped: 0, fraud: 0,
      conf: { legit: { approve: 0, review: 0, decline: 0 }, fraud: { approve: 0, review: 0, decline: 0 } } });
    let s = z();
    this.runs = [s];
    for (const r of this.ticks) {
      s = JSON.parse(JSON.stringify(s));
      const truth = r.truth ? "fraud" : "legit";
      s.cost += r.cost;
      s.conf[truth][r.action] += 1;
      if (r.action === "review") s.reviews += r.cost;
      else if (r.truth && r.action === "approve") s.missed += r.cost;
      else if (!r.truth && r.action === "decline") s.friction += r.cost;
      if (r.truth) { s.fraud += 1; if (r.action !== "approve") s.stopped += 1; }
      this.runs.push(s);
    }
  }

  count(t) { // transactions whose decision finished by replay time t
    let lo = 0, hi = this.cum.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.cum[m] <= t) lo = m + 1; else hi = m; }
    return lo;
  }

  mount() {
    const h = this.h;
    const model = (h.backend && h.backend.model ? h.backend.model.split("/").pop() : "") || "";
    this.playBtn = el("button", { class: "primary", onclick: () => this.toggle(), "aria-label": "Play or pause" }, "Play");
    this.speedSel = el("select", { "aria-label": "Replay speed", onchange: (e) => { this.speed = Number(e.target.value); } },
      ...[0.25, 0.5, 1, 2, 4, 16].map((v) => el("option", { value: v, selected: v === 1 }, `${v}×`)));
    this.clock = el("span", { class: "clock" });
    this.scrub = el("input", { id: "scrub", type: "range", min: 0, max: 1000, value: 0, "aria-label": "Replay position",
      oninput: (e) => { this.seek((Number(e.target.value) / 1000) * this.total); } });
    this.feed = el("ol", { "aria-label": "Transactions, newest first" });
    this.followBtn = el("button", { onclick: () => { this.follow = true; this.sel = null; this.render(true); } }, "Follow latest");
    this.cardBox = el("div");
    this.totalsBox = el("div");
    this.latBox = el("div");
    this.demoBox = el("div");
    const title = `${setupName(h.setup)} · window ${h.window.id}${model ? ` · ${model}` : ""}`;
    app.replaceChildren(
      el("section", { class: "panel" },
        el("div", { class: "controls" },
          el("span", { class: "title" }, title),
          this.demoBox, this.playBtn,
          el("button", { onclick: () => this.step(-1), "aria-label": "Previous transaction" }, "◀"),
          el("button", { onclick: () => this.step(1), "aria-label": "Next transaction" }, "▶"),
          el("button", { onclick: () => this.seek(0) }, "Restart"),
          this.speedSel, this.clock),
        this.scrub),
      el("div", { class: "replay" },
        el("section", { class: "panel feed" },
          el("div", { class: "feed-head" }, el("h2", {}, "Transactions"), this.followBtn), this.feed),
        el("section", { class: "panel card" }, el("h2", {}, "Decision"), this.cardBox),
        el("section", { class: "panel totals" }, el("h2", {}, "Running totals"), this.totalsBox),
        el("section", { class: "panel latency" }, el("h2", {}, "Decision latency"), this.latBox)),
      about(h, this.ticks, this.end, this.total));
    this.render(true);
  }

  toggle() {
    if (this.t >= this.total) this.seek(0);
    this.playing = !this.playing;
    this.last = null;
    this.render(false);
    if (this.playing) this.loop();
  }
  play() { if (!this.playing) this.toggle(); }

  loop() {
    cancelAnimationFrame(this.raf);
    const tick = (ts) => { this.frame(ts); if (this.playing) this.raf = requestAnimationFrame(tick); };
    this.raf = requestAnimationFrame(tick);
  }

  frame(ts) { // advance the clock to timestamp ts (requestAnimationFrame's clock); callable by hand
    if (!this.playing) return;
    if (this.last !== null) this.t = Math.min(this.total, this.t + (ts - this.last) * this.speed);
    this.last = ts;
    const done = this.t >= this.total;
    if (done) this.playing = false;
    this.render(false);
    if (done && this.onEnd) this.onEnd();
  }

  seek(t) { this.t = Math.max(0, Math.min(this.total, t)); this.last = null; this.render(true); }
  step(d) {
    this.playing = false;
    const n = Math.max(0, Math.min(this.ticks.length, this.count(this.t) + d));
    this.t = n === 0 ? 0 : this.cum[n - 1];
    this.follow = true;
    this.render(true);
  }
  stop() { this.playing = false; cancelAnimationFrame(this.raf); this.onEnd = null; }

  select(k) { this.sel = k; this.follow = false; this.render(true); }

  render(force) {
    const n = this.count(this.t);
    this.playBtn.textContent = this.playing ? "Pause" : this.t >= this.total && this.total > 0 ? "Replay" : "Play";
    this.clock.textContent = `${secs(this.t)} / ${secs(this.total)} · ${n} of ${this.ticks.length}`;
    if (document.activeElement !== this.scrub) this.scrub.value = this.total ? Math.round((1000 * this.t) / this.total) : 0;
    if (!force && n === this.shown) return;
    this.renderFeed(n, force);
    this.shown = n;
    const k = this.follow ? n - 1 : this.sel;
    this.cardBox.replaceChildren(k !== null && k >= 0 ? card(this.h, this.ticks[k]) :
      el("p", { class: "muted" }, "Press Play. Transactions appear when their recorded decision finished."));
    this.followBtn.disabled = this.follow;
    const run = n === this.ticks.length && this.end ? { ...this.runs[n], cost: this.end.summary.cost } : this.runs[n];
    this.totalsBox.replaceChildren(totals(this.h, run, n, this.ticks)); // the end uses the trace's own total (per-row costs are rounded)
    this.latBox.replaceChildren(latencyChart(this, n));
  }

  renderFeed(n, force) {
    const k = this.follow ? n - 1 : this.sel;
    if (force || n < this.shown) {
      this.feed.replaceChildren();
      for (let i = 0; i < n; i++) this.feed.prepend(this.feedRow(i));
    } else {
      for (let i = this.shown; i < n; i++) this.feed.prepend(this.feedRow(i));
    }
    for (const li of this.feed.querySelectorAll("li.sel")) li.classList.remove("sel");
    const s = k !== null && k >= 0 ? this.feed.querySelector(`li[data-k="${k}"]`) : null;
    if (s) s.classList.add("sel");
    if (this.follow) this.feed.scrollTop = 0;
  }

  feedRow(i) {
    const r = this.ticks[i];
    const why = r.by === "model" ? `model · ${ms(r.ms)}` : r.rule ? `${BY[r.by] || r.by}: ${r.rule}` : BY[r.by] || r.by;
    return el("li", { "data-k": i, tabindex: 0, onclick: () => this.select(i),
      onkeydown: (e) => { if (e.key === "Enter") this.select(i); } },
      el("span", { class: "k" }, `#${i + 1}`),
      el("span", { class: "what" }, `${r.tx.type} ${amount(r.tx.amount)}`, r.truth ? el("span", { class: "fraud-tag" }, " · fraud") : null),
      chip(r.action),
      el("span", { class: "by" }, why));
  }
}

function card(h, r) {
  const parts = [];
  parts.push(el("div", { class: "tx-line" }, `${r.tx.type} ${amount(r.tx.amount)}`));
  parts.push(el("div", { class: "muted" }, `#${r.i + 1} · to a ${r.tx.to} · hour ${r.tx.hour}:00 · step ${r.step} · ${h.dataset || "PaySim"} row ${r.row}`));
  const decided = [chip(r.action)];
  if (r.wanted !== r.action) decided.push(el("span", { class: "hint" }, ` wanted ${r.wanted}; the review budget was used up, so its fallback (${r.action}) applied`));
  const outcome = r.truth
    ? (r.action === "approve" ? "fraud let through" : r.action === "review" ? "fraud caught in review" : "fraud blocked")
    : (r.action === "approve" ? "legit approved" : r.action === "review" ? "legit cleared in review" : "legit customer declined");
  parts.push(el("div", { class: "section" },
    el("dl", { class: "kv" },
      el("dt", {}, "Decision"), el("dd", {}, ...decided),
      el("dt", {}, "Decided by"), el("dd", {}, BY[r.by] || r.by),
      el("dt", {}, "Truth"), el("dd", {}, r.truth ? el("span", { class: "fraud-tag" }, "fraud") : "legit"),
      el("dt", {}, "Outcome"), el("dd", {}, outcome),
      el("dt", {}, "Cost of this decision"), el("dd", {}, money(r.cost)),
      el("dt", {}, "Reviews left after it"), el("dd", {}, r.budget_left),
      el("dt", {}, "Decision time"), el("dd", {}, ms(r.ms)))));
  if (r.by === "model" && r.scores) {
    const opts = h.options || ACTIONS;
    const vals = ACTIONS.map((a) => r.scores[a]);
    const top = vals.indexOf(Math.max(...vals));
    parts.push(el("div", { class: "section" },
      el("h3", {}, "Model scores (uncalibrated)"),
      el("div", { class: "bars" }, ...opts.map((o, i) => el("div", { class: `bar-row${i === top ? " top" : ""}` },
        el("span", { class: "lab" }, o), el("span", { class: "val" }, fix(vals[i])),
        el("div", { class: "bar-track" }, el("div", { class: "bar-fill", style: `width:${(100 * vals[i]).toFixed(1)}%` }))))),
      r.orders && r.orders.length === 2 ? el("div", { class: "orders" },
        `Averaged over two readings. Listed order: ${r.orders[0].map((x) => fix(x)).join(" / ")}. ` +
        `Reversed order (shown in listed order): ${r.orders[1].map((x) => fix(x)).join(" / ")}.`)
        : r.orders && r.orders.length === 1 ? el("div", { class: "orders" },
          `One reading, options ${(h.backend || {}).orders === "rev" ? "reversed (shown in listed order)" : "as listed"}.`) : null,
      el("dl", { class: "kv", style: "margin-top:6px" },
        el("dt", {}, "Mass outside the labels"), el("dd", {}, fix(r.outside_mass, 4)),
        el("dt", {}, "Prompt tokens"), el("dd", {}, r.prompt_tokens))));
  } else if (r.by === "logreg" && r.scores) {
    parts.push(el("div", { class: "section" }, el("dl", { class: "kv" },
      el("dt", {}, "Logistic regression score (uncalibrated)"), el("dd", {}, fix(r.scores.fraud, 3)))));
  } else if (["logreg", "trees", "lm"].includes(r.by) && r.q !== undefined) {
    const kind = { logreg: "Logistic regression", trees: "Boosted trees", lm: "Language model" }[r.by];
    parts.push(el("div", { class: "section" }, el("dl", { class: "kv" },
      el("dt", {}, `${kind} score (uncalibrated)`), el("dd", {}, fix(r.q, 3)))));
  } else if (r.rule) {
    const rule = (h.rules || []).find((x) => x[0] === r.rule);
    parts.push(el("div", { class: "section" }, el("h3", {}, "Rule"), el("div", {}, r.rule),
      rule ? el("div", { class: "hint" }, `From the training steps: ${rule[3]}`) : null));
  }
  const signals = SIGNALS.filter(([k]) => k in r.signals);
  if (signals.length) parts.push(el("div", { class: "section" }, el("h3", {}, "Signals (earlier rows only)"),
    el("dl", { class: "kv" }, ...signals.flatMap(([k, label, f]) =>
      [el("dt", {}, label), el("dd", {}, f(r.signals[k]))]))));
  return el("div", {}, ...parts);
}

function totals(h, s, n, ticks) {
  const rows = ["legit", "fraud"].map((t) => el("tr", {}, el("td", {}, t === "fraud" ? el("span", { class: "fraud-tag" }, "fraud") : "legit"),
    ...ACTIONS.map((a) => el("td", {}, s.conf[t][a]))));
  const left = n ? ticks[n - 1].budget_left : Math.floor(h.costs.review_share * ticks.length);
  return el("div", {},
    el("div", { class: "hero" }, money(s.cost)),
    el("div", { class: "hint" }, `cost after ${n} of ${ticks.length} transactions`),
    el("dl", { class: "kv section" },
      el("dt", {}, "Fraud let through"), el("dd", {}, money(s.missed)),
      el("dt", {}, "Legit customers declined"), el("dd", {}, money(s.friction)),
      el("dt", {}, "Reviews"), el("dd", {}, money(s.reviews)),
      el("dt", {}, "Fraud stopped"), el("dd", {}, `${s.stopped} of ${s.fraud}`),
      el("dt", {}, "Reviews left"), el("dd", {}, left)),
    el("div", { class: "section scroll-x" },
      el("table", { class: "confusion" },
        el("thead", {}, el("tr", {}, el("th", {}, "truth"), ...ACTIONS.map((a) => el("th", {}, a)))),
        el("tbody", {}, ...rows))));
}

function latencyChart(rp, n) {
  const pts = [];
  for (let i = 0; i < rp.ticks.length; i++) if (rp.ticks[i].by === "model") pts.push(i);
  if (!pts.length) {
    return el("p", { class: "muted" },
      `No model decisions in this setup. All ${rp.ticks.length} decisions took ${ms(rp.total)} in total, so a 1× replay is nearly instant.`);
  }
  const W = 360, H = 170, L = 34, R = 8, T = 10, B = 22;
  const maxMs = Math.max(260, ...pts.map((i) => rp.ticks[i].ms)) * 1.08;
  const x = (i) => L + ((W - L - R) * i) / Math.max(1, rp.ticks.length - 1);
  const y = (v) => T + (H - T - B) * (1 - v / maxMs);
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Latency of each model decision, in ms, by transaction" });
  for (const v of [0, 100, 200]) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1 }));
    const t = svgEl("text", { x: L - 4, y: y(v) + 3.5, "text-anchor": "end" }); t.textContent = v; svg.append(t);
  }
  svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(250), y2: y(250), stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "4 3" }));
  const bl = svgEl("text", { x: W - R, y: y(250) - 4, "text-anchor": "end" }); bl.textContent = "bar: p90 ≤ 250 ms"; svg.append(bl);
  const xl = svgEl("text", { x: W - R, y: H - 4, "text-anchor": "end" }); xl.textContent = "transaction →"; svg.append(xl);
  const shown = pts.filter((i) => i < n);
  for (const i of shown) {
    svg.append(svgEl("circle", { cx: x(i), cy: y(rp.ticks[i].ms), r: i === (rp.follow ? n - 1 : rp.sel) ? 4 : 2.2,
      fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 1 }));
  }
  svg.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = null;
    for (const i of shown) if (best === null || Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    if (best === null || Math.abs(x(best) - px) > 12) return hideTip();
    const r = rp.ticks[best];
    showTip(e, [`#${best + 1} ${r.tx.type} ${amount(r.tx.amount)}`, `${ms(r.ms)} · ${r.action}`]);
  });
  svg.addEventListener("mouseleave", hideTip);
  svg.addEventListener("click", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = null;
    for (const i of shown) if (best === null || Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    if (best !== null) rp.select(best);
  });
  const lat = shown.map((i) => rp.ticks[i].ms).sort((a, b) => a - b);
  const q = (p) => (lat.length ? lat[Math.min(lat.length - 1, Math.ceil((p / 100) * lat.length) - 1)] : 0);
  return el("div", {}, svg,
    el("div", { class: "hint" }, lat.length
      ? `${lat.length} model decisions so far · p50 ${ms(q(50))} · p90 ${ms(q(90))} · max ${ms(lat[lat.length - 1])}. ` +
        "Each is one transaction read in both option orders."
      : "No model decision yet."));
}

// ---------------------------------------------------------------- About this run

function fact(label, value, hint) {
  return el("div", { class: "fact" }, el("span", {}, label), el("span", { class: "v" }, value),
    hint ? el("span", { class: "hint" }, hint) : null);
}
function factCard(title, ...facts) { return el("section", { class: "panel fact-card" }, el("h3", {}, title), ...facts); }
function noteCard(title, notes) {
  return el("section", { class: "panel note-card" }, el("h3", {}, title), el("ul", {}, ...notes.map((n) => el("li", {}, n))));
}

function ruleCard(h) {  // how this run turned a score into a decision: Phase 14's amount-aware rule, or two lines on the score
  const pol = h.policy;
  if (pol && pol.kind === "amount-aware") {
    return factCard("Decision rule",
      fact("Rule", "Amount-aware", "Lowest expected cost of approve, decline and review, for this row's amount"),
      fact("Review price", `${money(h.costs.review_fee)} + ${pol.lambda}`, pol.gamma ? `The fee plus lambda, which rises as reviews run short (gamma ${pol.gamma})` : "The fee plus lambda, the price of using up a review, picked on dev"),
      pol.flat ? fact("Customer line", `${money(pol.flat)} per decline`, "Added to the cost of declining when deciding, never to what the run is charged") : null,
      pol.shift ? fact("Long-run prior", `${pct(pol.shift[0], 2)} → ${pct(pol.shift[1], 2)}`, "Moves the fit's level and keeps the ranking") : null,
      fact("Weighs the score by", (pol.fit_features || ["logit"]).length > 1 ? "A dev fit of score and amount" : "A dev fit of the score", "Used only to weigh costs. Scores are uncalibrated, not probabilities"));
  }
  const th = h.thresholds;
  if (th && (th.review !== null || th.decline !== null)) {
    return factCard("Decision rule",
      fact("Rule", "Two lines on the score", "The same for every amount"),
      fact("Review from", th.review === null ? "never" : fix(th.review, 4), "Scores are uncalibrated"),
      fact("Decline from", th.decline === null ? "never" : fix(th.decline, 4), "Past the budget, a review becomes an approve"));
  }
  return null;
}

function about(h, ticks, end, total) {
  const s = end ? end.summary : null;
  const b = h.backend || {};
  const model = ticks.filter((r) => r.by === "model");
  const wrongDecl = ticks.filter((r) => !r.truth && r.action === "decline");
  const missed = ticks.filter((r) => r.truth && r.action === "approve");
  const sum = (rs) => rs.reduce((a, r) => a + r.cost, 0);
  const banner = !s ? el("div", { class: "banner bad" }, el("div", { class: "big" }, "This run did not finish (no end record)."))
    : el("div", { class: `banner ${missed.length || wrongDecl.length ? "bad" : "good"}` },
      el("div", { class: "big" }, `Cost ${money(s.cost)} · ${s.confusion.fraud.approve === 0 ? "every" : `${s.fraud - s.confusion.fraud.approve} of ${s.fraud}`} fraud stopped`
        + ` · ${wrongDecl.length} legit declined`),
      el("div", { class: "hint" }, `${setupName(h.setup)} on window ${h.window.id} (${h.window.split}).`));
  const natural = h.window.enriched === false; // IEEE-CIS: contiguous rows at the natural fraud rate, no filter
  const facts = [
    factCard("Window",
      natural ? fact("Window", h.window.id, `${h.window.split === "test" ? "Held-out test rows" : "Dev rows, after the fit rows"}, seed ${h.window.seed}`)
        : fact("Window", h.window.id, `${h.window.split === "test" ? "Held-out test steps" : "Training-period dev window"}, seed ${h.window.seed}`),
      fact(natural ? "Rows" : "Steps (hours)", `${h.window.steps[0]}–${h.window.steps[1]}`),
      fact("Transactions", h.window.size, natural ? (h.window.steps[1] - h.window.steps[0] + 1 === h.window.size ? "Contiguous rows in arrival order" : "Rows in arrival order, skipping rows other windows used") : "Sampled in arrival order from the window's hours"),
      natural ? fact("Fraud", s ? s.fraud : "–", "Natural rate, not enriched")
        : fact("Fraud", h.window.fraud, `Enriched; the natural rate in these hours is ${pct(h.window.natural_fraud_rate, 3)}`)),
    factCard("Setup",
      fact("Setup", h.setup, SETUP_NAME[h.setup] ? setupName(h.setup) : ""),
      model.length ? fact("Model", (b.model || "").split("/").pop() || b.kind, natural
        ? `${b.device_name || b.device || ""}, ${b.dtype || ""}${b.adapter ? `, LoRA ${b.adapter.dir}` : ""}`
        : b.orders
        ? `${b.kind === "llamacpp-systemone" ? `llama-server, ${b.gguf}` : `${b.device || ""}, ${b.dtype || ""}`}${b.adapter ? `, LoRA ${b.adapter.dir}` : ""}`
        : `${b.device_name || b.device || ""}, LM head ${b.head_dtype || "?"}`) : null,
      model.length ? (natural // these headers do not name the order; each decision records the orders it read
        ? (model[0].orders || []).length === 1 ? fact("Option order", "one", "One option order, chosen on dev windows")
          : fact("Option order", "both, averaged", "Each decision is read in both option orders and averaged")
        : b.orders
        ? fact("Option order", { both: "both, averaged", fwd: "as listed", rev: "reversed" }[b.orders] || b.orders,
          b.orders === "both" ? "Each decision is read in both option orders and averaged" : "One option order, chosen on dev windows")
        : fact("Order debias", h.order_debias ? "on" : "off", "Each decision is read in both option orders and averaged")) : null,
      natural ? null : fact("Type filter", (h.fraud_free_types || []).join(", "), "Approved without a model: no fraud in training")),
    ruleCard(h),
    factCard("Costs",
      fact("Wrong decline", `${pct(h.costs.friction_rate)} of amount`, `At least ${h.costs.friction_min}`),
      fact("Review", money(h.costs.review_fee), "An analyst finds the truth"),
      fact("Review budget", `${pct(h.costs.review_share)} of the window`, `${Math.floor(h.costs.review_share * h.window.size)} reviews, in arrival order`)),
  ];
  if (s) {
    facts.push(factCard("Result",
      fact("Cost", money(s.cost)),
      natural ? null : fact("Cost at the natural rate", money(s.cost_natural), "Each row reweighted to its hours' real fraud rate"),
      fact("Recall", fix(s.recall), "Fraud declined or caught in review"),
      fact("Precision", fix(s.precision), "Share of alerts that were fraud"),
      fact("Alert rate", pct(s.alert_rate, 1), "Share of transactions not approved"),
      fact("Reviews", `${s.reviews_done} of ${s.reviews_wanted} wanted`)));
    facts.push(factCard("Speed",
      fact("Replay length at 1×", secs(total), "The sum of the recorded decision times"),
      model.length ? fact("Model decisions", s.model_decisions) : null,
      model.length ? fact("Latency p50", ms(s.latency_ms.p50)) : null,
      model.length ? fact("Latency p90", ms(s.latency_ms.p90), "The eval's bar is p90 ≤ 250 ms") : null,
      model.length ? fact("Latency max", ms(s.latency_ms.max)) : null));
  }
  const notes = [];
  if (model.length) {
    const wanted = new Set(model.map((r) => r.wanted));
    if (wanted.size === 1) notes.push(`The model chose “${[...wanted][0]}” for all ${model.length} transactions it judged.`);
    const flips = model.filter((r) => r.orders && r.orders.length === 2 && r.orders[0].indexOf(Math.max(...r.orders[0])) !== r.orders[1].indexOf(Math.max(...r.orders[1])));
    if (flips.length) notes.push(`In ${flips.length} of ${model.length} decisions the two option orders disagreed on the top option.`);
    notes.push("Model scores are uncalibrated softmax values over the three labels, not fraud probabilities.");
  }
  if (wrongDecl.length) notes.push(`${wrongDecl.length} legit transactions were declined, costing ${money(sum(wrongDecl))}.`);
  if (missed.length) notes.push(`${missed.length} fraud transactions were approved, losing ${money(sum(missed))}.`);
  const out = ticks.find((r) => r.wanted === "review" && r.action !== "review");
  if (out) notes.push(`The review budget ran out at transaction #${out.i + 1}; later reviews became their fallback.`);
  if (!notes.length) notes.push("No fraud was let through and no legit customer was declined.");
  const ruleNotes = (h.rules || []).map((r) => `${r[0]} → ${r[1]}: ${r[3]}`);
  const kids = [el("h2", {}, "About this run"), banner, el("div", { class: "cards", style: "margin-top:12px" }, ...facts),
    el("div", { class: "cards", style: "margin-top:12px" }, noteCard("What happened", notes),
      noteCard("How a decision is replayed", [
        "Each transaction appears when its recorded decision finished.",
        natural ? "A row's score reads its own columns and earlier rows. No label is an input." : "Signals use only earlier rows; the four balance columns are never read.",
        "A review costs a fee and finds the truth until the budget runs out."]),
      h.setup === "rules" ? noteCard("The rule table (first match decides)", ruleNotes) : null)];
  if (h.example_prompt) {
    kids.push(el("details", { class: "prompt panel", style: "margin-top:12px" },
      el("summary", {}, "The prompt the model reads (first transaction it judged)"), el("pre", {}, h.example_prompt)));
  }
  return el("section", { class: "about" }, ...kids);
}

// ---------------------------------------------------------------- pages

async function openRun(file, opts) {
  app.replaceChildren(el("p", { class: "muted" }, `Loading ${file}…`));
  const trace = await getTrace(file);
  if (current) current.stop();
  current = new Replay(file, trace, opts);
  current.mount();
  document.title = `${trace.header.setup} ${trace.header.window.id} · Fraud Console`;
  return current;
}

// The runs table: sortable columns [key, label, numeric]. The sort stays while you open runs and come back.
const RUN_COLS = [["label", "run"], ["setup", "setup"], ["window", "window"], ["model", "model"],
  ["cost", "cost", true], ["recall", "recall", true], ["total_ms", "length at 1×", true]];
let runSort = { key: "created", dir: -1 }; // newest first, pinned runs on top
const byText = new Intl.Collator("en", { numeric: true }).compare; // test99 before test100

function sortRuns(rows) {
  const { key, dir } = runSort;
  const missing = (v) => v === null || v === undefined || v === "";
  rows.sort((a, b) => {
    if (key === "created" && b.pinned !== a.pinned) return b.pinned - a.pinned;
    const x = a[key], y = b[key];
    if (missing(x) || missing(y)) return missing(x) - missing(y); // no value: last either way
    const c = typeof x === "number" ? x - y : byText(String(x), String(y));
    return dir * c || byText(b.created || "", a.created || "");
  });
}

async function startPage() {
  document.title = "Fraud Console";
  const [{ runs }, { evals }] = await Promise.all([getJSON("api/runs"), getJSON("api/evals")]);
  const latest = evals[0];
  const finalTest = evals.find((e) => e.script === "fraud/eval_final.py");
  const setups = [...new Set(runs.map((r) => r.setup))].sort();
  const body = el("tbody");
  const setupSel = el("select", { "aria-label": "Setup" }, el("option", { value: "" }, "all setups"),
    ...setups.map((s) => el("option", { value: s }, s)));
  const evalBox = el("input", { type: "checkbox", id: "with-eval" });
  const heads = RUN_COLS.map(([key, label, right]) => {
    const btn = el("button", { class: "sort", onclick: () => {
      runSort = runSort.key === key ? { key, dir: -runSort.dir } : { key, dir: right ? -1 : 1 };
      fill();
    } }, label, el("span", { class: "arrow", "aria-hidden": "true" }));
    return { key, th: el("th", { class: right ? "r" : "" }, btn), btn };
  });
  const fill = () => {
    const rows = runs.filter((r) => (!setupSel.value || r.setup === setupSel.value) && (evalBox.checked || !r.eval));
    sortRuns(rows);
    for (const h of heads) {
      const on = h.key === runSort.key;
      h.th.setAttribute("aria-sort", on ? (runSort.dir > 0 ? "ascending" : "descending") : "none");
      h.btn.lastChild.textContent = on ? (runSort.dir > 0 ? " ▲" : " ▼") : "";
    }
    body.replaceChildren(...rows.map((r) => el("tr", { class: "link", tabindex: 0,
      onclick: () => { location.hash = `#/run/${encodeURIComponent(r.file)}`; },
      onkeydown: (e) => { if (e.key === "Enter") location.hash = `#/run/${encodeURIComponent(r.file)}`; } },
      el("td", {}, r.label, r.pinned ? el("span", { class: "chip plain", style: "margin-left:6px" }, "pinned") : null,
        r.finished ? null : el("span", { class: "chip fail", style: "margin-left:6px" }, "unfinished")),
      el("td", {}, r.setup), el("td", {}, r.window), el("td", {}, r.model || "–"),
      el("td", { class: "r" }, money(r.cost)), el("td", { class: "r" }, fix(r.recall)),
      el("td", { class: "r" }, r.total_ms !== null && r.total_ms !== undefined ? secs(r.total_ms) : "–"))));
    if (!rows.length) body.append(el("tr", {}, el("td", { colspan: 7, class: "muted" }, "No runs. Capture one: python -m fraud.capture --window dev0")));
  };
  setupSel.addEventListener("change", fill);
  evalBox.addEventListener("change", fill);
  fill();
  app.replaceChildren(
    el("h1", {}, "Approve, review or decline: System One on PaySim and IEEE-CIS"),
    el("p", { class: "intro" }, "A local model judges each TRANSFER and CASH_OUT (Qwen2.5-1.5B; later evals add fine-tuned Qwen2.5-1.5B, " +
      "d3-nano and d1-Omni-2B), and a rule filter approves the transaction types " +
      "that had no fraud in training. The console replays recorded runs at their recorded speed and shows the pre-registered eval " +
      "against rules-only, logistic regression, random and approve-all. IEEE-CIS runs (real card payments, no filter, a review at 10 USD) replay here too, and their evals are listed below."),
    el("div", { class: "cards" },
      finalTest ? el("section", { class: "panel bar-card" }, el("h3", {}, "The final test"),
        el("div", {}, `Eval ${finalTest.id}`),
        el("div", { class: "hint" }, "IEEE-CIS, 20 windows reserved first and run once. The numbers the write-up and the talk rest on."),
        el("p", {}, el("a", { href: `#/eval/${finalTest.id}` }, "Open the final test"))) : null,
      el("section", { class: "panel bar-card" }, el("h3", {}, "Latest eval"),
        latest ? el("div", {},
          el("div", {}, `Eval ${latest.id}${latest.complete === false ? " (incomplete)" : ""}`),
          el("div", { class: "hint" }, evalWhat(latest.script) || latest.script || ""),
          ...Object.entries(latest.bars || {}).map(([k, b]) => el("div", { style: "margin-top:6px" },
            chip(b.pass ? "pass" : "fail", b.pass ? "PASS" : "FAIL"), ` ${b.rule}`)),
          el("p", {}, el("a", { href: `#/eval/${latest.id}` }, "Open the eval"))) : el("p", { class: "muted" }, "No eval results yet.")),
      el("section", { class: "panel bar-card" }, el("h3", {}, "Auto-demo"),
        el("p", {}, "Plays the model runs one after another at 1×, pinned runs first."),
        el("button", { class: "primary", onclick: () => { location.hash = "#/demo"; } }, "Start the auto-demo"))),
    el("section", { class: "panel", style: "margin-top:12px" },
      el("h2", {}, "Evals"),
      el("div", { class: "scroll-x" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["eval", "what it is", "data and review fee", "script"].map((t) => el("th", {}, t)))),
        el("tbody", {}, ...evals.map((e) => el("tr", {}, el("td", {}, el("a", { href: `#/eval/${e.id}` }, e.id)), el("td", {}, evalWhat(e.script)),
          el("td", {}, e.ieee ? "IEEE-CIS, 10 USD" : "PaySim, 50 USD"), el("td", {}, e.script || "fraud/eval.py"))))))),
    el("section", { class: "panel", style: "margin-top:12px" },
      el("h2", {}, "Runs"),
      el("div", { class: "filters" }, setupSel, el("label", { for: "with-eval" }, evalBox, " include eval runs"),
        el("button", { onclick: () => { runSort = { key: "created", dir: -1 }; fill(); } }, "Newest first")),
      el("p", { class: "hint" }, "Click a column to sort by it; click again to reverse. Runs without a value (unfinished) go last."),
      el("div", { class: "scroll-x" }, el("table", {},
        el("thead", {}, el("tr", {}, ...heads.map((h) => h.th))),
        body))));
}

const EVAL_SERIES = [["hybrid-1.5b", "var(--series-1)"], ["rules", "var(--series-2)"], ["logreg", "var(--series-3)"],
  ["hybrid-ft", "var(--series-4)"], ["hybrid-d3ft", "var(--series-5)"], ["hybrid-d1ft", "var(--series-5)"]];  // one page never shows both
const BAR_TITLE = { bar1: "Bar 1: cost", bar2: "Bar 2: speed", barA: "Bar A: vs logistic regression", barB: "Bar B: speed",
  barC: "Bar C: vs fine-tuned 1.5B" };
const WRITE_UP = { "fraud/eval_ft.py": "fraud/RESULTS-FT.md", "fraud/eval_d3ft.py": "fraud/RESULTS-D3FT.md",
  "fraud/eval_d1ft.py": "fraud/RESULTS-D1FT.md", "fraud/eval_ieee.py": "fraud/RESULTS-IEEE.md", "fraud/eval_ieee_full.py": "fraud/RESULTS-IEEE-FULL.md",
  "fraud/eval_ieee_amount.py": "fraud/RESULTS-IEEE-AMOUNT.md", "fraud/eval_column_audit.py": "fraud/RESULTS-COLUMN-AUDIT.md", "fraud/eval_rule.py": "fraud/RESULTS-RULE.md",
  "fraud/eval_features.py": "fraud/RESULTS-FEATURES.md", "fraud/eval_tabular.py": "fraud/RESULTS-TABULAR.md", "fraud/eval_lm_nuc.py": "fraud/RESULTS-LM-NUC.md", "fraud/eval_combos.py": "fraud/RESULTS-COMBOS.md", "fraud/eval_lm_cost.py": "fraud/RESULTS-LM-COST.md", "fraud/eval_final.py": "fraud/RESULTS-FINAL.md" };

// fraud/eval_d3ft.py and fraud/eval_d1ft.py store only their own setup's windows (keyed by window) and point at the Phase 5
// eval for the other setups on the same windows; merge them into the shape of the other evals.
async function withBaselines(r) {
  if (!r.phase5_eval) return { r, traceEval: () => r.eval };
  const p5 = await getJSON(`results/eval/${encodeURIComponent(r.phase5_eval)}.json`);
  const own = Object.keys(r.totals).find((s) => !p5.totals[s]);
  const tot = r.totals[own];
  const natural = r.test_windows.map((w) => r.windows[w] && r.windows[w].cost_natural).filter((v) => v !== undefined);
  const merged = { ...r,
    windows: { [own]: r.windows, ...p5.windows },
    totals: { [own]: { ...tot, cost_natural_mean: natural.length ? natural.reduce((a, b) => a + b, 0) / natural.length : undefined },
      ...p5.totals } };
  return { r: merged, traceEval: (s) => (s === own ? r.eval : r.phase5_eval) };
}
const SPEED_BARS = new Set(["bar2", "barB"]);

// A scatter: one small dot per setup per window, and a soft dotted line at each setup's mean (r.means, the mean the table shows), labelled at the right edge.
// r.linear: a linear axis from 0 (IEEE-CIS, where the setups sit close together); otherwise log (PaySim, whose costs span orders of magnitude).
// Interactive: every setup of the eval has a toggle. r.chart_series (or EVAL_SERIES) start on; the rest (r.extra_series, or every other
// setup in r.windows) start off under "More setups" and take a free colour when turned on, at most PALETTE.length at once. The axis
// rescales to what is shown, hovering a setup picks it out, and a button hides or shows the mean lines. Motion is short and is off
// under prefers-reduced-motion (console.css).
const PALETTE = [1, 2, 3, 4, 5, 6, 7, 8].map((k) => `var(--series-${k})`);
function evalChart(r) {
  const ws = r.test_windows;
  const series = r.chart_series || EVAL_SERIES.filter(([s]) => r.windows[s]);
  const extra = (r.extra_series || Object.keys(r.windows)).filter((s) => !series.some(([t]) => t === s) && r.windows[s] && ws.some((w) => r.windows[s][w]));
  const label = r.label || setupName;
  const means = r.means || {};
  const W = 720, H = 280, L = 52, R = 70, T = 12, B = 26;
  const plotH = H - T - B;
  const x = (i) => L + ((W - L - R) * (i + 0.5)) / ws.length;
  const scale = r.linear ? "linear scale from 0" : "log scale";
  const dense = ws.length > 30;  // the 88-chain dev pages: no side-by-side offset, smaller dots
  const all = [...series.map(([s, c]) => [s, c]), ...extra.map((s) => [s, null])];
  const state = { on: new Set(series.map(([s]) => s)), meansOn: true };
  const svg = svgEl("svg", { class: "chart evalchart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Cost per ${r.unit || "test window"} in USD for ${series.map(([s]) => label(s)).join(", ")}, ${scale}, with each setup's mean as a dotted line` });
  const gTicks = svgEl("g", {}), gDots = svgEl("g", {}), gMeans = svgEl("g", { class: "means" }), gLabs = svgEl("g", { class: "means" });
  svg.append(gTicks);
  const every = Math.max(2, Math.ceil(ws.length / 20));
  ws.forEach((w, i) => {
    if (i % every === 0) { const t = svgEl("text", { x: x(i), y: H - 8, "text-anchor": "middle" }); t.textContent = w.replace(/^\D+/, ""); svg.append(t); }
  });
  svg.append(gDots, gMeans, gLabs);
  const nSide = Math.min(series.length, 5);
  // Per setup: a group of dots, a mean line (drawn at y 0 and moved by a transform, so it can glide) and a label with its leader.
  const parts = all.map(([s, color], j) => {
    const g = svgEl("g", { class: "s-group" });
    const dots = [];
    ws.forEach((w, i) => {
      const v = r.windows[s][w] && r.windows[s][w].cost;
      if (v === undefined || v === null) return;
      const dx = dense ? 0 : ((j % nSide) - (nSide - 1) / 2) * 3.5;
      const c = svgEl("circle", { class: "s-dot", cx: x(i) + dx, cy: T + plotH, r: dense ? 2 : 2.6, stroke: "var(--surface)", "stroke-width": 0.6, "fill-opacity": dense ? 0.7 : 0.85 });
      g.append(c);
      dots.push({ el: c, cx: x(i) + dx, v, w, s });
    });
    gDots.append(g);
    const m = means[s];
    const has = m !== undefined && m !== null;
    const mg = svgEl("g", { class: "s-mean" }), lg = svgEl("g", { class: "s-lab" });
    const line = svgEl("line", { class: "mean-line", x1: L, x2: W - R, y1: 0, y2: 0, "stroke-width": 1.5, "stroke-dasharray": "1.5 4", "stroke-linecap": "round" });
    const text = svgEl("text", { x: W - R + 10, y: 4 });
    const lead = svgEl("polyline", { class: "s-lead", fill: "none", "stroke-width": 1, "stroke-opacity": 0.6 });
    if (has) { text.textContent = money(m); mg.append(line); lg.append(text); gMeans.append(mg); gLabs.append(lg, lead); }
    return { s, color, g, dots, m: has ? m : null, mg, lg, line, text, lead, extra: j >= series.length };
  });
  function paint(p, color) {
    p.color = color;
    for (const d of p.dots) d.el.setAttribute("fill", color);
    p.line.setAttribute("stroke", color);
    p.lead.setAttribute("stroke", color);
    p.text.setAttribute("style", `fill:${color};font-weight:600`);
    if (p.swatch) p.swatch.style.background = color;
  }
  const place = (el, y) => { el.style.transform = `translate(0px, ${y.toFixed(1)}px)`; };
  let leadTimer = null;
  function layout(first) {
    const shown = parts.filter((p) => state.on.has(p.s));
    const vals = shown.flatMap((p) => p.dots.map((d) => d.v)).filter((v) => v > 0);
    let y, ticks;
    if (r.linear) {
      const max = Math.max(...vals, ...shown.map((p) => p.m || 0), 1);
      const step = niceStep(max, 5), top = Math.ceil(max / step) * step;
      ticks = []; for (let t = 0; t <= top + 1e-9; t += step) ticks.push(t);
      y = (v) => T + plotH * (1 - v / top);
    } else {
      const lo = Math.floor(Math.log10(Math.min(...vals))), hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...vals))));
      ticks = []; for (let p = lo; p <= hi; p++) ticks.push(10 ** p);
      y = (v) => T + plotH * (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo));
    }
    gTicks.replaceChildren(...ticks.flatMap((t) => {
      const tx = svgEl("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end", class: first ? "" : "fade-in" });
      tx.textContent = t >= 1e6 ? `${t / 1e6}M` : kUSD(t);
      return [svgEl("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: "var(--grid)", "stroke-width": 1 }), tx];
    }));
    for (const p of parts) {
      const on = state.on.has(p.s);
      for (const e of [p.g, p.mg, p.lg, p.lead]) e.classList.toggle("is-off", !on);
      if (!on && !p.placed) continue;  // a setup never shown keeps its dots at the bottom, hidden, until it is turned on
      p.placed = true;
      for (const d of p.dots) { d.cy = y(d.v); d.el.style.cy = `${d.cy.toFixed(1)}px`; }
      if (p.m !== null) { p.my = y(p.m); place(p.mg, p.my); }
    }
    // Labels of the shown means, sorted and pushed 13 px apart; leaders are redrawn once the lines have moved.
    const labs = shown.filter((p) => p.m !== null).sort((a, b) => a.my - b.my);
    const GAP = 13;
    labs.forEach((p, i) => { p.ly = i ? Math.max(p.my, labs[i - 1].ly + GAP) : p.my; });
    if (labs.length) labs[labs.length - 1].ly = Math.min(labs[labs.length - 1].ly, H - B);
    for (let i = labs.length - 2; i >= 0; i--) labs[i].ly = Math.min(labs[i].ly, labs[i + 1].ly - GAP);
    for (const p of labs) place(p.lg, p.ly);
    clearTimeout(leadTimer);
    const drawLeads = () => { for (const p of labs) { p.lead.setAttribute("points", `${W - R},${p.my} ${W - R + 5},${p.ly} ${W - R + 8},${p.ly}`); p.lead.classList.remove("moving"); } };
    if (first) drawLeads();
    else { for (const p of labs) p.lead.classList.add("moving"); leadTimer = setTimeout(drawLeads, 460); }
  }
  function focus(s) {
    svg.classList.toggle("dim", s !== null);
    for (const p of parts) for (const e of [p.g, p.mg, p.lg, p.lead]) e.classList.toggle("hot", p.s === s);
  }
  const note = el("span", { class: "hint", role: "status" });
  const sync = () => {
    for (const p of parts) p.btn.setAttribute("aria-pressed", String(state.on.has(p.s)));
    allBtn.disabled = series.every(([s]) => state.on.has(s)) && state.on.size === series.length;
    if (moreSum) moreSum.textContent = `More setups (${extra.length}${extra.some((s) => state.on.has(s)) ? `, ${extra.filter((s) => state.on.has(s)).length} shown` : ""})`;
  };
  function toggle(p) {
    note.textContent = "";
    if (state.on.has(p.s)) {
      if (state.on.size === 1) return;
      state.on.delete(p.s);
    } else {
      // Keep the setup's own colour unless a shown setup has it; otherwise take a free one. At most PALETTE.length at once.
      const used = new Set(parts.filter((q) => state.on.has(q.s)).map((q) => q.color));
      if (!p.color || used.has(p.color)) {
        const free = PALETTE.find((c) => !used.has(c));
        if (!free) { note.textContent = `At most ${PALETTE.length} setups at once: hide one first.`; return; }
        paint(p, free);
      }
      state.on.add(p.s);
    }
    sync();
    layout(false);
  }
  for (const p of parts) {
    p.swatch = el("i", {});
    p.btn = el("button", { class: "s-toggle", "aria-pressed": "false", title: `${setupName(p.s)}. Click to show or hide it.`, onclick: () => toggle(p),
      onmouseenter: () => { if (state.on.has(p.s)) focus(p.s); }, onmouseleave: () => focus(null),
      onfocus: () => { if (state.on.has(p.s)) focus(p.s); }, onblur: () => focus(null) }, p.swatch, label(p.s));
    if (p.color) paint(p, p.color);
  }
  const allBtn = el("button", { class: "s-all", onclick: () => {  // back to the starting setups and colours
    state.on = new Set(series.map(([s]) => s));
    for (const p of parts) if (!p.extra) paint(p, series.find(([s]) => s === p.s)[1]);
    note.textContent = "";
    sync();
    layout(false);
  } }, "Reset");
  const anyMean = parts.some((p) => p.m !== null);
  const meanBtn = !anyMean ? null : el("button", { class: "s-means", "aria-pressed": "true", onclick: () => {
    state.meansOn = !state.meansOn;
    meanBtn.setAttribute("aria-pressed", String(state.meansOn));
    svg.classList.toggle("no-means", !state.meansOn);
    if (state.meansOn) { svg.classList.remove("draw", "redraw"); void svg.getBoundingClientRect(); svg.classList.add("redraw"); }  // replay the lines' draw-in
  } }, el("span", { class: "dotline", "aria-hidden": "true" }), "Mean lines");
  const moreSum = extra.length ? el("summary", {}) : null;
  const more = extra.length ? el("details", { class: "more-setups" }, moreSum,
    el("div", { class: "legend toggles", role: "group", "aria-label": "More setups" }, ...parts.filter((p) => p.extra).map((p) => p.btn))) : null;
  svg.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W, py = ((e.clientY - box.top) / box.height) * H;
    let best = null, bd = 1e9;
    for (const p of parts) if (state.on.has(p.s)) for (const d of p.dots) { const dd = Math.hypot(d.cx - px, d.cy - py); if (dd < bd) { bd = dd; best = d; } }
    if (!best || bd > 18) return hideTip();
    showTip(e, [`${best.w} · ${label(best.s)}`, `cost ${money(best.v)}`, means[best.s] !== undefined ? `mean ${money(means[best.s])}` : ""].filter(Boolean));
  });
  svg.addEventListener("mouseleave", hideTip);
  sync();
  layout(true);
  svg.classList.add("draw");  // the first draw: dots fade up and the mean lines draw in from the left
  return el("div", {},
    el("div", { class: "chart-controls" },
      el("div", { class: "legend toggles", role: "group", "aria-label": "Setups on the chart" }, ...parts.filter((p) => !p.extra).map((p) => p.btn)),
      el("div", { class: "toggles" }, meanBtn, allBtn)),
    more, note,
    svg, el("div", { class: "hint" }, `Cost per ${r.unit || "test window"} in USD, ${scale} (${r.unit ? "its" : "window"} number on the x axis). ` +
      `${anyMean ? "Each dotted line is that setup's mean over all of them, the number in the table, written at the right edge. " : ""}` +
      `Click a setup to hide or show it; the axis rescales to what is shown. Hover one to pick it out.` +
      `${extra.length ? ` More setups adds any other setup in this eval, up to ${PALETTE.length} at once.` : ""}`));
}

function devGate(d) {
  if (d.variant) return devGateOrders(d);
  return el("section", { class: "panel bar-card", style: "margin-bottom:12px" }, el("h3", {}, "Dev gate (dev0–dev9)"),
    el("div", {}, chip(d.gate.pass ? "pass" : "fail", d.gate.pass ? "PASS" : "FAIL"), ` ${d.gate.rule}`),
    ...Object.entries(d.auc).map(([s, a]) => el("div", { style: "margin-top:6px" }, `${setupName(s)}: AUC ${a.auc.toFixed(3)}`,
      el("div", { class: "hint" }, `decline score, ${a.rows} model rows, ${a.fraud} fraud`))),
    el("div", { style: "margin-top:6px" }, `Thresholds: review ≥ ${d.thresholds.review}, decline ≥ ${d.thresholds.decline}`,
      el("div", { class: "hint" }, "picked on dev windows by cost, before any new test window ran")));
}

function devGateOrders(d) {  // fraud/eval_d3ft.py, eval_d1ft.py: one model read in both option orders, gated against logreg
  const order = { both: "both orders averaged", fwd: "options as listed", rev: "options reversed" };
  return el("section", { class: "panel bar-card", style: "margin-bottom:12px" }, el("h3", {}, "Dev gate (dev0–dev9)"),
    el("div", {}, chip(d.gate.pass ? "pass" : "fail", d.gate.pass ? "PASS" : "FAIL"), ` ${d.gate.rule}`),
    el("div", { style: "margin-top:6px" }, `Model: AUC ${d.auc[d.variant].toFixed(4)} (${order[d.variant]})`,
      el("div", { class: "hint" }, `decline score, ${d.rows} model rows, ${d.fraud} fraud`)),
    el("div", { style: "margin-top:6px" }, `Logistic regression: AUC ${d.logreg.auc.toFixed(4)}`,
      el("div", { class: "hint" }, "same rows; the gate")),
    el("div", { style: "margin-top:6px" }, `Option order: ${order[d.variant]}`,
      el("div", { class: "hint" }, `AUC both ${d.auc.both.toFixed(4)}, listed ${d.auc.fwd.toFixed(4)}, reversed ${d.auc.rev.toFixed(4)}; ` +
        `both orders are kept only if they gain ≥ 0.005`)),
    el("div", { style: "margin-top:6px" }, `Thresholds: review ≥ ${d.thresholds.review}, decline ≥ ${d.thresholds.decline}`,
      el("div", { class: "hint" }, "picked on dev windows by cost, before any test window ran")));
}

function trainingCard(t) {
  return el("section", { class: "panel bar-card", style: "margin-bottom:12px" }, el("h3", {}, "Training"),
    el("div", {}, `${t.steps_done} of ${t.steps} batches${t.stopped_early ? " (stopped early)" : ""}`,
      el("div", { class: "hint" }, "batches of 8; the step count was fixed from a smoke run to fit a time cap")),
    el("div", { style: "margin-top:6px" }, `${(t.train_s / 60).toFixed(1)} min`,
      el("div", { class: "hint" }, `${t.step_s_mean.toFixed(2)} s per batch, peak ${t.peak_mem_gib.toFixed(2)} GiB on the iGPU`)),
    el("div", { style: "margin-top:6px" }, `Loss ${t.loss_first100.toFixed(3)} → ${t.loss_last100.toFixed(3)}`,
      el("div", { class: "hint" }, "mean of the first and the last 100 batches")));
}

async function evalPage(id) {
  const { evals } = await getJSON("api/evals");
  const meta = id ? evals.find((e) => e.id === id) : evals[0];
  if (!meta) { app.replaceChildren(el("p", { class: "muted" }, "No eval results in fraud/results/ yet.")); return; }
  const raw = await getJSON(`results/${meta.file.split("/").map(encodeURIComponent).join("/")}`);
  if (raw.setups) return ieeeEval(raw);
  const { r, traceEval } = await withBaselines(raw);
  const { runs } = await getJSON("api/runs");
  const have = new Set(runs.map((x) => x.file));
  document.title = `Eval ${r.eval} · Fraud Console`;
  const setups = Object.keys(r.totals);
  const bars = Object.entries(r.bars).map(([k, b]) => el("section", { class: "panel bar-card" },
    el("h3", {}, BAR_TITLE[k] || k),
    el("div", {}, chip(b.pass ? "pass" : "fail", b.pass ? "PASS" : "FAIL")),
    el("div", { class: "value", style: "margin-top:6px" }, b.value === null ? "–" : SPEED_BARS.has(k) ? `${b.value.toFixed(1)} ms` : `${b.value} of ${r.test_windows.length}`),
    el("div", { class: "hint" }, b.rule)));
  const tot = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ...["setup", "mean cost", "mean cost (natural rate)", "recall", "precision", "alert rate", "p50 / p90 ms"]
      .map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
    el("tbody", {}, ...setups.map((s) => { const v = r.totals[s]; const l = v.latency_ms;
      return el("tr", {}, el("td", {}, setupName(s)), el("td", { class: "r" }, money(v.cost_mean)), el("td", { class: "r" }, money(v.cost_natural_mean)),
        el("td", { class: "r" }, fix(v.recall_mean)), el("td", { class: "r" }, fix(v.precision_mean)), el("td", { class: "r" }, pct(v.alert_rate_mean, 1)),
        el("td", { class: "r" }, l ? `${l.p50.toFixed(0)} / ${l.p90.toFixed(0)}` : "–")); })));
  const perWin = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, el("th", {}, "window"), ...setups.map((s) => el("th", { class: "r" }, s)))),
    el("tbody", {}, ...r.test_windows.map((w) => el("tr", {}, el("td", {}, w), ...setups.map((s) => {
      const v = r.windows[s] && r.windows[s][w];
      const f = `${traceEval(s)}_eval_${s}_${w}.jsonl`;
      return el("td", { class: "r" }, !v ? "–" : have.has(f) ? el("a", { href: `#/run/${encodeURIComponent(f)}` }, money(v.cost)) : money(v.cost));
    })))));
  const paired = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ...["pair", "windows", "first cheaper", "second cheaper", "ties", "sign test p"].map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
    el("tbody", {}, ...Object.entries(r.paired).filter(([, v]) => v.windows).map(([k, v]) => el("tr", {}, el("td", {}, k),
      el("td", { class: "r" }, v.windows), el("td", { class: "r" }, v.a_cheaper), el("td", { class: "r" }, v.b_cheaper),
      el("td", { class: "r" }, v.ties), el("td", { class: "r" }, v.sign_test_p.toPrecision(2))))));
  const cal = Object.entries(r.totals).filter(([, v]) => v.calibration);
  app.replaceChildren(
    el("h1", {}, `Eval ${r.eval}`),
    el("p", { class: "hint", style: "margin-top:-6px" }, evalWhat(r.script)),
    el("p", { class: "intro" }, `PaySim (simulated): ${r.test_windows.length} held-out test windows, run once per setup. Pre-registered in the docstring of ${r.script || "fraud/eval.py"} before any ` +
      `test window ran. Complete: ${r.complete ? "yes" : "no"}. Write-up with labelled claims: ` +
      `${WRITE_UP[r.script] || "fraud/RESULTS.md"}.` +
      (r.phase5_eval ? ` The other setups are the committed results of eval ${r.phase5_eval} on the same windows.` : "")),
    ...(r.training ? [trainingCard(r.training)] : []),
    ...(r.dev ? [devGate(r.dev)] : []),
    el("div", { class: "cards" }, ...bars),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Cost per window"),
      evalChart({ ...r, means: Object.fromEntries(setups.map((s) => [s, r.totals[s].cost_mean])) })),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Setups (means over the test windows)"), el("div", { class: "scroll-x" }, tot)),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Paired window counts"), el("div", { class: "scroll-x" }, paired)),
    cal.length ? el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Model scores (uncalibrated)"),
      el("div", { class: "scroll-x" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["setup", "decisions", "top score mean", "top option right", "ECE", "outside mass"].map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
        el("tbody", {}, ...cal.map(([s, v]) => { const c = v.calibration; return el("tr", {}, el("td", {}, s), el("td", { class: "r" }, c.decisions),
          el("td", { class: "r" }, fix(c.mean_top_score, 3)), el("td", { class: "r" }, pct(c.top_right_rate, 1)), el("td", { class: "r" }, fix(c.ece, 3)),
          el("td", { class: "r" }, fix(c.outside_mass_mean, 4))); })))),
      el("p", { class: "hint" }, "“Right” means legit → approve, fraud → review or decline. ECE compares the top score with how often the top option was right.")) : null,
    el("details", { class: "panel more", style: "margin-top:12px" }, el("summary", {}, "Cost per window, every setup"),
      el("p", { class: "hint" }, "A linked cost opens that window's recorded run (eval traces stay local)."), el("div", { class: "scroll-x" }, perWin)));
}

function answers(r) {  // the pre-registered questions in order, then what was only reported
  const rows = [];
  for (const [k, v] of Object.entries(r.verdict || {})) (Array.isArray(v) ? v : [v]).forEach((p) => rows.push({ ...p, q: k }));
  return rows;
}

// Each answer as a column: its mean paired difference as a point, its 95% interval as a vertical bar, USD on the y axis and a dashed
// zero line across. rows: {label, sub, tag, mean, lo, hi, dist}. title names the panel; each panel has its own scale.
function forest(rows, title) {
  const lo = Math.min(0, ...rows.map((r) => r.lo)), hi = Math.max(0, ...rows.map((r) => r.hi));
  const step = niceStep(hi - lo, 4);
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
  const pos = (v) => `calc(8px + ${((b - v) / (b - a)).toFixed(4)} * (100% - 16px))`;  // top offset inside a plot, 8 px clear at each end
  const ticks = []; for (let t = a; t <= b + 1e-9; t += step) ticks.push(Math.round(t));
  const lines = () => ticks.map((t) => el("span", { class: t === 0 ? "vf-zero" : "vf-gridline", style: `top:${pos(t)}` }));
  return el("div", { class: "vf", style: `flex:${rows.length} 1 ${56 + rows.length * 96}px` },
    title ? el("h3", {}, title) : null,
    el("div", { class: "scroll-x" }, el("div", { class: "vf-grid", style: `grid-template-columns:56px repeat(${rows.length}, minmax(96px, 1fr))` },
      el("div", { class: "vf-axis", "aria-hidden": "true" }, ...ticks.map((t) => el("span", { style: `top:${pos(t)}` }, signed(t)))),
      ...rows.map((r) => el("div", { class: `vf-col${r.dist ? " yes" : ""}` },
        el("div", { class: "vf-plot", title: `${r.label} ${r.sub}: ${signed(r.mean)} USD, 95% CI ${signed(r.lo)} to ${signed(r.hi)}` },
          ...lines(),
          el("span", { class: "vf-ci", style: `top:${pos(r.hi)};height:calc(${((r.hi - r.lo) / (b - a)).toFixed(4)} * (100% - 16px))` }),
          el("span", { class: "vf-dot", style: `top:${pos(r.mean)}` })),
        el("div", { class: "vf-val" }, signed(r.mean), el("span", { class: "hint" }, r.dist ? "excludes 0" : "includes 0")),
        el("div", { class: "vf-name" }, r.label, el("span", { class: "hint" }, r.sub), r.tag ? el("span", { class: "vf-tag" }, r.tag) : null))))));
}

// One comparison of two setups in the order its question was registered: the first setup (a) minus the second (b), whichever
// came out cheaper. The eval files store mean_diff as the cheaper side minus the dearer one, so it is never positive; this turns it
// back, as slides/build_data.py diff() does for the deck. Every comparison on an eval page goes through here.
function compare(p) {
  const flip = Boolean(p.better) && p.better !== p.a;
  const [lo, hi] = p.mean_diff_ci95;
  return { first: p.a, second: p.b, mean: flip ? -p.mean_diff : p.mean_diff, lo: flip ? -hi : lo, hi: flip ? -lo : hi,
    firstCheaper: flip ? p.dearer : p.cheaper, firstDearer: flip ? p.cheaper : p.dearer, windows: p.cheaper + p.dearer + (p.ties || 0),
    dist: Boolean(p.distinguishable) };
}
const questionTag = (q) => q.replace(/_/g, " ");
// The comparison as one plain sentence: the answer first, the statistics after it (GRADE's summary-of-findings order).
// Returns nodes; each setup's name is quoted, with its full name and id on hover.
function sentence(c, unit) {
  const who = (s) => el("span", { title: `${setupName(s)} (${s})` }, `“${shortName(s)}”`);
  const bound = (v) => { const r = Math.round(v); return r < 0 ? `${money(-r)} less` : r > 0 ? `${money(r)} more` : "0"; };
  const m = Math.round(c.mean);
  const head = m === 0 ? [who(c.first), ` cost the same per ${unit} as `, who(c.second), " to the dollar"]
    : [who(c.first), ` cost ${money(Math.abs(m))} USD ${m < 0 ? "less" : "more"} per ${unit} than `, who(c.second)];
  return [...head, `, ${c.dist || m === 0 ? "and" : "but"} the 95% interval of the difference runs from ${bound(c.lo)} to ${bound(c.hi)}, so the two are ${c.dist ? "" : "not "}distinguishable.`];
}
function sentences(qs, unit) {  // every answer as a sentence, in registered order
  return el("ol", { class: "said" }, ...qs.map((p) => el("li", {}, el("span", { class: "vf-tag" }, questionTag(p.q)), " ", ...sentence(compare(p), unit))));
}

function verdictBanner(p, unit) {  // the first registered question, from the data
  const c = compare(p);
  return el("div", { class: `banner${c.dist ? " good" : ""}` },
    el("div", { class: "hint" }, `The first registered question (${questionTag(p.q)})`),
    el("div", { class: "big" }, ...sentence(c, unit)),
    el("div", {}, `Paired difference ${signed(c.mean)} USD per ${unit}, the first minus the second; 95% CI ${signed(c.lo)} to ${signed(c.hi)}. ` +
      `The first was cheaper in ${c.firstCheaper} of ${c.windows} ${unit}s.`),
    el("div", { class: "hint" }, "Distinguishable means the 95% window-bootstrap interval of the mean paired difference excludes 0."));
}

// ---------------------------------------------------------------- cost and speed (the final test)

// The final test timed every language-model setup and not logreg; fraud/time_logreg.py timed it afterwards, one payment at a
// time on a laptop CPU. Different hardware from the model's RTX 3060 Ti, and the chart says so.
const LOGREG_TIMING = "results/time_logreg/20261112-095810.json";

// Each timed setup: its p90 per payment and mean cost. Ideal = under the registered latency bar and not distinguishably dearer
// than the cheapest setup, by the paired comparisons the eval already has. Approve-all has no time and isn't drawn.
function speedPoints(r, timing, qs) {
  const pts = Object.entries(r.setups).filter(([s, v]) => (v.latency_ms && v.latency_ms.p90) || s === timing.setup)
    .map(([s, v]) => ({ s, cost: v.mean_cost, ms: s === timing.setup ? timing.latency_ms.p90 : v.latency_ms.p90, cpu: s === timing.setup }));
  const cheapest = pts.reduce((a, b) => (b.cost < a.cost ? b : a));
  for (const p of pts) {
    const q = qs.find((x) => new Set([x.a, x.b, p.s, cheapest.s]).size === 2 && p !== cheapest);
    p.vsCheapest = q ? compare(q) : null;
    p.ideal = p.ms <= r.latency_bar_ms && (p === cheapest || Boolean(q && !q.distinguishable));
  }
  return { pts, cheapest };
}

function speedChart(pts, bar) {
  const W = 720, H = 300, L = 62, R = 210, T = 12, B = 42, plotW = W - L - R, plotH = H - T - B;
  const lo = Math.floor(Math.log10(Math.min(...pts.map((p) => p.ms)))), hi = Math.ceil(Math.log10(Math.max(bar, ...pts.map((p) => p.ms))));
  const x = (v) => L + plotW * (Math.log10(v) - lo) / (hi - lo);
  const costs = pts.map((p) => p.cost), step = niceStep(Math.max(...costs) - Math.min(...costs), 4);
  const a = Math.floor(Math.min(...costs) / step) * step - step / 2, b = Math.ceil(Math.max(...costs) / step) * step + step / 2;
  const y = (v) => T + plotH * (v - a) / (b - a);  // cheaper at the top
  const inn = pts.filter((p) => p.ideal), out = pts.filter((p) => !p.ideal && p.ms <= bar);
  // The corner's lower edge sits between the dearest setup that passes and the cheapest timed one that doesn't: not a threshold.
  const edge = out.length ? (Math.max(...inn.map((p) => p.cost)) + Math.min(...out.map((p) => p.cost))) / 2 : b;
  const svg = svgEl("svg", { class: "chart speedchart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Mean cost per window against p90 time per payment, log scale, for ${pts.map((p) => shortName(p.s)).join(", ")}. Shaded: fast and not distinguishably dearer than the cheapest.` });
  svg.append(svgEl("rect", { class: "ideal", x: L, y: T, width: x(bar) - L, height: y(edge) - T }));
  const tag = svgEl("text", { class: "ideal-tag", x: L + 8, y: T + 16 }); tag.textContent = "ideal: fast and not distinguishably dearer than the cheapest";
  svg.append(tag);
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) {
    svg.append(svgEl("line", { x1: L, x2: L + plotW, y1: y(t), y2: y(t), stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = svgEl("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }); tx.textContent = money(t); svg.append(tx);
  }
  for (let p = lo; p <= hi; p++) {
    svg.append(svgEl("line", { x1: x(10 ** p), x2: x(10 ** p), y1: T, y2: T + plotH, stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = svgEl("text", { x: x(10 ** p), y: H - 24, "text-anchor": "middle" }); tx.textContent = `${10 ** p >= 1 ? 10 ** p : (10 ** p).toFixed(-p)} ms`; svg.append(tx);
  }
  const xl = svgEl("text", { x: L + plotW / 2, y: H - 4, "text-anchor": "middle", class: "axis-name" }); xl.textContent = "p90 time to decide one payment (log scale): faster to the left"; svg.append(xl);
  const yl = svgEl("text", { x: 14, y: T + plotH / 2, "text-anchor": "middle", class: "axis-name", transform: `rotate(-90 14 ${T + plotH / 2})` }); yl.textContent = "mean cost per window, USD"; svg.append(yl);
  svg.append(svgEl("line", { class: "bar-line", x1: x(bar), x2: x(bar), y1: T, y2: T + plotH }));
  const bt = svgEl("text", { x: x(bar) + 4, y: T + plotH - 6, class: "bar-tag" }); bt.textContent = `${bar} ms bar`; svg.append(bt);
  // A label sits right of its dot unless it would cross the bar; those go to a column right of the plot, on leader lines.
  // Labels in one place are spread so the setups that share a time don't overlap.
  const col = L + plotW + 12;
  const placed = [...pts].sort((p, q) => p.cost - q.cost).map((p) => {
    const cx = x(p.ms), lx = cx + 16 + 170 > x(bar) && cx < x(bar) && cx + 16 + 170 > L + plotW * 0.4 ? col : cx + 16;
    return { p, cx, cy: y(p.cost), lx, ly: y(p.cost) };
  });
  for (let i = 1; i < placed.length; i++) {
    for (const o of placed.slice(0, i)) if (o.lx === placed[i].lx && placed[i].ly - o.ly < 32) placed[i].ly = o.ly + 32;
  }
  for (const { p, cx, cy, lx, ly } of placed) {
    if (lx === col) svg.append(svgEl("polyline", { class: "sp-lead", points: `${cx + 7},${cy} ${col - 18},${ly} ${col - 4},${ly}` }));
    svg.append(svgEl("circle", { class: `sp-dot${p.ideal ? " in" : ""}`, cx, cy, r: p.cpu ? 6 : 5 }));
    const t = svgEl("text", { x: lx, y: ly + 4, class: p.cpu ? "sp-lab strong" : "sp-lab" });
    t.textContent = shortName(p.s);
    const t2 = svgEl("tspan", { x: lx, dy: 15, class: "sp-sub" }); t2.textContent = `${money(p.cost)} USD · ${ms(p.ms)}${p.cpu ? ", laptop CPU" : ""}`;
    t.append(t2); svg.append(t);
  }
  return svg;
}

// The chart's answer as one sentence, before the chart. lr: logreg's point; dist: whether its cost is distinguishable from the
// cheapest setup's (null if it is the cheapest, undefined if never compared). The deck says the same (slides/build_data.py).
function speedSentence(lr, fastestLm, cheapest, dist) {
  const who = (s) => el("span", { title: `${setupName(s)} (${s})` }, `“${shortName(s)}”`);
  const times = Number((fastestLm / lr.ms).toPrecision(2));
  const tail = dist === null ? ["and it is the cheapest setup."] : dist === undefined ? ["and it was not compared with the cheapest setup."]
    : [`and its cost is ${dist ? "distinguishably dearer than" : "not distinguishable from"} the cheapest setup's, `, who(cheapest), "."];
  return [who(lr.s), ` decides one payment in ${ms(lr.ms)} at p90, about ${money(times)} times faster than the language model's ${ms(fastestLm)}, `, ...tail];
}

function speedPanel(r, timing, qs) {
  const { pts, cheapest } = speedPoints(r, timing, qs);
  const lr = pts.find((p) => p.cpu);
  const fastestLm = Math.min(...pts.filter((p) => !p.cpu).map((p) => p.ms));
  const dist = lr === cheapest ? null : lr.vsCheapest ? lr.vsCheapest.dist : undefined;
  return el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Cost and speed"),
    el("p", { class: "said-one" }, ...speedSentence(lr, fastestLm, cheapest.s, dist)),
    el("div", { class: "scroll-x" }, speedChart(pts, r.latency_bar_ms)),
    el("p", { class: "hint" }, `Shaded: under the registered ${r.latency_bar_ms} ms bar and not distinguishably dearer than the cheapest setup, by the paired comparisons above. ` +
      "Its lower edge sits halfway between the dearest setup that passes and the cheapest that doesn't; the edge itself is not a threshold. Approving everything has no model to time and isn't drawn."),
    el("p", { class: "hint" }, `Different hardware: the language model was timed during the final test on an RTX 3060 Ti, lone prompts, its forward pass only. ` +
      `Logreg was timed afterwards on a laptop CPU (${timing.machine.cpu}), one payment at a time, features included, on the 30 dev windows, not FINAL rows. Source: fraud/${LOGREG_TIMING.replace(/^results/, "results")}, written by ${timing.script}.`));
}

async function ieeeEval(r) {  // IEEE-CIS evals (eval_ieee*.py and the phases after them): one shape, a 10 USD review
  const { runs } = await getJSON("api/runs");
  const have = new Set(runs.map((x) => x.file));
  document.title = `Eval ${r.eval} · Fraud Console`;
  // Phase 11's file lists its dev windows too; a setup's costs are for the test windows only, in file order.
  const scored = Object.values(r.setups)[0].windows.length;
  const wins = Object.keys(r.windows).filter((w, _, all) => all.length === scored || !/^dev\d+$/.test(w)).slice(-scored);
  const dev = r.test_rows === undefined;
  const unit = dev ? "chain" : "window";
  const order = Object.entries(r.setups).sort((a, b) => a[1].mean_cost - b[1].mean_cost);
  const name = (s) => shortName(s);
  const nameCell = (s) => el("td", { title: `${setupName(s)} (${s})` }, name(s));
  const count = (x) => (x === undefined || x === null ? "–" : x.toLocaleString("en-US"));
  let after = null;  // the final test's read-outs, asked for after its results were seen
  let timing = null;  // logreg timed one payment at a time, after the final test (fraud/time_logreg.py)
  if (r.script === "fraud/eval_final.py") {
    try { after = { readout: await getJSON("results/eval_final/rich_cust_readout.json"), extras: await getJSON("results/eval_final/extras.json") }; }
    catch (err) { after = null; }
    if (after && after.readout.eval !== r.eval) after = null;
    try { timing = await getJSON(LOGREG_TIMING); } catch (err) { timing = null; }
  }
  const cheapest = order.filter(([s]) => s !== "ieee-approve-all").slice(0, 4).map(([s]) => s);
  const chart = { test_windows: wins, windows: {}, means: {}, linear: true, label: name, unit: dev ? "held-out dev chain" : "",
    chart_series: [...cheapest, "ieee-approve-all"].filter((s) => r.setups[s]).map((s, i) => [s, `var(--series-${i + 1})`]) };
  chart.extra_series = order.map(([s]) => s).filter((s) => !chart.chart_series.some(([t]) => t === s));  // every other setup, cheapest first, off at the start
  for (const [s] of order) {
    chart.windows[s] = {}; chart.means[s] = r.setups[s].mean_cost;
    wins.forEach((w, i) => { chart.windows[s][w] = { cost: r.setups[s].windows[i] }; });
  }
  const head = (cols, opt = []) => el("thead", {}, el("tr", {}, ...cols.map((t, i) => el("th", { class: `${i ? "r" : ""}${opt.includes(i) ? " opt" : ""}` }, t))));
  // Cost and customers by default; AUC, recall, precision, reviews and declines one click away.
  const tot = el("table", { class: "data" },
    head(["setup", "mean cost", "95% CI", "good customers declined", "fraud let through", "AUC", "recall", "precision", "reviews", "declines"], [5, 6, 7, 8, 9]),
    el("tbody", {}, ...order.map(([s, v]) => el("tr", {}, nameCell(s), el("td", { class: "r" }, money(v.mean_cost)),
      el("td", { class: "r" }, `${money(v.cost_ci95[0])}–${money(v.cost_ci95[1])}`), el("td", { class: "r" }, count(v.good_customers_declined)), el("td", { class: "r" }, money(v.fraud_let_through)),
      el("td", { class: "r opt" }, v.auc === null ? "–" : fix(v.auc, 3)), el("td", { class: "r opt" }, fix(v.recall, 3)), el("td", { class: "r opt" }, fix(v.precision, 3)),
      el("td", { class: "r opt" }, count(v.reviews)), el("td", { class: "r opt" }, count(v.declines))))));
  const moreBtn = el("button", { "aria-expanded": "false", onclick: (e) => {
    const on = tot.classList.toggle("show-more");
    e.currentTarget.setAttribute("aria-expanded", String(on));
    e.currentTarget.textContent = on ? "Fewer columns" : "More columns: AUC, recall, precision, reviews, declines";
  } }, "More columns: AUC, recall, precision, reviews, declines");
  const answerTable = (rows) => el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ...["first", "second", "question"].map((t) => el("th", {}, t)), ...[`${unit}s first cheaper / dearer`, "first minus second, USD", "95% CI", "distinguishable"].map((t) => el("th", { class: "r" }, t)))),
    el("tbody", {}, ...rows.map((p) => { const c = compare(p);
      return el("tr", {}, nameCell(c.first), nameCell(c.second), el("td", {}, questionTag(p.q)),
        el("td", { class: "r" }, `${c.firstCheaper} / ${c.firstDearer}`), el("td", { class: "r" }, signed(c.mean)), el("td", { class: "r" }, `${signed(c.lo)} to ${signed(c.hi)}`),
        el("td", { class: "r" }, chip(c.dist ? "pass" : "fail", c.dist ? "yes" : "no"))); })));
  const qs = answers(r);
  const ans = answerTable(qs);
  // The picture of the answers: comparisons against approve-all are about ten times larger, so they get their own panel and scale.
  const toRow = (p) => { const c = compare(p);
    return { label: name(c.first), sub: `minus ${name(c.second)}`, tag: questionTag(p.q), mean: c.mean, lo: c.lo, hi: c.hi, dist: c.dist }; };
  const isAll = (p) => p.a === "ieee-approve-all" || p.b === "ieee-approve-all";
  const main = qs.filter((p) => !isAll(p)).map(toRow), vsAll = qs.filter(isAll).map(toRow);
  const readoutQs = after ? after.readout.against.map((p) => ({ ...p, q: "read-out" })) : [];
  const told = after ? after.readout.told_10_usd : null;
  const afterPanel = !after ? [] : [el("section", { class: "panel", style: "margin-top:12px" },
    el("h2", {}, "After the test: the owner's verdict, and a read-out ", chip("plain", "read-out, not a test")),
    el("p", {}, `The owner judged the recommended setup “really not a decision model doing its grunt work”: its rank correlation with the tabular score it reads is ${fix(after.extras.signal_rank_corr, 3)} on these rows. ` +
      "So the setup that meets the owner's constraint is d1-Omni-2B with no tabular score. The registered result above stands as it came out."),
    el("p", { class: "hint" }, "The customer line on that setup was asked for after every result here was seen. It uses the recorded scores, no model ran, and it answers no registered question. Intervals are for scale. Source: fraud/results/eval_final/rich_cust_readout.json and extras.json; fraud/RESULTS-FINAL.md, After the test."),
    el("div", { class: "scroll-x" }, el("table", { class: "data" },
      head(["setup", "mean cost", "95% CI", "good customers declined", "declined or reviewed", "frauds stopped", "fraud let through", "reviews"]),
      el("tbody", {}, ...["final-rich-cust", "final-rich-amt", "final-logreg-amt", "final-signal-cust"].filter((s) => after.readout.table[s]).map((s) => { const v = after.readout.table[s];
        return el("tr", {}, nameCell(s), el("td", { class: "r" }, money(v.mean_cost)), el("td", { class: "r" }, `${money(v.cost_ci95[0])}–${money(v.cost_ci95[1])}`),
          el("td", { class: "r" }, count(v.good_customers_declined)), el("td", { class: "r" }, count(v.good_declined_or_reviewed)), el("td", { class: "r" }, `${count(v.frauds_stopped)} of ${count(v.frauds)}`),
          el("td", { class: "r" }, money(v.fraud_usd_let_through)), el("td", { class: "r" }, count(v.reviews))); })))),
    sentences(readoutQs, unit),
    el("div", { class: "scroll-x", style: "margin-top:10px" }, answerTable(readoutQs)),
    // told_10_usd.against_baseline is already rich-cust minus the baseline (fraud/dev/final_rich_cust.py), so it needs no flip.
    el("p", { class: "hint" }, `If a declined good customer truly cost 10 USD more, an assumption nobody measured: `,
      ...sentence({ first: "final-rich-cust", second: "final-logreg-amt", mean: told.against_baseline, lo: told.ci[0], hi: told.ci[1], dist: told.ci[0] > 0 || told.ci[1] < 0 }, unit)))];
  const perWin = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, el("th", {}, dev ? "chain" : "window"), ...order.map(([s]) => el("th", { class: "r", title: `${setupName(s)} (${s})` }, name(s))))),
    el("tbody", {}, ...wins.map((w, i) => el("tr", {}, el("td", {}, w), ...order.map(([s, v]) => {
      const f = `${r.eval}_eval_${s}_${w}.jsonl`;
      return el("td", { class: "r" }, have.has(f) ? el("a", { href: `#/run/${encodeURIComponent(f)}` }, money(v.windows[i])) : money(v.windows[i]));
    })))));
  const sens = r.sensitivity ? Object.entries(r.sensitivity) : [];
  const set_names = sens.length ? Object.keys(sens[0][1].mean_cost) : [];
  const facts = [
    factCard("Data",
      fact("Data set", "IEEE-CIS", "Real card payments (Vesta, Kaggle), no type filter"),
      dev ? fact("Split", "Dev only", `No test row. Reads here are flattered: lambda was picked with these rows in. Stated in the docstring of ${r.script}`)
        : fact("Split", "Held-out test", `Run once per setup. Pre-registered in the docstring of ${r.script}`),
      fact(dev ? "Chains" : "Windows", wins.length, "1,000 rows each, in arrival order"),
      dev ? null : fact("Test rows", money(r.test_rows), `${count(r.test_fraud)} frauds`)),
    factCard("Assumed costs",
      fact("Review", `${money(r.costs.review_fee)} USD`, "An analyst finds the truth. An assumption, not measured"),
      fact("Wrong decline", `${pct(r.costs.friction_rate)} of the amount`, `At least ${money(r.costs.friction_min)} USD`),
      fact("Review budget", `${pct(r.costs.review_share)} of a ${unit}`, "Past it, a review becomes an approve")),
    r.recommended ? factCard("Recommended by the rule fixed before the run",
      fact("Setup", name(r.recommended), "Chosen for customers and the owner's constraint. Its dollars are not distinguishable from the baseline's; see the answers" +
        (after ? ". After the test the owner rejected it, because it mostly follows the tabular score it reads; see After the test" : ""))) : null,
  ];
  const notes = noteCard("Read with care", [
    "Model scores are uncalibrated, never probabilities.",
    "Every number here is for the assumed costs above.",
    `Don't compare the costs of different evals: they use different ${unit}s.`,
    "Only the questions named in the pre-registration are questions; “reported” ones are not corrected for multiple comparisons.",
    `Write-up with labelled claims: ${WRITE_UP[r.script] || "fraud/PROGRESS.md"}.`]);
  app.replaceChildren(
    el("h1", {}, `Eval ${r.eval}`),
    el("p", { class: "hint", style: "margin-top:-6px" }, evalWhat(r.script)),
    qs.length ? verdictBanner(qs[0], unit) : null,
    el("div", { class: "cards", style: "margin-top:12px" }, ...facts, notes),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Answers"),
      qs.length ? sentences(qs, unit) : null,
      el("p", { class: "hint" }, `Mean paired difference in USD per ${unit}: the first setup minus the second, in the order each question was registered, with its 95% window-bootstrap interval. Below the dashed 0 line: the first is cheaper. Filled: the interval excludes 0.`),
      el("div", { class: "vf-row" }, main.length ? forest(main, vsAll.length ? "Between setups" : null) : null,
        vsAll.length ? forest(vsAll, "Against approving everything (own scale)") : null),
      el("details", { class: "more", style: "margin-top:10px" }, el("summary", {}, "The answers as a table"), el("div", { class: "scroll-x" }, ans))),
    ...afterPanel,
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, dev ? "Cost per dev chain" : "Cost per window"), evalChart(chart)),
    timing && r.latency_bar_ms && r.setups[timing.setup] ? speedPanel(r, timing, qs) : null,
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Setups, cheapest first"),
      el("p", { class: "hint" }, `Mean cost in USD per ${unit} of 1,000 rows. Counts are over all ${wins.length} ${unit}s. Recall and precision count a review or a decline as an alert. Hover a name for the full one.`),
      el("div", { class: "filters" }, moreBtn), el("div", { class: "scroll-x" }, tot)),
    ...(sens.length ? [el("details", { class: "panel more", style: "margin-top:12px" }, el("summary", {}, "Other review fees"), el("p", { class: "hint" }, "Mean cost in USD per window. Lines or lambda re-picked on dev. No bar and no interval. The cost map (Costs, at the top) has Phase 16's intervals."),
      el("div", { class: "scroll-x" }, el("table", { class: "data" }, el("thead", {}, el("tr", {}, el("th", {}, "cost set"), ...set_names.map((s) => el("th", { class: "r", title: `${setupName(s)} (${s})` }, name(s))))),
        el("tbody", {}, ...sens.map(([k, v]) => el("tr", {}, el("td", {}, k), ...set_names.map((s) => el("td", { class: "r" }, money(v.mean_cost[s])))))))))] : []),
    el("details", { class: "panel more", style: "margin-top:12px" }, el("summary", {}, dev ? "Cost per dev chain, every setup" : "Cost per window, every setup"),
      el("p", { class: "hint" }, "In USD. A linked cost opens that window's recorded run. Eval traces stay on the machine that made them and are not in git, so most costs here have no link."), el("div", { class: "scroll-x" }, perWin)),
    setupLegend(order.map(([s]) => s)));
}

// ---------------------------------------------------------------- the cost map (Phase 16)

// fraud/results/eval_cost_stress/<run>.json, written by fraud/eval_cost_stress.py: Phase 14's recorded scores on its 20 test windows,
// re-priced under other costs. No model ran. diff is the rule MINUS two lines (a plus means the rule costs more): the first
// minus the second, the same sign compare() gives the eval pages' answers.
const pctTight = (v) => pct(v, Math.round(v * 1000) % 10 ? 1 : 0);  // 7.5%, 0.5%, 10%
const COST_FAMILIES = {
  "review fee": { key: "review_fee", label: "Review fee", fmt: (v) => `${money(v)} USD`, what: "what one review costs" },
  "review share": { key: "review_share", label: "Review budget", fmt: (v) => pctTight(v), what: "the share of a window that can be reviewed" },
  "friction rate": { key: "friction_rate", label: "Wrong decline, % of amount", fmt: (v) => pctTight(v), what: "what declining a good customer costs, as a share of the amount" },
  "friction minimum": { key: "friction_min", label: "Wrong decline, at least", fmt: (v) => `${fix(v, v % 1 ? 1 : 0)} USD`, what: "the least a wrongly declined payment costs" },
  "flat per good customer declined": { key: "decline_flat", label: "Flat price per declined good customer", fmt: (v) => `${fix(v, 0)} USD`, what: "a flat price per good customer declined, on top of the 10%" },
  "flat only, no percentage": { key: "decline_flat", label: "Flat price only, no percentage", fmt: (v) => `${fix(v, 0)} USD`, what: "a flat price per good customer declined, with no percentage and no minimum" },
};
const FAMILY_ORDER = ["review fee", "flat per good customer declined", "flat only, no percentage", "friction rate", "friction minimum", "review share"];
let costView = { family: "review fee", scorer: "logreg", i: null };  // kept while you leave the page and come back

function isBase(row, base) {
  return Object.entries(base).every(([k, v]) => row.costs[k] === v) && !row.costs.decline_flat;
}

function costCard(title, hint, x, repicked) {
  const verdict = x.distinguishable ? (x.rule_wins ? chip("pass", "rule cheaper, distinguishable") : chip("fail", "two lines cheaper, distinguishable")) : chip("plain", "not distinguishable");
  return el("section", { class: "panel fact-card" }, el("h3", {}, title), el("div", { class: "hint", style: "margin-bottom:6px" }, hint),
    fact("Rule minus two lines", `${signed(x.diff)} USD`, `Per window. 95% CI ${signed(x.ci[0])} to ${signed(x.ci[1])}`),
    el("div", { class: "fact" }, el("span", {}, "Verdict"), el("span", { class: "v" }, verdict)),
    fact("The rule", `${money(x.rule_cost)} USD`, "Mean cost per window"),
    fact("Two lines", `${money(x.thresholds_cost)} USD`, "Mean cost per window"),
    fact("Windows the rule is cheaper", `${x.rule_cheaper_windows} of 20`),
    fact("Good customers declined", `${count0(x.rule_good_declined)} vs ${count0(x.thresholds_good_declined)}`, "The rule vs two lines, over the 20 windows"),
    fact("Frauds stopped", `${count0(x.rule_frauds_stopped)} vs ${count0(x.thresholds_frauds_stopped)}`, `The rule vs two lines; ${money(x.rule_fraud_usd_stopped)} vs ${money(x.thresholds_fraud_usd_stopped)} USD of fraud`),
    repicked ? fact("Re-picked on dev", `lambda ${x.lambda}`, `Two lines at ${fix(x.thresholds[0], 4)} / ${fix(x.thresholds[1], 4)} on the uncalibrated score`) : null);
}
const count0 = (x) => (x === undefined || x === null ? "–" : x.toLocaleString("en-US"));

// The difference and its interval at every tested value, both views. Values are evenly spaced and labelled, since they aren't in dollars.
function costChart(rows, scorer, sel, baseI, fam, onPick) {
  const W = 720, H = 300, L = 60, R = 12, T = 14, B = 44;
  const views = [["deployed", "var(--series-1)", "As deployed"], ["repicked", "var(--series-2)", "Re-picked"]];
  const all = rows.flatMap((r) => views.flatMap(([v]) => r[`${scorer}_${v}`].ci));
  const lo0 = Math.min(0, ...all), hi0 = Math.max(0, ...all), step = niceStep(hi0 - lo0, 5);
  const lo = Math.floor(lo0 / step) * step, hi = Math.ceil(hi0 / step) * step;
  const band = (W - L - R) / rows.length;
  const x = (i) => L + band * (i + 0.5);
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `The rule minus two lines, USD per window, with 95% intervals, as deployed and re-picked, at each tested ${fam.label.toLowerCase()}` });
  svg.append(svgEl("rect", { x: L + band * sel, y: T, width: band, height: H - T - B, fill: "var(--surface-2)" }));
  for (let t = lo; t <= hi + 1e-9; t += step) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: t === 0 ? "var(--ink-2)" : "var(--grid)", "stroke-width": t === 0 ? 1.2 : 1, "stroke-dasharray": t === 0 ? "4 3" : "" }));
    const tx = svgEl("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }); tx.textContent = signed(t); svg.append(tx);
  }
  const zl = svgEl("text", { x: L + 4, y: y(0) - 5 }); zl.textContent = "0: the two cost the same"; svg.append(zl);
  rows.forEach((r, i) => {
    const t = svgEl("text", { x: x(i), y: H - B + 16, "text-anchor": "middle", style: i === sel ? "fill:var(--ink);font-weight:600" : "" });
    t.textContent = fam.fmt(r.value).replace(" USD", ""); svg.append(t);
    if (i === baseI) { const b = svgEl("text", { x: x(i), y: H - B + 30, "text-anchor": "middle", style: "fill:var(--ink-2)" }); b.textContent = "base"; svg.append(b); }
    views.forEach(([v, color], j) => {
      const d = r[`${scorer}_${v}`], cx = x(i) + (j ? 5 : -5);
      svg.append(svgEl("line", { x1: cx, x2: cx, y1: y(d.ci[0]), y2: y(d.ci[1]), stroke: color, "stroke-width": 2 }));
      svg.append(svgEl("circle", { cx, cy: y(d.diff), r: 4.5, fill: d.distinguishable ? color : "var(--surface)", stroke: color, "stroke-width": 2 }));
    });
  });
  const unitTxt = svgEl("text", { x: L, y: H - 4 }); unitTxt.textContent = `${fam.label} →${fam.key === "review_fee" || fam.key === "friction_min" || fam.key === "decline_flat" ? " (USD)" : ""}`; svg.append(unitTxt);
  const pick = (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    return Math.max(0, Math.min(rows.length - 1, Math.floor((px - L) / band)));
  };
  svg.addEventListener("mousemove", (e) => {
    const i = pick(e), r = rows[i];
    const line = (v, lab) => { const d = r[`${scorer}_${v}`]; return `${lab}: ${signed(d.diff)} (${signed(d.ci[0])} to ${signed(d.ci[1])})${d.distinguishable ? "" : ", not distinguishable"}`; };
    showTip(e, [fam.fmt(r.value), line("deployed", "As deployed"), line("repicked", "Re-picked")]);
  });
  svg.addEventListener("mouseleave", hideTip);
  svg.addEventListener("click", (e) => onPick(pick(e)));
  svg.style.cursor = "pointer";
  return el("div", {},
    el("div", { class: "legend" }, ...views.map(([, c, lab]) => el("span", {}, el("i", { style: `background:${c}` }), lab)), el("span", {}, "Filled: the interval excludes 0. Hollow: not distinguishable.")),
    svg);
}

function stopsSentence(st, fam, base, view, below, above) {  // one sentence from the file's stops: where the mean stops being lower, and where the interval stops excluding 0
  if (!st) return "";
  const f = (v) => fam.fmt(v);
  const side = (s, dir) => {
    const mean = s.first_not_lower, ci = s.first_not_distinguishably_lower;
    if (mean === null && ci === null) return `${dir}, the rule stays distinguishably cheaper at every value tried`;
    const a = mean === null ? "its mean stays lower at every value tried" : mean === base ? `its mean is already not lower at ${f(mean)}` : `its mean stops being lower at ${f(mean)}`;
    const b = ci === null ? "the interval excludes 0 throughout" : ci === base ? `the interval already includes 0 at ${f(ci)}` : `the interval stops excluding 0 at ${f(ci)}`;
    return `${dir}, ${b} and ${a}`;
  };
  const parts = [above ? side(st.above_base, `going up from ${f(base)}`) : null, below ? side(st.below_base, `going down from ${f(base)}`) : null].filter(Boolean);
  return `${view}: ${parts.join("; ")}.`;
}

async function costsPage() {
  document.title = "Cost map · Fraud Console";
  const { files } = await getJSON("api/costs");
  if (!files.length) { app.replaceChildren(el("p", { class: "muted" }, "No cost map in fraud/results/eval_cost_stress/ yet. Run python -m fraud.eval_cost_stress.")); return; }
  const r = await getJSON(`results/${files[0].split("/").map(encodeURIComponent).join("/")}`);
  const fams = FAMILY_ORDER.filter((f) => r.families[f]).concat(Object.keys(r.families).filter((f) => !FAMILY_ORDER.includes(f) && COST_FAMILIES[f]));
  if (!r.families[costView.family]) costView = { family: fams[0], scorer: "logreg", i: null };
  const famSel = el("select", { id: "cost-family", "aria-label": "Cost assumption" }, ...fams.map((f) => el("option", { value: f, selected: f === costView.family }, COST_FAMILIES[f].label)));
  const scorerBox = el("div", { class: "seg", role: "radiogroup", "aria-label": "Scorer" });
  const slider = el("input", { type: "range", id: "cost-step", min: 0, step: 1, "aria-label": "Tested value" });
  const valueOut = el("output", { for: "cost-step", class: "cost-value" });
  const ticksBox = el("div", { class: "cost-ticks", "aria-hidden": "true" });
  const body = el("div");
  const draw = () => {
    const fam = COST_FAMILIES[costView.family], rows = r.families[costView.family], sc = costView.scorer;
    const baseI = rows.findIndex((x) => isBase(x, r.base_costs));
    if (costView.i === null || costView.i >= rows.length) costView.i = baseI >= 0 ? baseI : 0;
    const i = costView.i, row = rows[i];
    slider.max = rows.length - 1; slider.value = i;
    slider.setAttribute("aria-valuetext", `${fam.label} ${fam.fmt(row.value)}${i === baseI ? ", the base" : ""}`);
    valueOut.textContent = `${fam.label}: ${fam.fmt(row.value)}${i === baseI ? " (the base, Phase 14's costs)" : ""}`;
    ticksBox.replaceChildren(...rows.map((x, k) => el("span", { class: `${k === baseI ? "base" : ""}${k === i ? " on" : ""}${k % 2 ? " odd" : ""}`,
      style: `left:${rows.length > 1 ? (100 * k) / (rows.length - 1) : 50}%;transform:translateX(${k === 0 ? "0" : k === rows.length - 1 ? "-100%" : "-50%"})` },
      fam.fmt(x.value).replace(" USD", ""), k === baseI ? " · base" : "")));
    scorerBox.replaceChildren(...[["logreg", "Logreg", "Phase 16's registered primary"], ["adapter", "The adapter", "Phase 12's adapter, not a final one"]].map(([k, lab, hint]) =>
      el("button", { role: "radio", "aria-checked": String(k === sc), class: k === sc ? "on" : "", onclick: () => { costView.scorer = k; draw(); } }, lab, el("span", { class: "hint" }, hint))));
    const st = (r.stops[costView.family] || {});
    const baseVal = baseI >= 0 ? rows[baseI].value : rows[0].value;
    const tableRows = rows.map((x, k) => { const a = x[`${sc}_deployed`], b = x[`${sc}_repicked`];
      const cell = (d) => el("td", { class: "r" }, `${signed(d.diff)}`, el("div", { class: "hint" }, `${signed(d.ci[0])} to ${signed(d.ci[1])}${d.distinguishable ? "" : ", not dist."}`));
      return el("tr", { class: k === i ? "sel" : "" }, el("td", {}, fam.fmt(x.value), k === baseI ? el("span", { class: "hint" }, " base") : null),
        cell(a), el("td", { class: "r" }, `${a.rule_good_declined} / ${a.thresholds_good_declined}`), cell(b), el("td", { class: "r" }, `${b.rule_good_declined} / ${b.thresholds_good_declined}`), el("td", { class: "r" }, b.lambda)); });
    body.replaceChildren(
      el("div", { class: "cards two", style: "margin-top:12px" },
        costCard("As deployed", "Phase 14's decisions unchanged; only the prices move. The rule isn't told the new cost.", row[`${sc}_deployed`], false),
        costCard("Re-picked", "The rule is told the new cost: lines and lambda re-picked on the 30 dev windows.", row[`${sc}_repicked`], true)),
      el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, `The rule minus two lines, at every tested ${fam.label.toLowerCase()}`),
        costChart(rows, sc, i, baseI, fam, (k) => { costView.i = k; draw(); }),
        el("p", { class: "hint" }, `USD per window, with 95% window-bootstrap intervals. Below 0, the rule is cheaper. The shaded column is the chosen value; click a column to choose it. ${sc === "adapter" ? "Scorer: Phase 12's adapter, not a final one." : "Scorer: logreg, Phase 16's registered primary."}`),
        el("p", {}, stopsSentence(st[`${sc}_deployed`], fam, baseVal, "As deployed", rows.some((x) => x.value < baseVal), rows.some((x) => x.value > baseVal))),
        el("p", {}, stopsSentence(st[`${sc}_repicked`], fam, baseVal, "Re-picked", rows.some((x) => x.value < baseVal), rows.some((x) => x.value > baseVal))),
        baseI < 0 ? el("p", { class: "hint" }, "Phase 14's costs are not in this family: it drops the percentage and the minimum. The sentences above start from its first value.") : null),
      el("details", { class: "panel more", style: "margin-top:12px" }, el("summary", {}, `The whole family as a table (${sc === "adapter" ? "the adapter" : "logreg"})`),
        el("p", { class: "hint" }, "Rule minus two lines, USD per window, with the 95% CI under it. Good customers declined over the 20 windows, the rule / two lines."),
        el("div", { class: "scroll-x" }, el("table", { class: "data" },
          el("thead", {}, el("tr", {}, ...["value", "as deployed", "declined, rule / lines", "re-picked", "declined, rule / lines", "lambda"].map((t, k) => el("th", { class: k ? "r" : "" }, t)))),
          el("tbody", {}, ...tableRows)))));
  };
  famSel.addEventListener("change", () => { costView.family = famSel.value; costView.i = null; draw(); });
  slider.addEventListener("input", () => { costView.i = Number(slider.value); draw(); });
  app.replaceChildren(
    el("h1", {}, "Cost map: where does the rule stop winning?"),
    el("p", { class: "hint", style: "margin-top:-6px" }, `Phase 16: Phase 14's recorded scores re-priced under other costs. Run ${r.run}, scores from eval ${r.scores_from_eval}. No model ran.`),
    el("div", { class: "cards", style: "margin-top:12px" },
      noteCard("Read this first", [
        "Earlier test: Phase 14's 20 test windows, re-priced from recorded scores. Not the final test, and not paired with it.",
        "Many cost sets, not corrected for multiple comparisons: a map, not separate findings.",
        "Every price is an assumption nobody measured.",
        "The rule against two lines on the same scores. It does not compare the language model with logreg."]),
      noteCard("The two views", [
        "As deployed: Phase 14's decisions stay, only the prices move. This is a rule that isn't told its true costs.",
        "Re-picked: the rule and the lines are told the new cost and re-picked on 30 dev windows.",
        `Base: Phase 14's costs, a review at ${money(r.base_costs.review_fee)} USD, a wrong decline at ${pct(r.base_costs.friction_rate)} of the amount (at least ${money(r.base_costs.friction_min)} USD), ${pct(r.base_costs.review_share)} reviewed. The deck and every eval stay at these.`])),
    el("section", { class: "panel cost-controls", style: "margin-top:12px" },
      el("div", { class: "filters" }, el("label", { for: "cost-family" }, "Cost assumption "), famSel, scorerBox),
      el("div", { class: "hint", id: "cost-what" }),
      valueOut, slider, ticksBox),
    body);
  const what = () => { document.getElementById("cost-what").textContent = `Changes one thing from Phase 14's costs: ${COST_FAMILIES[costView.family].what}. The values are the ones Phase 16 tested; the console never re-decides a row, so the slider steps between them.`; };
  famSel.addEventListener("change", what);
  what();
  draw();
}

async function startDemo() {
  const { runs } = await getJSON("api/runs");
  let pool = runs.filter((r) => r.finished && r.model_decisions > 0);
  if (!pool.length) pool = runs.filter((r) => r.finished);
  if (!pool.length) { app.replaceChildren(el("p", { class: "muted" }, "No finished runs to play.")); return; }
  const pinned = pool.filter((r) => r.pinned), rest = pool.filter((r) => !r.pinned);
  for (let i = rest.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; }
  demo = { list: [...pinned, ...rest], k: 0, timer: null };
  const playNext = async () => {
    if (!demo) return;
    const item = demo.list[demo.k % demo.list.length];
    const rp = await openRun(item.file, { onEnd: () => { if (demo) demo.timer = setTimeout(() => { if (demo) { demo.k += 1; playNext(); } }, 3000); } });
    rp.demoBox.replaceChildren(el("span", { class: "demo-badge" }, `Auto-demo ${demo.k % demo.list.length + 1} of ${demo.list.length}`),
      " ", el("button", { onclick: () => { location.hash = `#/run/${encodeURIComponent(item.file)}`; } }, "Stop"));
    rp.play();
  };
  playNext();
}

function stopDemo() { if (demo) { clearTimeout(demo.timer); demo = null; } }

async function route() {
  stopDemo();
  if (current) { current.stop(); current = null; }
  hideTip();
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  try {
    if (h.startsWith("run/")) await openRun(h.slice(4));
    else if (h === "eval") await evalPage(null);
    else if (h.startsWith("eval/")) await evalPage(h.slice(5));
    else if (h === "demo") await startDemo();
    else if (h === "costs") await costsPage();
    else await startPage();
  } catch (err) {
    app.replaceChildren(el("p", {}, `Could not load: ${err.message}`), el("p", {}, el("a", { href: "#/" }, "Back to the runs")));
  }
  window.scrollTo(0, 0);
}

// For checks without a visible tab (requestAnimationFrame does not fire while hidden): frame(ts).
window.frame = (ts) => { if (current) current.frame(ts); };
window.consoleState = () => (current ? { t: current.t, total: current.total, shown: current.shown, playing: current.playing } : null);
window.addEventListener("hashchange", route);
route();
