/* The fraud analyst console: replays recorded traces (fraud/runs/*.jsonl) and shows the eval.
   Replay only: nothing here runs a model. The replay clock is the sum of the recorded decision
   times, so at 1x each transaction appears when its decision finished in the recorded run. */
"use strict";

const app = document.getElementById("app");
const ACTIONS = ["approve", "review", "decline"];
const BY = {
  model: "the model", filter: "type filter (rule)", rules: "rules-only table",
  logreg: "logistic regression", random: "random", all: "approve-all",
};
const SETUP_NAME = {
  hybrid: "hybrid (filter + model)", "hybrid-1.5b": "hybrid (filter + Qwen2.5-1.5B)", rules: "rules-only",
  logreg: "filter + logistic regression", random: "filter + random", "approve-all": "approve-all",
};
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
    const title = `${SETUP_NAME[h.setup] || h.setup} · window ${h.window.id}${model ? ` · ${model}` : ""}`;
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
  parts.push(el("div", { class: "muted" }, `#${r.i + 1} · to a ${r.tx.to} · hour ${r.tx.hour}:00 · step ${r.step} · PaySim row ${r.row}`));
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
      r.orders ? el("div", { class: "orders" },
        `Averaged over two readings. Listed order: ${r.orders[0].map((x) => fix(x)).join(" / ")}. ` +
        `Reversed order (shown in listed order): ${r.orders[1].map((x) => fix(x)).join(" / ")}.`) : null,
      el("dl", { class: "kv", style: "margin-top:6px" },
        el("dt", {}, "Mass outside the labels"), el("dd", {}, fix(r.outside_mass, 4)),
        el("dt", {}, "Prompt tokens"), el("dd", {}, r.prompt_tokens))));
  } else if (r.by === "logreg" && r.scores) {
    parts.push(el("div", { class: "section" }, el("dl", { class: "kv" },
      el("dt", {}, "Logistic regression score (uncalibrated)"), el("dd", {}, fix(r.scores.fraud, 3)))));
  } else if (r.rule) {
    const rule = (h.rules || []).find((x) => x[0] === r.rule);
    parts.push(el("div", { class: "section" }, el("h3", {}, "Rule"), el("div", {}, r.rule),
      rule ? el("div", { class: "hint" }, `From the training steps: ${rule[3]}`) : null));
  }
  parts.push(el("div", { class: "section" }, el("h3", {}, "Signals (earlier rows only)"),
    el("dl", { class: "kv" }, ...SIGNALS.filter(([k]) => k in r.signals).flatMap(([k, label, f]) =>
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
      el("div", { class: "hint" }, `${SETUP_NAME[h.setup] || h.setup} on window ${h.window.id} (${h.window.split}).`));
  const facts = [
    factCard("Window",
      fact("Window", h.window.id, `${h.window.split === "test" ? "Held-out test steps" : "Training-period dev window"}, seed ${h.window.seed}`),
      fact("Steps (hours)", `${h.window.steps[0]}–${h.window.steps[1]}`),
      fact("Transactions", h.window.size, "Sampled in arrival order from the window's hours"),
      fact("Fraud", h.window.fraud, `Enriched; the natural rate in these hours is ${pct(h.window.natural_fraud_rate, 3)}`)),
    factCard("Setup",
      fact("Setup", h.setup, SETUP_NAME[h.setup] || ""),
      model.length ? fact("Model", (b.model || "").split("/").pop() || b.kind, `${b.device_name || b.device || ""}, LM head ${b.head_dtype || "?"}`) : null,
      model.length ? fact("Order debias", h.order_debias ? "on" : "off", "Each decision is read in both option orders and averaged") : null,
      fact("Type filter", (h.fraud_free_types || []).join(", "), "Approved without a model: no fraud in training")),
    factCard("Costs",
      fact("Wrong decline", `${pct(h.costs.friction_rate)} of amount`, `At least ${h.costs.friction_min}`),
      fact("Review", money(h.costs.review_fee), "An analyst finds the truth"),
      fact("Review budget", `${pct(h.costs.review_share)} of the window`, `${Math.floor(h.costs.review_share * h.window.size)} reviews, in arrival order`)),
  ];
  if (s) {
    facts.push(factCard("Result",
      fact("Cost", money(s.cost)),
      fact("Cost at the natural rate", money(s.cost_natural), "Each row reweighted to its hours' real fraud rate"),
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
    const flips = model.filter((r) => r.orders && r.orders[0].indexOf(Math.max(...r.orders[0])) !== r.orders[1].indexOf(Math.max(...r.orders[1])));
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
        "Signals use only earlier rows; the four balance columns are never read.",
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

async function startPage() {
  document.title = "Fraud Console";
  const [{ runs }, { evals }] = await Promise.all([getJSON("api/runs"), getJSON("api/evals")]);
  const latest = evals[0];
  const setups = [...new Set(runs.map((r) => r.setup))].sort();
  const body = el("tbody");
  const setupSel = el("select", { "aria-label": "Setup" }, el("option", { value: "" }, "all setups"),
    ...setups.map((s) => el("option", { value: s }, s)));
  const evalBox = el("input", { type: "checkbox", id: "with-eval" });
  const fill = () => {
    const rows = runs.filter((r) => (!setupSel.value || r.setup === setupSel.value) && (evalBox.checked || !r.eval));
    rows.sort((a, b) => (b.pinned - a.pinned) || 0);
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
    el("h1", {}, "Approve, review or decline: System One on PaySim"),
    el("p", { class: "intro" }, "A local Qwen2.5-1.5B judges each TRANSFER and CASH_OUT, and a rule filter approves the transaction types " +
      "that had no fraud in training. The console replays recorded runs at their recorded speed and shows the pre-registered eval " +
      "against rules-only, logistic regression, random and approve-all."),
    el("div", { class: "cards" },
      el("section", { class: "panel bar-card" }, el("h3", {}, "Latest eval"),
        latest ? el("div", {},
          el("div", {}, `Eval ${latest.id}${latest.complete ? "" : " (incomplete)"}`),
          ...Object.entries(latest.bars || {}).map(([k, b]) => el("div", { style: "margin-top:6px" },
            chip(b.pass ? "pass" : "fail", b.pass ? "PASS" : "FAIL"), ` ${b.rule}`)),
          el("p", {}, el("a", { href: `#/eval/${latest.id}` }, "Open the eval"))) : el("p", { class: "muted" }, "No eval results yet.")),
      el("section", { class: "panel bar-card" }, el("h3", {}, "Auto-demo"),
        el("p", {}, "Plays the model runs one after another at 1×, pinned runs first."),
        el("button", { class: "primary", onclick: () => { location.hash = "#/demo"; } }, "Start the auto-demo"))),
    el("section", { class: "panel", style: "margin-top:12px" },
      el("h2", {}, "Runs"),
      el("div", { class: "filters" }, setupSel, el("label", { for: "with-eval" }, evalBox, " include eval runs")),
      el("div", { class: "scroll-x" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "run"), el("th", {}, "setup"), el("th", {}, "window"), el("th", {}, "model"),
          el("th", { class: "r" }, "cost"), el("th", { class: "r" }, "recall"), el("th", { class: "r" }, "length at 1×"))),
        body))));
}

const EVAL_SERIES = [["hybrid-1.5b", "var(--series-1)"], ["rules", "var(--series-2)"], ["logreg", "var(--series-3)"]];

function evalChart(r) {
  const ws = r.test_windows;
  const series = EVAL_SERIES.filter(([s]) => r.windows[s]);
  const vals = series.flatMap(([s]) => ws.map((w) => r.windows[s][w] && r.windows[s][w].cost)).filter((v) => v > 0);
  const lo = Math.floor(Math.log10(Math.min(...vals))), hi = Math.ceil(Math.log10(Math.max(...vals)));
  const W = 720, H = 260, L = 52, R = 10, T = 10, B = 26;
  const x = (i) => L + ((W - L - R) * (i + 0.5)) / ws.length;
  const y = (v) => T + (H - T - B) * (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo));
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Cost per test window for the hybrid, rules-only and logistic regression, log scale" });
  for (let p = lo; p <= hi; p++) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(10 ** p), y2: y(10 ** p), stroke: "var(--grid)", "stroke-width": 1 }));
    const t = svgEl("text", { x: L - 6, y: y(10 ** p) + 4, "text-anchor": "end" });
    t.textContent = p >= 6 ? `${10 ** (p - 6)}M` : p >= 3 ? `${10 ** (p - 3)}k` : `${10 ** p}`;
    svg.append(t);
  }
  ws.forEach((w, i) => {
    if (i % 2 === 0) { const t = svgEl("text", { x: x(i), y: H - 8, "text-anchor": "middle" }); t.textContent = w.replace("test", ""); svg.append(t); }
  });
  const dots = [];
  series.forEach(([s, color], j) => ws.forEach((w, i) => {
    const v = r.windows[s][w] && r.windows[s][w].cost;
    if (v === undefined) return;
    const c = svgEl("circle", { cx: x(i) + (j - 1) * 6, cy: y(v), r: 4.5, fill: color, stroke: "var(--surface)", "stroke-width": 2 });
    svg.append(c);
    dots.push({ cx: x(i) + (j - 1) * 6, cy: y(v), s, w, v });
  }));
  svg.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W, py = ((e.clientY - box.top) / box.height) * H;
    let best = null, bd = 1e9;
    for (const d of dots) { const dd = Math.hypot(d.cx - px, d.cy - py); if (dd < bd) { bd = dd; best = d; } }
    if (!best || bd > 18) return hideTip();
    showTip(e, [`${best.w} · ${best.s}`, `cost ${money(best.v)}`]);
  });
  svg.addEventListener("mouseleave", hideTip);
  return el("div", {},
    el("div", { class: "legend" }, ...series.map(([s, c]) => el("span", {}, el("i", { style: `background:${c}` }), SETUP_NAME[s] || s))),
    svg, el("div", { class: "hint" }, "Cost per test window, log scale (window number on the x axis). Random and approve-all are in the table below."));
}

async function evalPage(id) {
  const { evals } = await getJSON("api/evals");
  const meta = id ? evals.find((e) => e.id === id) : evals[0];
  if (!meta) { app.replaceChildren(el("p", { class: "muted" }, "No eval results in fraud/results/eval/ yet.")); return; }
  const r = await getJSON(`results/eval/${encodeURIComponent(meta.file)}`);
  const { runs } = await getJSON("api/runs");
  const have = new Set(runs.map((x) => x.file));
  document.title = `Eval ${r.eval} · Fraud Console`;
  const setups = Object.keys(r.totals);
  const bars = Object.entries(r.bars).map(([k, b]) => el("section", { class: "panel bar-card" },
    el("h3", {}, k === "bar1" ? "Bar 1: cost" : "Bar 2: speed"),
    el("div", {}, chip(b.pass ? "pass" : "fail", b.pass ? "PASS" : "FAIL")),
    el("div", { class: "value", style: "margin-top:6px" }, k === "bar2" ? (b.value === null ? "–" : `${b.value.toFixed(1)} ms`) : `${b.value} of ${r.test_windows.length}`),
    el("div", { class: "hint" }, b.rule)));
  const tot = el("table", {},
    el("thead", {}, el("tr", {}, ...["setup", "mean cost", "mean cost (natural rate)", "recall", "precision", "alert rate", "p50 / p90 ms"]
      .map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
    el("tbody", {}, ...setups.map((s) => { const v = r.totals[s]; const l = v.latency_ms;
      return el("tr", {}, el("td", {}, SETUP_NAME[s] || s), el("td", { class: "r" }, money(v.cost_mean)), el("td", { class: "r" }, money(v.cost_natural_mean)),
        el("td", { class: "r" }, fix(v.recall_mean)), el("td", { class: "r" }, fix(v.precision_mean)), el("td", { class: "r" }, pct(v.alert_rate_mean, 1)),
        el("td", { class: "r" }, l ? `${l.p50.toFixed(0)} / ${l.p90.toFixed(0)}` : "–")); })));
  const perWin = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "window"), ...setups.map((s) => el("th", { class: "r" }, s)))),
    el("tbody", {}, ...r.test_windows.map((w) => el("tr", {}, el("td", {}, w), ...setups.map((s) => {
      const v = r.windows[s] && r.windows[s][w];
      const f = `${r.eval}_eval_${s}_${w}.jsonl`;
      return el("td", { class: "r" }, !v ? "–" : have.has(f) ? el("a", { href: `#/run/${encodeURIComponent(f)}` }, money(v.cost)) : money(v.cost));
    })))));
  const paired = el("table", {},
    el("thead", {}, el("tr", {}, ...["pair", "windows", "first cheaper", "second cheaper", "ties", "sign test p"].map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
    el("tbody", {}, ...Object.entries(r.paired).filter(([, v]) => v.windows).map(([k, v]) => el("tr", {}, el("td", {}, k),
      el("td", { class: "r" }, v.windows), el("td", { class: "r" }, v.a_cheaper), el("td", { class: "r" }, v.b_cheaper),
      el("td", { class: "r" }, v.ties), el("td", { class: "r" }, v.sign_test_p.toPrecision(2))))));
  const cal = Object.entries(r.totals).filter(([, v]) => v.calibration);
  app.replaceChildren(
    el("h1", {}, `Eval ${r.eval}`),
    el("p", { class: "intro" }, `${r.test_windows.length} held-out test windows, run once per setup. Pre-registered in the docstring of fraud/eval.py before any ` +
      `test window ran. Complete: ${r.complete ? "yes" : "no"}. Write-up with labelled claims: fraud/docs/results.md.`),
    el("div", { class: "cards" }, ...bars),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Cost per window"), evalChart(r)),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Setups (means over the test windows)"), el("div", { class: "scroll-x" }, tot)),
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Paired window counts"), el("div", { class: "scroll-x" }, paired)),
    cal.length ? el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Model scores (uncalibrated)"),
      el("div", { class: "scroll-x" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["setup", "decisions", "top score mean", "top option right", "ECE", "outside mass"].map((t, i) => el("th", { class: i ? "r" : "" }, t)))),
        el("tbody", {}, ...cal.map(([s, v]) => { const c = v.calibration; return el("tr", {}, el("td", {}, s), el("td", { class: "r" }, c.decisions),
          el("td", { class: "r" }, fix(c.mean_top_score, 3)), el("td", { class: "r" }, pct(c.top_right_rate, 1)), el("td", { class: "r" }, fix(c.ece, 3)),
          el("td", { class: "r" }, fix(c.outside_mass_mean, 4))); })))),
      el("p", { class: "hint" }, "“Right” means legit → approve, fraud → review or decline. ECE compares the top score with how often the top option was right.")) : null,
    el("section", { class: "panel", style: "margin-top:12px" }, el("h2", {}, "Cost per window, every setup"),
      el("p", { class: "hint" }, "A linked cost opens that window's recorded run (eval traces stay local)."), el("div", { class: "scroll-x" }, perWin)));
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
