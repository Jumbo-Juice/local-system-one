// The Master Viewer's shell: the three ways to watch (load a trace, a random run, auto-demo), the run
// pool served by `python -m viewer`, playback, and the page around a game's stage. Nothing here knows
// a game's rules: each game viewer in games/ draws its own stage and lists its own facts.
"use strict";

const $ = (id) => document.getElementById(id);
const HOLD_S = 3;  // auto-demo: seconds a finished run's outcome stays up before the pane loads the next
const traceSlot = $("trace");
const EMBEDDED = traceSlot.textContent.trim();  // a bundled page's run (python -m viewer --bundle)
let mode = null;                                // "start" | "single" | "auto"

// ---------------------------------------------------------------- the pool

// From the server: {pools: ["dungeon", "shooter"], runs: [{path, game, scenario, label, created, ...}]},
// runs newest first. null when the page was opened from disk: then only Load trace works.
let pool = null;

async function fetchPool() {
  try {
    const res = await fetch("api/runs", { cache: "no-store" });
    pool = res.ok ? await res.json() : null;
  } catch {
    pool = null;
  }
  return pool;
}

const inFilter = (entry, filter) => filter === "all" || entry.game === filter;
const playable = (filter) => (pool ? pool.runs.filter((e) => inFilter(e, filter) && Games.get(e.scenario)) : []);
const findEntry = (path) => (pool ? pool.runs.find((e) => e.path === path) : null) || null;
const gameTitle = (name) => (Games.get(name) ? Games.get(name).title : name);
const when = (created) => String(created || "").replace("T", " ").slice(0, 16);

async function fetchRun(entry) {
  const res = await fetch(encodeURI(entry.path), { cache: "no-store" });
  if (!res.ok) throw new Error(`${entry.path}: HTTP ${res.status}`);
  return Run.parse(await res.text());
}

// The All / Shooter / Dungeon / ... buttons, one per pool folder, with run counts.
function renderFilter(box, current, onPick) {
  const pools = pool ? pool.pools : [];
  const options = [["all", "All"], ...pools.map((g) => [g, Games.get(g) ? gameTitle(g) : `${g} (no viewer yet)`])];
  box.replaceChildren(...options.map(([value, label]) => {
    const b = document.createElement("button");
    b.textContent = `${label} · ${pool ? pool.runs.filter((e) => inFilter(e, value)).length : 0}`;
    b.setAttribute("aria-pressed", String(value === current));
    b.onclick = () => onPick(value);
    return b;
  }));
}

// ---------------------------------------------------------------- page chrome

function show(view) {
  mode = view;
  for (const id of ["start", "single", "auto"]) $(id).hidden = id !== view;
  $("wrap").classList.toggle("wide", view === "auto");
}

function setNav(buttons) {
  $("nav").replaceChildren(...buttons.map(([label, onClick]) => {
    const b = document.createElement("button"); b.textContent = label; b.onclick = onClick; return b;
  }));
}

// "SHOOTER  label · 2026-09-29 13:33 · path" above the stage.
function setWhere(game, entry) {
  const parts = [];
  const g = document.createElement("b"); g.textContent = gameTitle(game).toUpperCase(); parts.push(g);
  const rest = [entry.label, when(entry.created), entry.path].filter(Boolean).join(" · ");
  parts.push(document.createTextNode("  " + rest));
  $("where").replaceChildren(...parts);
}

// ---------------------------------------------------------------- start screen

let startFilter = "all";

async function showStart() {
  show("start");
  setNav([]);
  $("where").textContent = "";
  await fetchPool();
  if (pool && startFilter !== "all" && !pool.pools.includes(startFilter)) startFilter = "all";
  renderFilter($("filter"), startFilter, (value) => { startFilter = value; showStart(); });
  $("filter").hidden = !pool;
  const n = playable(startFilter).length;
  $("choose-random").disabled = $("choose-auto").disabled = !n;
  const note = $("pool-note");
  if (!pool) {
    note.replaceChildren("This page was opened from disk, so it cannot see the run pools. Random run and Auto-demo need the viewer's server:");
    const pre = document.createElement("pre"); pre.className = "cmd"; pre.textContent = ".venv/Scripts/python -m viewer";
    note.append(pre);
  } else {
    const counts = pool.pools.map((g) => `${pool.runs.filter((e) => e.game === g).length} ${gameTitle(g).toLowerCase()}`).join(", ");
    note.textContent = pool.runs.length
      ? `${pool.runs.length} runs in runs/ (${counts}), newest ${when(pool.runs[0].created)}. Pinned runs are listed in runs/README.md.`
      : "No runs yet. Capture one, for example: python -m demo.shooter.capture --config config/mock.toml --seed 0";
  }
}

$("choose-random").onclick = () => { location.hash = `random=${startFilter}`; };
$("choose-auto").onclick = () => { location.hash = `auto=${startFilter}`; };
$("file").onchange = (e) => { const f = e.target.files[0]; if (f) openFile(f); e.target.value = ""; };
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) openFile(f);
});

// ---------------------------------------------------------------- one run, full view

const stage = $("stage"), stageCtx = stage.getContext("2d");
const S = { run: null, clock: 0, playing: false, speed: 1, recorder: null, lastInspected: -1 };

function openSingle(run, entry, { tick = null, nav = [] } = {}) {
  show("single");
  setNav(nav);
  setWhere(run.header.scenario, entry);
  $("problem").hidden = true;
  Object.assign(S, { run, clock: 0, playing: tick == null, lastInspected: -1 });
  fillAbout(run);
  if (tick != null) {
    const j = run.ticks.findIndex((t) => t.tick === Number(tick));
    if (j >= 0) S.clock = run.start[j] + run.dur[j] * 0.5;
  }
  updatePlay();
}

function problem(err, entry) {
  show("single");
  S.run = null;
  if (entry) setWhere(entry.game || "run", entry);
  $("problem").hidden = false;
  $("problem").querySelector("p").textContent = String(err.message || err);
  stageCtx.setTransform(1, 0, 0, 1, 0, 0);
  stageCtx.fillStyle = "#05070b"; stageCtx.fillRect(0, 0, stage.width, stage.height);
}

async function showRun(path, { tick = null, from = null } = {}) {
  if (!pool) await fetchPool();
  const entry = findEntry(path) || { path, label: path.split("/").pop() };
  const nav = from ? [["◀ Back to auto-demo", () => { location.hash = `auto=${from}`; }]] : [];
  try {
    openSingle(await fetchRun(entry), entry, { tick, nav });
  } catch (err) {
    setNav(nav); problem(err, entry);
  }
}

async function showRandom(filter, path) {
  if (!pool) await fetchPool();
  if (!path) {
    const runs = playable(filter), current = S.run && S.entryPath;
    const choices = runs.length > 1 ? runs.filter((e) => e.path !== current) : runs;
    if (!choices.length) { location.hash = "start"; return; }
    location.hash = `random=${filter}&run=${encodeURIComponent(choices[Math.floor(Math.random() * choices.length)].path)}`;
    return;
  }
  const entry = findEntry(path) || { path, label: path.split("/").pop() };
  const nav = [["Another random run", () => { location.hash = `random=${filter}`; }]];
  try {
    openSingle(await fetchRun(entry), entry, { nav });
    S.entryPath = path;
  } catch (err) {
    setNav(nav); problem(err, entry);
  }
}

async function openFile(file) {
  const entry = { label: file.name };
  history.pushState(null, "", "#file");
  try {
    openSingle(Run.parse(await file.text()), entry);
  } catch (err) {
    setNav([]); problem(err, entry);
  }
}

function frameSingle(dt) {
  const R = S.run;
  if (!R) return;
  if (S.playing) {
    S.clock += dt * S.speed;
    if (S.clock >= R.total) { S.clock = R.total - 1e-6; S.playing = false; updatePlay(); if (S.recorder) S.recorder.stop(); }
  }
  R.game.drawStage(stageCtx, R, S.clock, S.playing);
  const [i] = Run.locate(R, S.clock);
  $("scrub").value = String(Math.round(10000 * S.clock / R.total));
  $("readout").textContent = `tick ${R.ticks[i].tick} / ${R.n - 1} · ${Run.fmtTime(S.clock)}`;
  if (!S.playing || S.speed <= 2) inspect(i);
}

function updatePlay() {
  $("play").textContent = S.playing ? "Pause" : "Play";
  $("play").setAttribute("aria-pressed", String(S.playing));
}

function stepTick(delta) {
  const R = S.run;
  if (!R) return;
  S.playing = false; updatePlay();
  const [i] = Run.locate(R, S.clock);
  const j = Math.max(0, Math.min(R.n - 1, i + delta));
  S.clock = R.start[j] + R.dur[j] * 0.9;
}

$("play").onclick = () => { if (!S.run) return; if (S.clock >= S.run.total - 1e-3) S.clock = 0; S.playing = !S.playing; updatePlay(); };
$("restart").onclick = () => { S.clock = 0; S.lastInspected = -1; };
$("back").onclick = () => stepTick(-1);
$("fwd").onclick = () => stepTick(1);
$("speed").onchange = (e) => { S.speed = parseFloat(e.target.value); };
$("scrub").oninput = (e) => { if (S.run) S.clock = Math.min(S.run.total - 1e-6, S.run.total * e.target.value / 10000); };
$("rec").onclick = () => {
  if (S.recorder) { S.recorder.stop(); return; }
  if (!S.run || !stage.captureStream || !window.MediaRecorder) { $("rec").textContent = "Recording not supported here"; return; }
  const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  const chunks = [], name = `${S.run.header.scenario}_seed${S.run.map.seed}.webm`;
  S.recorder = new MediaRecorder(stage.captureStream(30), { mimeType: type, videoBitsPerSecond: 10e6 });
  S.recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  S.recorder.onstop = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
    a.download = name;
    a.click();
    S.recorder = null; $("rec").classList.remove("on"); $("rec").textContent = "Record WebM";
  };
  S.clock = 0; S.playing = true; updatePlay();
  S.recorder.start(); $("rec").classList.add("on"); $("rec").textContent = "Stop recording";
};

// What the model saw: every decision of the tick, with the prompt rebuilt from the recorded parts.
function promptText(d, header) {
  const parts = [];
  if (d.state) parts.push("State:\n" + d.state);
  if (d.context) parts.push("Context:\n" + d.context);
  parts.push("Question: " + d.question);
  const labelled = header.engine.answer === "label";
  parts.push("Options:\n" + d.options.map((o, k) => (labelled ? d.labels[k] + ": " : "- ") + o).join("\n"));
  parts.push(labelled ? "Reply with the label of one option." : "Reply with one option, copied exactly.");
  const note = header.engine.prompt_order === "state_first" ? "" : `\n\n(prompt_order is ${header.engine.prompt_order}: the real order differs)`;
  return parts.join("\n\n") + "\n\nAssistant (prefilled): " + header.engine.prefill + note;
}

function inspect(i) {
  if (i === S.lastInspected) return;
  S.lastInspected = i;
  const R = S.run, t = R.ticks[i];
  $("inspect-tick").textContent = t.tick;
  $("decs").replaceChildren(...t.decisions.map((d) => decisionCard(d, R)));
}

function decisionCard(d, R) {
  const el = (tag, cls, txt) => { const x = document.createElement(tag); if (cls) x.className = cls; if (txt != null) x.textContent = txt; return x; };
  const art = el("article", "dec"), head = el("div");
  const tr = d.tournament;
  head.append(el("span", "tier", R.game.tierLabels[d.tier] || d.tier), el("span", "meta", d.method === "only_option"
    ? "  only option · no model call"
    : `  ${d.kind} · ${d.prompt_tokens} prompt tokens` + (tr ? ` · tournament${tr.round ? " round " + tr.round : ""} over ${tr.options} options` : "")));
  art.append(head, el("pre", null, promptText(d, R.header)), el("p", "answer", d.method === "only_option"
    ? `Committed: ${d.options[0]}`
    : `Chose ${d.labels[d.choice]} → ${d.options[d.choice]} (p ${d.probs[d.choice].toFixed(3)}). ` +
      `Mass outside the labels ${(100 * d.outside).toFixed(2)}%. Unconstrained top token: ${JSON.stringify(d.top_token)}.`));
  if (d.orders) {  // order averaging: both readings and their mean
    const tb = el("table"), hr = el("tr");
    for (const h of ["option", "as listed", "reversed", "mean"]) hr.append(el("th", null, h));
    tb.append(hr);
    d.options.forEach((o, k) => {
      const row = el("tr");
      for (const v of [`${d.labels[k]}: ${o.split(" (")[0]}`, d.orders[0][k].toFixed(3), d.orders[1][k].toFixed(3), d.probs[k].toFixed(3)]) row.append(el("td", null, v));
      tb.append(row);
    });
    art.append(tb);
  }
  return art;
}

// About this run: outcome banner, facts as titled cards (one fact per row, explanation as the hint
// under the value), notes as titled cards of short bullets. The game supplies the content.
function fillAbout(R) {
  const [word, tone, detail] = R.game.verdict(R), b = R.header.backend || {};
  const box = $("verdict"), strong = document.createElement("strong"), span = document.createElement("span"), p = document.createElement("p");
  box.style.setProperty("--tone", `var(--${tone})`);
  span.textContent = " " + detail; strong.append(word, span);
  p.textContent = `${b.model || b.kind} on ${b.device || "?"}, ${b.dtype || "?"} · seed ${R.map.seed}`;
  box.replaceChildren(strong, p);
  $("facts").replaceChildren(...R.game.facts(R).map(([title, rows]) => {
    const card = document.createElement("div"), h = document.createElement("h3"), dl = document.createElement("dl");
    card.className = "card"; h.textContent = title;
    for (const [k, val, hint] of rows) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = k; dd.textContent = String(val);
      if (hint) { const sm = document.createElement("small"); sm.textContent = hint; dd.append(sm); }
      dl.append(dt, dd);
    }
    card.append(h, dl); return card;
  }));
  const playback = ["Playback and video", [
    "1× plays at the recorded tick times. Ticks shorter than 50 ms (for example with the no-model mock backend) are shown for 50 ms.",
    ["Record WebM works in Chrome and Edge. Convert the video with:", "ffmpeg -i run.webm -c:v libx264 -pix_fmt yuv420p -crf 18 run.mp4"],
  ]];
  $("notes").replaceChildren(...[...R.game.notes(R), playback].map(([title, items]) => {
    const card = document.createElement("div"), h = document.createElement("h3"), ul = document.createElement("ul");
    card.className = "card"; h.textContent = title;
    for (const item of items) {
      const li = document.createElement("li");
      if (Array.isArray(item)) { const pre = document.createElement("pre"); pre.textContent = item[1]; li.append(item[0], pre); }
      else li.textContent = item;
      ul.append(li);
    }
    card.append(h, ul); return card;
  }));
  $("inspect-note").textContent = R.game.inspectNote;
  $("template").textContent = "System prompt:\n" + R.header.engine.system_prompt +
    `\n\nExample of a full prompt as the model saw it (${R.game.exampleFrom}, state shown as <state>):\n\n` + R.header.example_prompt;
}

// ---------------------------------------------------------------- auto-demo: two panes, newest to oldest

// One queue for both panes: the filtered pool, newest first. A pane that finishes shows its outcome
// for HOLD_S, then takes the next older run; after the oldest the pool is re-read and the queue
// starts again at the newest, so runs finished in the meantime join the next cycle.
const A = { filter: null, queue: [], next: 0, cycle: 1, panes: [], paused: false, speed: 1, gen: 0, refreshing: null };

async function showAuto(filter) {
  show("auto");
  setNav([]);
  $("where").textContent = "Auto-demo";
  if (A.filter === filter && A.panes.length) { renderAuto(); return; }  // back from a pane's full view
  A.gen++;
  Object.assign(A, { filter, next: 0, cycle: 1, refreshing: null });
  await fetchPool();
  A.queue = playable(filter);
  A.panes = [makePane(), makePane()];
  $("panes").replaceChildren(...A.panes.map((p) => p.el));
  renderAuto();
  for (const p of A.panes) await advance(p);  // in order: pane 1 gets the newest, pane 2 the next
}

function renderAuto() {
  renderFilter($("auto-filter"), A.filter, (value) => { location.hash = `auto=${value}`; });
  $("auto-play").textContent = A.paused ? "Play" : "Pause";
  $("auto-play").setAttribute("aria-pressed", String(!A.paused));
  const next = A.queue[A.next];
  $("queue").textContent = !A.queue.length ? "" : `${A.queue.length} runs, newest first · cycle ${A.cycle}` +
    (next ? ` · next: ${next.label}` : " · next: back to the newest");
  $("auto-note").textContent = !pool ? "The run pools need the viewer's server: python -m viewer"
    : !A.queue.length ? "No runs in this pool yet." : "Click a pane to open its run in the full view.";
}

function makePane() {
  const el = document.createElement("article"); el.className = "pane";
  const title = document.createElement("button"); title.className = "pane-title"; title.title = "Open this run in the full view";
  const canvas = document.createElement("canvas"), strip = document.createElement("div"); strip.className = "strip";
  el.append(title, canvas, strip);
  const pane = { el, title, canvas, ctx: canvas.getContext("2d"), strip, entry: null, run: null, rect: null,
    clock: 0, hold: 0, loading: false, lastTick: -1 };
  const open = () => { if (pane.entry) location.hash = `run=${encodeURIComponent(pane.entry.path)}&from=${A.filter}`; };
  title.onclick = open; canvas.onclick = open;
  return pane;
}

async function nextEntry() {
  while (A.next >= A.queue.length) {  // past the oldest (or empty): re-read the pool, start at the newest
    if (!A.refreshing) {
      A.refreshing = (async () => {
        await fetchPool();
        A.queue = playable(A.filter); A.next = 0; A.cycle++;
        A.refreshing = null;
      })();
    }
    await A.refreshing;
    if (!A.queue.length) return null;
  }
  return A.queue[A.next++];
}

async function advance(pane) {
  const gen = A.gen;
  pane.loading = true;
  for (let tries = 0; tries <= A.queue.length; tries++) {
    const entry = await nextEntry();
    if (gen !== A.gen) return;
    if (!entry) break;
    try {
      const run = await fetchRun(entry);
      if (gen !== A.gen) return;
      setPaneRun(pane, entry, run);
      break;
    } catch (err) {
      console.warn("auto-demo skipped a run:", entry.path, err);
    }
  }
  pane.loading = false;
  renderAuto();
}

function setPaneRun(pane, entry, run) {
  Object.assign(pane, { entry, run, rect: run.game.pane(run), clock: 0, hold: 0, lastTick: -1 });
  pane.canvas.width = pane.rect.w; pane.canvas.height = pane.rect.h;
  const game = document.createElement("span"), label = document.createElement("span"), time = document.createElement("span");
  game.className = "game"; game.textContent = run.game.title;
  label.className = "label"; label.textContent = entry.label + (entry.pinned ? " ★" : "");
  time.className = "when"; time.textContent = when(entry.created);
  pane.title.replaceChildren(game, label, time);
}

function framePane(pane, dt) {
  const R = pane.run;
  if (!R) return;
  if (!A.paused) {
    if (pane.clock < R.total) {
      pane.clock = Math.min(R.total, pane.clock + dt * A.speed);
      if (pane.clock >= R.total) pane.hold = HOLD_S;
    } else if (!pane.loading) {
      pane.hold -= dt;  // real seconds, whatever the speed
      if (pane.hold <= 0) advance(pane);
    }
  }
  const clock = Math.min(pane.clock, R.total - 1e-6);
  pane.ctx.setTransform(1, 0, 0, 1, -pane.rect.x, -pane.rect.y);
  R.game.drawPane(pane.ctx, R, clock);
  const [i] = Run.locate(R, clock);
  if (i !== pane.lastTick) { pane.lastTick = i; renderStrip(pane, R, i); }
}

// One line under a pane: the answer in force for each tier, with its probability.
function renderStrip(pane, R, i) {
  pane.strip.replaceChildren(...R.game.tiers.map((tier) => {
    const d = Run.held(R, tier, i), chip = document.createElement("span");
    chip.className = "chip";
    const name = document.createElement("b"), what = document.createElement("span"), p = document.createElement("i");
    name.textContent = R.game.tierLabels[tier] || tier;
    what.textContent = d ? d.options[d.choice] : "not decided yet";
    p.textContent = !d ? "" : d.method === "only_option" ? "only" : Math.round(100 * d.probs[d.choice]) + "%";
    chip.title = d ? `${name.textContent}: ${d.options[d.choice]}` : "";
    chip.append(name, what, p);
    return chip;
  }));
}

$("auto-play").onclick = () => { A.paused = !A.paused; renderAuto(); };
$("auto-speed").onchange = (e) => { A.speed = parseFloat(e.target.value); };

// ---------------------------------------------------------------- routing, keys, the frame loop

// #start · #run=<path>[&tick=N][&from=<filter>] · #random=<filter>[&run=<path>] · #auto=<filter> · #file
function route() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (EMBEDDED && (!location.hash || location.hash === "#")) {
    const entry = { label: traceSlot.dataset.name || "bundled run" };
    try { openSingle(Run.parse(EMBEDDED), entry); } catch (err) { problem(err, entry); }
    return;
  }
  if (h.has("auto")) return showAuto(h.get("auto") || "all");
  if (h.has("random")) return showRandom(h.get("random") || "all", h.get("run"));
  if (h.has("run")) return showRun(h.get("run"), { tick: h.get("tick"), from: h.get("from") });
  if (h.has("file") && S.run) return;  // a file loaded from disk; nothing to reload
  return showStart();
}
window.addEventListener("hashchange", route);

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  if (mode === "auto" && e.code === "Space") { e.preventDefault(); $("auto-play").click(); return; }
  if (mode !== "single") return;
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  else if (e.code === "ArrowLeft") stepTick(-1);
  else if (e.code === "ArrowRight") stepTick(1);
  else if (e.code === "Home") $("restart").click();
});

let lastFrame = null;
function frame(ts) {
  const dt = lastFrame == null ? 0 : Math.min(0.25, (ts - lastFrame) / 1000);
  lastFrame = ts;
  if (mode === "single") frameSingle(dt);
  else if (mode === "auto") for (const p of A.panes) framePane(p, dt);
  requestAnimationFrame(frame);
}

$("home").href = "#start";
route();
requestAnimationFrame(frame);
