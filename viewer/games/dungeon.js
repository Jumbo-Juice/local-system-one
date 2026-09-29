// The first dungeon demo's viewer (demo/dungeon): draws one tick of a recorded dungeon run.
// Stage: 1920x1080, map and HUD on the left, one decision card per tier on the right, the latency
// timeline under the cards. The auto-demo panes show the HUD and map only (pane()).
"use strict";

(() => {
  const { UI, MONO, font, spacing, text, fit, wrap, rrect, panel, pill, lerp, ease, alpha, mix } = Paint;
  const W = 1920, H = 1080, CS = 38, MX = 40, MY = 250;
  // One deliberate dark look, matching the video-style stage (the Doom demo in sgoedecke/system-one).
  const K = {
    ground: "#0a111c", void: "#070c14", panel: "#111d2c", panelHi: "#15263a", edge: "#1e3048",
    ink: "#dce7f2", muted: "#8ca4b9", faint: "#56708a", cyan: "#66d9f0", mint: "#8bf2c6", amber: "#f2c14e",
    red: "#f2665e", violet: "#a78bfa", orange: "#f29b4e", pink: "#ff86b0",
    wall: "#1c2a3e", wallTop: "#2c405c", wallFace: "#131e2d", floor: "#15233a", floorLine: "#1a2b45",
    corridor: "#111d30", jamb: "#8a7048", slate: "#35506e",
  };
  const CARDS = {
    strategy: { x: 992, y: 112, w: 888, h: 262 },
    target: { x: 992, y: 386, w: 888, h: 306 },
    action: { x: 992, y: 704, w: 888, h: 218 },
  };
  const TL = { x: 992, y: 956, w: 888, h: 92 };
  const TIER_LABEL = { strategy: "Strategy", target: "Target", action: "Action" };

  // The run, context and playback state being drawn; set by begin() for each frame and pane.
  let R = null, ctx = null, clock = 0, playing = false;
  function begin(context, run, time, isPlaying) {
    ctx = context; R = run; clock = time; playing = isPlaying;
    Paint.use(context, K);
  }

  // ---------------------------------------------------------------- per-run preparation

  function prepare(run) {
    run.open = openCells(run.map);
    run.base = drawBase(run);
    run.timeline = drawTimelineBase(run);
  }

  function openCells(map) {
    const kind = new Map();  // "x,y" -> room | door | corridor
    for (const r of map.rooms) for (let y = r.y0; y < r.y0 + r.h; y++) for (let x = r.x0; x < r.x0 + r.w; x++) kind.set(x + "," + y, "room");
    for (const c of map.corridors) {
      for (const [x, y] of c.cells) kind.set(x + "," + y, "corridor");
      for (const [x, y] of c.doors) kind.set(x + "," + y, "door");
    }
    return kind;
  }

  const cx = (x) => MX + x * CS + CS / 2, cy = (y) => MY + y * CS + CS / 2;

  // ---------------------------------------------------------------- static map layer

  function drawBase(run) {
    const map = run.map, off = document.createElement("canvas");
    off.width = map.width * CS; off.height = map.height * CS;
    const g = off.getContext("2d");
    g.fillStyle = K.void; g.fillRect(0, 0, off.width, off.height);
    const kind = run.open, isOpen = (x, y) => kind.has(x + "," + y);
    for (let y = 0; y < map.height; y++) for (let x = 0; x < map.width; x++) {
      const px = x * CS, py = y * CS, k = kind.get(x + "," + y);
      if (!k) {
        let near = false;
        for (let dy = -1; dy <= 1 && !near; dy++) for (let dx = -1; dx <= 1; dx++) if (isOpen(x + dx, y + dy)) { near = true; break; }
        if (!near) continue;
        g.fillStyle = K.wall; g.fillRect(px, py, CS, CS);
        g.fillStyle = K.wallTop; g.fillRect(px, py, CS, 3);
        if (isOpen(x, y + 1)) { g.fillStyle = K.wallFace; g.fillRect(px, py + CS - 9, CS, 9); }
        continue;
      }
      g.fillStyle = k === "corridor" ? K.corridor : K.floor; g.fillRect(px, py, CS, CS);
      g.strokeStyle = K.floorLine; g.lineWidth = 1; g.strokeRect(px + 0.5, py + 0.5, CS - 1, CS - 1);
      if (k === "door") {
        g.fillStyle = K.jamb;
        if (isOpen(x - 1, y) && isOpen(x + 1, y)) { g.fillRect(px, py, CS, 5); g.fillRect(px, py + CS - 5, CS, 5); }
        else { g.fillRect(px, py, 5, CS); g.fillRect(px + CS - 5, py, 5, CS); }
      }
    }
    return off;
  }

  // ---------------------------------------------------------------- glyphs

  function gemGlyph(x, y, s) {
    ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s * 0.8, y - s * 0.2); ctx.lineTo(x, y + s); ctx.lineTo(x - s * 0.8, y - s * 0.2); ctx.closePath();
    ctx.fillStyle = K.violet; ctx.fill();
    ctx.beginPath(); ctx.moveTo(x - s * 0.8, y - s * 0.2); ctx.lineTo(x + s * 0.8, y - s * 0.2); ctx.strokeStyle = "#d8ccff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  function foodGlyph(x, y, s) {
    ctx.beginPath(); ctx.arc(x, y + 2, s * 0.75, 0, 7); ctx.fillStyle = K.orange; ctx.fill();
    ctx.beginPath(); ctx.ellipse(x + 4, y - s * 0.7, 5, 3, -0.6, 0, 7); ctx.fillStyle = "#6fcf7f"; ctx.fill();
  }
  function potionGlyph(x, y, s) {
    ctx.fillStyle = K.pink;
    ctx.beginPath(); ctx.arc(x, y + 3, s * 0.7, 0, 7); ctx.fill();
    ctx.fillRect(x - 3.5, y - s * 0.95, 7, s * 0.8);
    ctx.fillStyle = "#c9a27a"; ctx.fillRect(x - 4.5, y - s * 1.1, 9, 4);
    ctx.fillStyle = "#ffd3e3"; ctx.beginPath(); ctx.arc(x - 3, y + 1, 2.5, 0, 7); ctx.fill();
  }
  function keyGlyph(x, y, s) {
    ctx.strokeStyle = K.amber; ctx.lineWidth = 3.5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(x - s * 0.45, y, s * 0.38, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - s * 0.08, y); ctx.lineTo(x + s * 0.9, y); ctx.moveTo(x + s * 0.55, y); ctx.lineTo(x + s * 0.55, y + s * 0.4);
    ctx.moveTo(x + s * 0.85, y); ctx.lineTo(x + s * 0.85, y + s * 0.35); ctx.stroke(); ctx.lineCap = "butt";
  }
  function lockGlyph(x, y) {
    ctx.fillStyle = K.amber; rrect(x - 7, y - 2, 14, 11, 2); ctx.fill();
    ctx.strokeStyle = K.amber; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y - 3, 5, Math.PI, 0); ctx.stroke();
  }
  function exitGlyph(x, y, s, locked, lit) {
    ctx.save();
    if (lit) { ctx.shadowColor = K.mint; ctx.shadowBlur = 24; }
    for (let i = 0; i < 4; i++) {
      ctx.fillStyle = alpha(K.mint, 0.35 + 0.18 * i);
      ctx.fillRect(x - s + i * s * 0.25, y - s + i * s * 0.5, s * 2 - i * s * 0.25, s * 0.42);
    }
    ctx.restore();
    if (locked) lockGlyph(x + s * 0.55, y + s * 0.45);
  }
  function enemyGlyph(x, y, s, mode, t) {
    const col = mode === "stunned" ? "#7d8796" : mode === "chase" ? K.red : "#b8504b";
    ctx.save();
    if (mode === "chase") { ctx.shadowColor = K.red; ctx.shadowBlur = 16; }
    ctx.beginPath();
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2 + (mode === "chase" ? t * 2 : 0), r = i % 2 ? s * 0.72 : s;
      ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
    }
    ctx.closePath(); ctx.fillStyle = col; ctx.fill(); ctx.restore();
    ctx.fillStyle = K.void;
    ctx.beginPath(); ctx.arc(x - s * 0.28, y - s * 0.1, s * 0.16, 0, 7); ctx.arc(x + s * 0.28, y - s * 0.1, s * 0.16, 0, 7); ctx.fill();
    if (mode === "stunned") {
      for (let i = 0; i < 3; i++) {
        const a = t * 3 + i * 2.1;
        ctx.fillStyle = K.amber; ctx.beginPath(); ctx.arc(x + Math.cos(a) * s * 0.9, y - s * 0.9 + Math.sin(a) * 4, 2.5, 0, 7); ctx.fill();
      }
    }
  }
  function agentGlyph(x, y, s, dir, hurt) {
    ctx.save(); ctx.shadowColor = hurt > 0 ? K.red : K.mint; ctx.shadowBlur = 18 + 20 * hurt;
    ctx.beginPath(); ctx.arc(x, y, s, 0, 7); ctx.fillStyle = hurt > 0 ? mix(K.mint, K.red, hurt) : K.mint; ctx.fill(); ctx.restore();
    ctx.lineWidth = 3; ctx.strokeStyle = K.void; ctx.beginPath(); ctx.arc(x, y, s, 0, 7); ctx.stroke();
    if (dir) {
      ctx.fillStyle = K.void; ctx.beginPath();
      ctx.arc(x + dir[0] * s * 0.5, y + dir[1] * s * 0.5, s * 0.28, 0, 7); ctx.fill();
    }
  }

  // ---------------------------------------------------------------- stage

  function drawStage(context, run, time, isPlaying) {
    begin(context, run, time, isPlaying);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "#0c1523"); grad.addColorStop(1, K.ground);
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
    const m = Run.moment(R, clock);
    drawHeader(m.tick);
    drawOrder();
    drawHud(m.world);
    drawMap(m);
    for (const tier of Object.keys(CARDS)) drawCard(tier, m.i);
    drawTimeline(m.p);
    drawOutcome(m.i, m.f);
  }

  // The part of the stage an auto-demo pane shows: the HUD and the map, in stage coordinates.
  function pane(run) {
    return { x: 24, y: 188, w: Math.max(944, run.map.width * CS + 32), h: MY - 188 + run.map.height * CS + 16 };
  }

  function drawPane(context, run, time) {
    begin(context, run, time, true);
    const r = pane(run);
    ctx.fillStyle = K.ground; ctx.fillRect(r.x, r.y, r.w, r.h);
    const m = Run.moment(R, clock);
    drawHud(m.world);
    drawMap(m);
    drawOutcome(m.i, m.f);
  }

  function drawHeader(t) {
    spacing(3); text("SYSTEM ONE · DUNGEON", 40, 66, K.ink, 40, 700); spacing(0);
    const b = R.header.backend || {};
    const dev = [b.device, b.dtype].filter(Boolean).join(" · ");
    text(`${Run.modelName(R)}${dev ? " · " + dev : ""} · local · one batched forward pass per tick · replay of recorded decisions · seed ${R.map.seed}`,
      40, 96, K.muted, 17);
    const tickStr = `TICK ${String(t.tick).padStart(3, "0")} / ${R.n - 1}`;
    spacing(2); text(tickStr, 1880, 62, K.ink, 34, 700, MONO, "right"); spacing(0);
    const b2 = t.batch;
    const batch = b2.decisions ? `${b2.decisions} decision${b2.decisions > 1 ? "s" : ""} in one batch · ${Math.round(b2.forward_ms)} ms forward`
                               : "no model call this tick";
    text(`${Run.fmtTime(clock)} recorded time · ${batch}`, 1880, 94, K.muted, 17, 400, UI, "right");
  }

  function drawOrder() {
    panel(40, 112, 912, 72);
    spacing(2); text("STANDING ORDER", 60, 142, K.cyan, 15, 700); spacing(0);
    const lines = wrap(R.header.standing_order, 700, 21, 600);
    lines.slice(0, 2).forEach((l, k) => text(l, 232, 142 + k * 26 - (lines.length > 1 ? 0 : -8), K.ink, 21, 600));
  }

  function drawHud(w) {
    const a = w.agent, y = 204;
    const bar = (label, x, v, col) => {
      spacing(1.5); text(label, x, y + 22, K.muted, 14, 700); spacing(0);
      const bx = x + 74; rrect(bx, y + 10, 130, 14, 7); ctx.fillStyle = "#0b1522"; ctx.fill();
      if (v > 0) { rrect(bx, y + 10, Math.max(14, 130 * v / 100), 14, 7); ctx.fillStyle = col; ctx.fill(); }
      text(String(v), bx + 140, y + 23, K.ink, 18, 700, MONO);
    };
    const hc = a.health > 60 ? K.mint : a.health > 30 ? K.amber : K.red;
    const ec = a.energy >= 35 ? K.cyan : a.energy >= 15 ? K.amber : K.red;
    bar("HEALTH", 40, a.health, hc);
    bar("ENERGY", 300, a.energy, ec);
    gemGlyph(572, y + 16, 10); text(`${a.gems}`, 590, y + 23, K.ink, 18, 700, MONO);
    spacing(1.5); text("GEMS", 620, y + 22, K.muted, 14, 700); spacing(0);
    if (a.has_key) keyGlyph(700, y + 16, 16); else { ctx.globalAlpha = 0.3; keyGlyph(700, y + 16, 16); ctx.globalAlpha = 1; }
    spacing(1.5); text(a.has_key ? "KEY CARRIED" : "NO KEY", 722, y + 22, a.has_key ? K.amber : K.muted, 14, 700); spacing(0);
    text(`${w.seen.length}/9`, 952, y + 23, K.ink, 18, 700, MONO, "right");
    spacing(1.5); text("ROOMS", 906, y + 22, K.muted, 14, 700, UI, "right"); spacing(0);
  }

  function drawMap({ i, f, tick: t, world, after, p }) {
    const map = R.map, now = performance.now() / 1000;
    ctx.drawImage(R.base, MX, MY);
    const seen = new Set(world.seen);
    const e = ease(f);
    // Items (drawn under the fog: the viewer sees them, the agent may not).
    const it = world.items;
    for (const [x, y] of it.gems) gemGlyph(cx(x), cy(y), 11);
    for (const [x, y] of it.food) foodGlyph(cx(x), cy(y), 11);
    for (const [x, y] of it.potions) potionGlyph(cx(x), cy(y), 11);
    if (it.key) keyGlyph(cx(it.key[0]), cy(it.key[1]), 14);
    const escaped = R.end && R.end.outcome === "escaped" && i === R.n - 1 && f > 0.85;
    exitGlyph(cx(map.exit[0]), cy(map.exit[1]), 14, !world.agent.has_key, world.agent.has_key || escaped);
    // Enemies, interpolated between this tick and the next.
    const nextEnemies = new Map(after.enemies.map((x) => [x.id, x]));
    for (const en of t.world.enemies) {
      const nx = nextEnemies.get(en.id) || en;
      const mode = (f < 0.5 ? en : nx).mode;
      enemyGlyph(lerp(cx(en.pos[0]), cx(nx.pos[0]), e), lerp(cy(en.pos[1]), cy(nx.pos[1]), e), 14, mode, now);
    }
    // Fog: rooms and corridors the agent has not seen.
    for (const r of map.rooms) {
      if (seen.has(r.id)) continue;
      ctx.fillStyle = "rgba(6,10,17,0.68)";
      ctx.fillRect(MX + (r.x0 - 1) * CS, MY + (r.y0 - 1) * CS, (r.w + 2) * CS, (r.h + 2) * CS);
    }
    for (const c of map.corridors) {
      if (seen.has(c.rooms[0]) || seen.has(c.rooms[1])) continue;
      ctx.fillStyle = "rgba(6,10,17,0.68)";
      for (const [x, y] of c.cells.concat(c.doors)) ctx.fillRect(MX + x * CS, MY + y * CS, CS, CS);
    }
    for (const r of map.rooms) {
      spacing(1.5);
      const label = r.name.toUpperCase() + (seen.has(r.id) ? "" : " · UNSEEN");
      text(label, MX + r.x0 * CS + 4, MY + r.y0 * CS - 6, seen.has(r.id) ? K.muted : K.faint, 13, 700);
      spacing(0);
    }
    // Recent path, so loops and oscillation are visible.
    const trail = [];
    for (let k = Math.max(0, i - 14); k <= i; k++) trail.push(R.ticks[k].world.agent.pos);
    trail.forEach(([x, y], k) => {
      ctx.fillStyle = alpha(K.mint, 0.06 + 0.3 * (k / trail.length));
      ctx.beginPath(); ctx.arc(cx(x), cy(y), 4, 0, 7); ctx.fill();
    });
    // Agent and its current target.
    const a0 = t.world.agent.pos, a1 = after.agent.pos;
    const ax = lerp(cx(a0[0]), cx(a1[0]), e), ay = lerp(cy(a0[1]), cy(a1[1]), e);
    if (t.target) {
      const tx = cx(t.target[0]), ty = cy(t.target[1]);
      ctx.setLineDash([6, 6]); ctx.strokeStyle = alpha(K.cyan, 0.7); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
      const pr = 15 + 3 * Math.sin(now * 5);
      ctx.strokeStyle = K.cyan; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(tx, ty, pr, 0, 7); ctx.stroke();
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        ctx.beginPath(); ctx.moveTo(tx + dx * (pr - 5), ty + dy * (pr - 5)); ctx.lineTo(tx + dx * (pr + 6), ty + dy * (pr + 6)); ctx.stroke();
      }
    }
    const step = { "move north": [0, -1], "move south": [0, 1], "move east": [1, 0], "move west": [-1, 0] }[t.move] || null;
    const hurt = recentAge(i, f, "hit", 1.2);
    agentGlyph(ax, ay, 14, step, hurt);
    // Event pop-ups: events of tick k appear as its move lands.
    for (let k = Math.max(0, i - 3); k <= i; k++) {
      const age = p - (k + 0.85);
      if (age < 0 || age > 2.5) continue;
      for (const ev of R.ticks[k].events) popup(ev, age, R.ticks[k + 1] ? R.ticks[k + 1].world.agent.pos : a1);
    }
    if (hurt > 0) {
      const g = ctx.createRadialGradient(MX + 456, MY + 399, 250, MX + 456, MY + 399, 620);
      g.addColorStop(0, "rgba(242,102,94,0)"); g.addColorStop(1, `rgba(242,102,94,${0.35 * hurt})`);
      ctx.fillStyle = g; ctx.fillRect(MX, MY, map.width * CS, map.height * CS);
    }
    const s = t.stuck_streak;
    if (s >= 6) {
      pill(`NO PROGRESS FOR ${s} TICKS`, MX + map.width * CS - 10, MY + 24, K.void, K.red);
    }
  }

  function recentAge(i, f, kind, span) {
    let best = 0;
    for (let k = Math.max(0, i - 3); k <= i; k++) {
      const age = i + f - (k + 0.85);
      if (age >= 0 && age < span && R.ticks[k].events.some((e) => e.kind === kind)) best = Math.max(best, 1 - age / span);
    }
    return best;
  }

  function popup(ev, age, pos) {
    const labels = { gem: ["+1 GEM", K.violet], food: [`+${ev.gain} ENERGY`, K.orange], potion: [`+${ev.gain} HEALTH`, K.pink],
      key: ["KEY!", K.amber], hit: [`−${ev.damage} HEALTH`, K.red], starving: [`−${ev.damage} STARVING`, K.red],
      exit_locked: ["LOCKED: NO KEY", K.amber], bump: ["BUMP", K.faint], blocked: ["BLOCKED", K.faint] };
    let where = pos, label, col;
    if (ev.kind === "room_seen") {
      const r = R.map.rooms[ev.room];
      where = [r.x0 + (r.w - 1) / 2, r.y0 + (r.h - 1) / 2]; label = r.name.toUpperCase() + " DISCOVERED"; col = K.cyan;
    } else if (labels[ev.kind]) [label, col] = labels[ev.kind]; else return;
    const a = age < 0.2 ? age / 0.2 : 1 - (age - 0.2) / 2.3;
    ctx.globalAlpha = Math.max(0, a);
    spacing(1.5);
    font(ev.kind === "room_seen" ? 20 : 18, 800);
    ctx.lineWidth = 5; ctx.strokeStyle = K.void; ctx.textAlign = "center";
    const y = cy(where[1]) - 26 - age * 18;
    ctx.strokeText(label, cx(where[0]), y); ctx.fillStyle = col; ctx.fillText(label, cx(where[0]), y);
    spacing(0); ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- decision cards

  function cardInfo(tier, i) {
    const t = R.ticks[i];
    const here = t.decisions.filter((d) => d.tier === tier);
    const goal = t.goals[tier] || {};
    if (here.length) {
      const d = here[here.length - 1];
      let status, color, bg;
      if (d.method === "only_option") { status = "ONLY OPTION · NO MODEL CALL"; color = K.void; bg = K.muted; }
      else if (d.tournament && !d.tournament.done) { status = "DECIDING · TOURNAMENT"; color = K.void; bg = K.amber; }
      else if (tier === "action") { status = `DECIDED · TICK ${t.tick}`; color = K.void; bg = K.mint; }
      else {
        const prev = i > 0 ? (R.ticks[i - 1].goals[tier] || {}).choice : null;
        const changed = prev !== goal.choice;
        status = changed ? `NEW · TICK ${t.tick}` : `SAME ANSWER · TICK ${t.tick}`;
        color = K.void; bg = changed ? K.mint : K.muted;
      }
      return { d, at: i, fresh: true, status, color, bg };
    }
    const k = R.last[tier][i];
    if (goal.waiting) return { d: Run.held(R, tier, i), at: k, fresh: false, status: "DECIDING…", color: K.void, bg: K.amber, blink: true };
    if (k < 0) return { d: null, at: -1, fresh: false, status: "NOT DECIDED YET", color: K.ink, bg: K.slate };
    return { d: Run.held(R, tier, i), at: k, fresh: false, status: `HELD · SINCE TICK ${goal.since ?? R.ticks[k].tick}`, color: K.ink, bg: K.slate };
  }

  function drawCard(tier, i) {
    const r = CARDS[tier], info = cardInfo(tier, i), t = R.ticks[i];
    const meta = R.header.tiers.find((x) => x.name === tier) || { every: 1, kind: "control" };
    panel(r.x, r.y, r.w, r.h, info.fresh ? K.panel : "#0f1a28");
    spacing(2.5); text(TIER_LABEL[tier].toUpperCase(), r.x + 20, r.y + 32, K.cyan, 20, 800); spacing(0);
    font(20, 800); spacing(2.5); const lw = ctx.measureText(TIER_LABEL[tier].toUpperCase()).width; spacing(0);
    const cadence = meta.kind === "plan" ? `PLAN · every ${meta.every} ticks, or at once when needed` : "CONTROL · every tick";
    text(cadence, r.x + 34 + lw, r.y + 32, K.muted, 15, 500);
    if (!info.blink || Math.floor(performance.now() / 450) % 2 === 0 || !playing) pill(info.status, r.x + r.w - 18, r.y + 26, info.color, info.bg);
    else pill(info.status, r.x + r.w - 18, r.y + 26, info.color, alpha(K.amber, 0.55));
    const d = info.d;
    if (!d) {
      text(meta.question || "", r.x + 20, r.y + 64, K.muted, 17);
      text("Waiting for the first decision.", r.x + 20, r.y + 100, K.faint, 17);
      return;
    }
    const dim = info.fresh ? 1 : 0.62;
    ctx.globalAlpha = dim;
    let y = r.y + 62;
    text(d.question, r.x + 20, y, K.ink, 17, 500);
    y += 8;
    let note = null;
    const prog = (t.tournaments || {})[tier];
    if (info.fresh && d.tournament) {
      const tr = d.tournament;
      note = tr.done ? `Tournament final (round ${tr.round}) over ${tr.options} options: the winner below is committed.`
        : `Tournament over ${tr.options} options · round ${tr.round}` + (prog ? ` · ${prog.resolved}/${prog.contested} groups done` : "") +
          ` · this group: ${d.options.length} options · group winners meet in the next round`;
    } else if (info.blink) {
      note = prog ? `Tournament over ${prog.options} options in progress: round ${prog.round}, ${prog.resolved}/${prog.contested} groups done. Last answer shown dimmed.`
        : `A new answer is due on a later tick (at most ${R.header.plan_budget ?? "∞"} planning decision per tick). Last answer shown dimmed.`;
    }
    if (note) {
      ctx.globalAlpha = 1;
      text(fit(note, r.w - 40, 15, 600)[0], r.x + 20, y + 16, K.amber, 15, 600);
      ctx.globalAlpha = dim;
      y += 22;
    }
    const rowH = 22, top = y + 8;
    d.options.forEach((opt, k) => {
      const yy = top + k * rowH, chosen = k === d.choice, pr = d.probs[k];
      if (chosen) { rrect(r.x + 12, yy, r.w - 24, rowH - 2, 5); ctx.fillStyle = alpha(K.mint, 0.1); ctx.fill(); }
      const lab = d.labels[k] || "·";
      text(lab, r.x + 34, yy + 16, chosen ? K.mint : K.faint, 15, 700, MONO, "center");
      const [s, sz] = fit(opt, 470, 17, chosen ? 700 : 400);
      text(s, r.x + 58, yy + 16, chosen ? K.mint : "#b9c9d8", sz, chosen ? 700 : 400);
      const bx = r.x + 548, bw = 230;
      rrect(bx, yy + 6, bw, 9, 4.5); ctx.fillStyle = "#0b1522"; ctx.fill();
      if (pr > 0.002) { rrect(bx, yy + 6, Math.max(9, bw * pr), 9, 4.5); ctx.fillStyle = chosen ? K.mint : K.slate; ctx.fill(); }
      const pct = pr >= 0.995 ? "100%" : pr < 0.005 ? "<1%" : Math.round(pr * 100) + "%";
      text(pct, r.x + r.w - 20, yy + 16, chosen ? K.mint : K.muted, 16, 700, MONO, "right");
    });
    const src = R.ticks[info.at] || t;
    let foot;
    if (d.method === "only_option") foot = `committed without a model call at tick ${src.tick}`;
    else {
      const b = src.batch;
      foot = `answer ${d.labels[d.choice]} · p ${d.probs[d.choice].toFixed(2)} · tick ${src.tick} batch: ${b.decisions} decision${b.decisions > 1 ? "s" : ""}, ` +
        `${Math.round(b.forward_ms)} ms · ${d.prompt_tokens} prompt tokens · mass outside the labels ${(100 * d.outside).toFixed(1)}%`;
    }
    const [fs, fsz] = fit(foot, r.w - 40, 14, 400, MONO, 11);
    text(fs, r.x + 20, r.y + r.h - 14, K.muted, fsz, 400, MONO);
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- timeline

  function drawTimelineBase(run) {
    const off = document.createElement("canvas"); off.width = TL.w; off.height = TL.h;
    const g = off.getContext("2d");
    const n = run.n, bw = TL.w / n, top = 34, bot = TL.h - 18, ph = bot - top;
    const cap = Math.max(300, Math.ceil(run.fwMax / 100) * 100);
    g.fillStyle = K.panel; g.fillRect(0, 0, TL.w, TL.h);
    g.strokeStyle = K.muted; g.setLineDash([3, 4]);
    const ref = bot - ph * 200 / cap;  // one reference line: the plot is short
    g.beginPath(); g.moveTo(0, ref); g.lineTo(TL.w, ref); g.stroke(); g.setLineDash([]);
    run.ticks.forEach((t, k) => {
      const x = k * bw, ms = t.batch.forward_ms;
      const plan = t.decisions.some((d) => d.kind === "plan" && d.method !== "only_option");
      const h = ph * Math.min(ms, cap) / cap;
      g.fillStyle = plan ? K.cyan : K.slate; g.fillRect(x, bot - h, Math.max(1, bw - (bw > 3 ? 1 : 0)), h);
      if (t.stuck_streak >= 6) { g.fillStyle = K.red; g.fillRect(x, bot + 2, Math.max(1, bw), 4); }
      for (const ev of t.events) {
        const col = { key: K.amber, hit: K.red, died: K.red, escaped: K.mint, food: K.orange, potion: K.pink, gem: K.violet, room_seen: K.cyan }[ev.kind];
        if (!col) continue;
        const big = ["key", "hit", "died", "escaped"].includes(ev.kind);
        g.fillStyle = col; g.beginPath(); g.arc(x + bw / 2, big ? 14 : 27, big ? 4.5 : 3, 0, 7); g.fill();
      }
    });
    g.font = `400 11px ${MONO}`; g.fillStyle = K.muted; g.textAlign = "right";
    g.fillText("200 ms", TL.w - 4, ref - 3);
    g.fillText(`tick ${n - 1}`, TL.w - 4, TL.h - 4); g.textAlign = "left"; g.fillText("tick 0", 4, TL.h - 4);
    return off;
  }

  function drawTimeline(p) {
    spacing(2); text("FORWARD PASS PER TICK", TL.x, TL.y - 8, K.cyan, 13, 700); spacing(0);
    text(`median ${Math.round(R.fwMedian)} ms · p90 ${Math.round(R.fwP90)} ms · max ${Math.round(R.fwMax)} ms`,
      TL.x + TL.w, TL.y - 8, K.muted, 13, 400, MONO, "right");
    ctx.save(); rrect(TL.x, TL.y, TL.w, TL.h, 8); ctx.clip(); ctx.drawImage(R.timeline, TL.x, TL.y); ctx.restore();
    rrect(TL.x, TL.y, TL.w, TL.h, 8); ctx.strokeStyle = K.edge; ctx.lineWidth = 1; ctx.stroke();
    const x = TL.x + TL.w * (p / R.n);
    ctx.fillStyle = K.ink; ctx.fillRect(x - 1, TL.y, 2, TL.h);
    const legend = [["plan + control", K.cyan], ["control only", K.slate], ["no progress 6+ ticks", K.red]];
    let lx = TL.x + 236;
    for (const [s, c] of legend) {
      ctx.fillStyle = c; ctx.fillRect(lx, TL.y - 18, 10, 10); text(s, lx + 15, TL.y - 8, K.muted, 12, 500); font(12, 500); lx += 30 + ctx.measureText(s).width;
    }
  }

  function drawOutcome(i, f) {
    if (!R.end || i < R.n - 1 || f < 0.9) return;
    const o = R.end.outcome, s = R.end.summary || {};
    const [word, col] = o === "escaped" ? ["ESCAPED", K.mint] : o === "died" ? [`DIED · ${String(R.end.cause || "").toUpperCase()}`, K.red]
      : o === "timeout" ? ["OUT OF TIME", K.amber] : ["STOPPED", K.muted];
    const bx = MX + 106, by = MY + 300, bw = 700, bh = 150;
    ctx.fillStyle = "rgba(7,12,20,0.86)"; rrect(bx, by, bw, bh, 12); ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
    spacing(4); text(word, bx + bw / 2, by + 70, col, 50, 800, UI, "center"); spacing(0);
    text(`tick ${R.end.tick} · ${s.gems ?? 0} gems · ${s.hits ?? 0} hits taken · ${s.rooms_seen ?? "?"}/9 rooms seen`,
      bx + bw / 2, by + 112, K.ink, 20, 500, UI, "center");
  }

  // ---------------------------------------------------------------- About this run (the shell renders these)

  function verdict(run) {
    const e = run.end;
    const cause = !e || !e.cause ? "" : e.cause === "starvation" ? "· starved " : e.cause === "enemy" ? "· caught by the enemy " : `· ${e.cause} `;
    return !e ? ["Unfinished", "amber", "the trace has no end record"]
      : e.outcome === "escaped" ? ["Escaped", "mint", `at tick ${e.tick}`]
      : e.outcome === "died" ? ["Died", "red", `${cause}at tick ${e.tick}`]
      : e.outcome === "timeout" ? ["Out of time", "amber", `stopped at tick ${e.tick}`]
      : [e.outcome, "amber", `at tick ${e.tick}`];
  }

  // Facts as titled cards of [label, value, hint?] rows. Keep values short; explanations go in the hint.
  function facts(run) {
    const s = (run.end && run.end.summary) || {}, m = run.map, v = (x) => x ?? "?";
    const room = (k) => { const n = m.rooms[m[k]].name; return n[0].toUpperCase() + n.slice(1); };
    const br = run.header.brain;
    return [
      ["Map", [["Seed", m.seed], ["Start", room("start_room")], ["Key room", room("key_room")], ["Exit", room("exit_room")]]],
      ["Progress", [
        ["Key", s.key_tick != null ? `picked up at tick ${s.key_tick}` : "never picked up"],
        ["Gems", v(s.gems)],
        ["Rooms seen", `${v(s.rooms_seen)} of ${m.rooms.length}`],
        ["Stuck ticks", v(s.stuck_ticks), "ticks in streaks of 6+ without getting closer to the same target"],
      ]],
      ["Survival", [["Hits taken", v(s.hits)], ["Food eaten", v(s.food_eaten)], ["Potions drunk", v(s.potions_drunk)]]],
      ["Model", [
        ["Model decisions", v(s.model_decisions), `${v(s.planning_decisions)} planning, ${v(s.tournament_decisions)} in tournaments`],
        ["Only-option commits", v(s.only_option_commits), "no model call needed"],
        ["Forward pass", `${Math.round(run.fwMedian)} ms median`, `per tick; p90 ${Math.round(run.fwP90)} ms, max ${Math.round(run.fwMax)} ms`],
        ["Planning budget", `${run.header.plan_budget ?? "unlimited"} per tick`, `tournament groups of ${run.header.group_size}`],
        ["Move wording", br ? br.label_style : "not recorded", br && br.enemy_aware ? "enemy-aware" : null],
        ["Captured", String(run.header.created || "?").replace("T", " ")],
      ]],
    ];
  }

  function notes() {
    return [
      ["Reading the map", ["Dimmed rooms: not seen by the agent yet. You see the whole map; the agent does not."]],
      ["Reading the decisions", [
        "Every decision was made by the model while the run was captured. This page only replays the recorded trace; it does not recompute or re-decide anything.",
        "Probabilities are the model's next-token softmax over the option labels. They are not calibrated.",
        "\"Only option\": one goal or target was possible, so it was committed without a model call.",
      ]],
    ];
  }

  Games.add({
    scenario: "dungeon",
    title: "Dungeon",
    tiers: Object.keys(CARDS),
    tierLabels: TIER_LABEL,
    inspectNote: "Each decision below was one row in that tick's batched forward pass. The prompt is rebuilt from the " +
      "recorded parts in the engine's layout; the exact chat template is under Prompt template.",
    exampleFrom: "action tier",
    prepare, drawStage, pane, drawPane, verdict, facts, notes,
  });
})();
