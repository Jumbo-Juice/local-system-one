// The dungeon demo's viewer (demo/dungeon, rebuilt 2026-09-30): draws one tick of a recorded run.
// Stage: 1920x1080, map and HUD on the left, one decision card per tier on the right, the latency
// timeline under the cards. The auto-demo panes show the HUD and map only (pane()).
"use strict";

(() => {
  const { UI, MONO, font, spacing, text, fit, wrap, rrect, panel, pill, lerp, ease, alpha, mix } = Paint;
  const W = 1920, H = 1080, CS = 27, MX = 40, MY = 250;
  // A darker, greener crypt than the shooter, with the same contrast rules: floors lighter than walls.
  const K = {
    ground: "#0a1014", void: "#04070a", panel: "#131d24", panelDim: "#0f171d", edge: "#2a3b47",
    ink: "#edf4f2", muted: "#aebfc0", faint: "#7f9396", cyan: "#6fd8f5", mint: "#7cf5c4", amber: "#ffc94d",
    red: "#ff5a5a", violet: "#b99bff", pink: "#ff86b0", slate: "#475d68", track: "#081014", orange: "#ff9f43",
    wall: "#121b20", wallTop: "#26363f", wallFace: "#0a1115", floor: "#2f434b", floorLine: "#3a5159",
    corridor: "#28393f", jamb: "#c9a86a", ghoul: "#9be36b", dash: "#e6fff5",
  };
  const CARDS = {
    strategy: { x: 992, y: 112, w: 888, h: 206 },
    target: { x: 992, y: 330, w: 888, h: 280 },
    move: { x: 992, y: 622, w: 888, h: 322 },
  };
  const TL = { x: 992, y: 976, w: 888, h: 76 };
  const TIER_LABEL = { strategy: "Strategy", target: "Target", move: "Move" };
  const ROW = 22, STUCK = 6, LOOP = 30;
  const STEP = { "move north": [0, -1], "move south": [0, 1], "move east": [1, 0], "move west": [-1, 0],
    "dash north": [0, -1], "dash south": [0, 1], "dash east": [1, 0], "dash west": [-1, 0] };

  let R = null, ctx = null, clock = 0, playing = false;
  function begin(context, run, time, isPlaying) {
    ctx = context; R = run; clock = time; playing = isPlaying;
    Paint.use(context, K);
  }

  // ---------------------------------------------------------------- per-run preparation

  function prepare(run) {
    run.rules = run.map.rules;
    run.loop = (run.header.brain && run.header.brain.loop) || LOOP;
    run.open = openCells(run.map);
    run.base = drawBase(run);
    run.timeline = drawTimelineBase(run);
  }

  function openCells(map) {
    const kind = new Map();  // "x,y" -> room | door | corridor
    for (const r of map.rooms) for (let y = r.y0; y < r.y0 + r.h; y++) for (let x = r.x0; x < r.x0 + r.w; x++) kind.set(x + "," + y, "room");
    for (const c of map.corridors) {
      for (const [x, y] of c.cells) kind.set(x + "," + y, "corridor");
      for (const door of c.doors) for (const [x, y] of door) kind.set(x + "," + y, "door");
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
        if (isOpen(x, y + 1)) { g.fillStyle = K.wallFace; g.fillRect(px, py + CS - 7, CS, 7); }
        continue;
      }
      g.fillStyle = k === "corridor" ? K.corridor : K.floor; g.fillRect(px, py, CS, CS);
      g.strokeStyle = K.floorLine; g.lineWidth = 1; g.strokeRect(px + 0.5, py + 0.5, CS - 1, CS - 1);
    }
    g.fillStyle = K.jamb;  // a post on each side of every three-cell doorway
    for (const c of map.corridors) for (const door of c.doors) {
      const xs = door.map((d) => d[0]), ys = door.map((d) => d[1]);
      if (new Set(xs).size === 1) {
        const x = xs[0] * CS, y0 = Math.min(...ys) * CS, y1 = (Math.max(...ys) + 1) * CS;
        g.fillRect(x, y0 - 4, CS, 4); g.fillRect(x, y1, CS, 4);
      } else {
        const y = ys[0] * CS, x0 = Math.min(...xs) * CS, x1 = (Math.max(...xs) + 1) * CS;
        g.fillRect(x0 - 4, y, 4, CS); g.fillRect(x1, y, 4, CS);
      }
    }
    return off;
  }

  // ---------------------------------------------------------------- glyphs

  function gemGlyph(x, y, s) {
    ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s * 0.8, y - s * 0.2); ctx.lineTo(x, y + s); ctx.lineTo(x - s * 0.8, y - s * 0.2); ctx.closePath();
    ctx.fillStyle = K.violet; ctx.fill();
    ctx.beginPath(); ctx.moveTo(x - s * 0.8, y - s * 0.2); ctx.lineTo(x + s * 0.8, y - s * 0.2); ctx.strokeStyle = "#e4dbff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  function potionGlyph(x, y, s) {
    ctx.fillStyle = K.pink;
    ctx.beginPath(); ctx.arc(x, y + 3, s * 0.7, 0, 7); ctx.fill();
    ctx.fillRect(x - 3, y - s * 0.95, 6, s * 0.8);
    ctx.fillStyle = "#e0bb8a"; ctx.fillRect(x - 4, y - s * 1.1, 8, 4);
    ctx.fillStyle = "#ffe0ec"; ctx.beginPath(); ctx.arc(x - 3, y + 1, 2.2, 0, 7); ctx.fill();
  }
  function keyGlyph(x, y, s) {
    ctx.strokeStyle = K.amber; ctx.lineWidth = 3.5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(x - s * 0.45, y, s * 0.38, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - s * 0.08, y); ctx.lineTo(x + s * 0.9, y); ctx.moveTo(x + s * 0.55, y); ctx.lineTo(x + s * 0.55, y + s * 0.4);
    ctx.moveTo(x + s * 0.85, y); ctx.lineTo(x + s * 0.85, y + s * 0.35); ctx.stroke(); ctx.lineCap = "butt";
  }
  function lockGlyph(x, y) {
    ctx.fillStyle = K.amber; rrect(x - 6, y - 2, 12, 10, 2); ctx.fill();
    ctx.strokeStyle = K.amber; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y - 3, 4.5, Math.PI, 0); ctx.stroke();
  }
  function exitGlyph(x, y, s, locked, lit) {
    ctx.save();
    if (lit) { ctx.shadowColor = K.mint; ctx.shadowBlur = 24; }
    for (let i = 0; i < 4; i++) {
      ctx.fillStyle = alpha(K.mint, 0.45 + 0.17 * i);
      ctx.fillRect(x - s + i * s * 0.25, y - s + i * s * 0.5, s * 2 - i * s * 0.25, s * 0.42);
    }
    ctx.restore();
    if (locked) lockGlyph(x + s * 0.55, y + s * 0.45);
  }
  function sleepGlyph(x, y, t) {
    const b = Math.sin(t * 2) * 2;
    text("z", x, y + b, K.ink, 13, 800); text("z", x + 7, y - 7 + b, K.ink, 10, 800);
  }
  // A ghoul: a hooded wisp. Bright and glowing while it chases; grey while asleep; dim while tired
  // or backing off.
  function ghoulGlyph(x, y, s, g, t) {
    const calm = !g.awake || g.resting || g.tired;
    const col = !g.awake ? mix(K.ghoul, "#2f434b", 0.55) : calm ? mix(K.ghoul, "#2f434b", 0.35) : K.ghoul;
    ctx.save();
    if (!calm) { ctx.shadowColor = K.ghoul; ctx.shadowBlur = 16; }
    ctx.beginPath();
    ctx.arc(x, y - s * 0.15, s * 0.85, Math.PI, 0);
    const wave = calm ? 0 : Math.sin(t * 6) * 2;
    ctx.lineTo(x + s * 0.85, y + s * 0.8);
    for (let k = 3; k >= 0; k--) ctx.lineTo(x - s * 0.85 + k * s * 0.57, y + s * (k % 2 ? 0.55 : 0.85) + wave);
    ctx.closePath(); ctx.fillStyle = col; ctx.fill(); ctx.restore();
    ctx.lineWidth = 2; ctx.strokeStyle = K.void; ctx.stroke();
    ctx.fillStyle = K.void;
    ctx.beginPath(); ctx.arc(x - s * 0.3, y - s * 0.15, s * 0.16, 0, 7); ctx.arc(x + s * 0.3, y - s * 0.15, s * 0.16, 0, 7); ctx.fill();
    if (!g.awake) sleepGlyph(x + s, y - s, t);
    else if (g.tired) text("…", x + s * 0.9, y - s * 0.7, K.muted, 16, 800);
  }
  function agentGlyph(x, y, s, dir, hurt) {
    ctx.save(); ctx.shadowColor = hurt > 0 ? K.red : K.mint; ctx.shadowBlur = 16 + 20 * hurt;
    ctx.beginPath(); ctx.arc(x, y, s, 0, 7); ctx.fillStyle = hurt > 0 ? mix(K.mint, K.red, hurt) : K.mint; ctx.fill(); ctx.restore();
    ctx.lineWidth = 3; ctx.strokeStyle = K.void; ctx.beginPath(); ctx.arc(x, y, s, 0, 7); ctx.stroke();
    if (dir) {
      ctx.fillStyle = K.void; ctx.beginPath();
      ctx.arc(x + dir[0] * s * 0.5, y + dir[1] * s * 0.5, s * 0.28, 0, 7); ctx.fill();
    }
  }
  // A dash: speed lines and fading after-images from where it started to where the agent is now.
  function dashTrail(x0, y0, x1, y1, f) {
    const fade = Math.max(0, 1 - Math.max(0, f - 0.85) / 0.15);
    ctx.save(); ctx.strokeStyle = alpha(K.dash, 0.55 * fade); ctx.lineWidth = 6; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.restore(); ctx.lineCap = "butt";
    for (let k = 1; k <= 3; k++) {
      const q = k / 4;
      ctx.fillStyle = alpha(K.mint, 0.18 * fade * q);
      ctx.beginPath(); ctx.arc(lerp(x0, x1, q), lerp(y0, y1, q), 10, 0, 7); ctx.fill();
    }
  }

  // ---------------------------------------------------------------- stage

  function drawStage(context, run, time, isPlaying) {
    begin(context, run, time, isPlaying);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "#0d161b"); grad.addColorStop(1, K.ground);
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
    const rows = b2.decisions ? `${b2.decisions} decision${b2.decisions > 1 ? "s" : ""}${R.header.order_debias ? " × 2 orders" : ""} in one batch · ${Math.round(b2.forward_ms)} ms forward`
                              : "no model call this tick";
    text(`${Run.fmtTime(clock)} recorded time · ${rows}`, 1880, 94, K.muted, 17, 400, UI, "right");
  }

  function drawOrder() {
    panel(40, 112, 912, 78);
    spacing(2); text("STANDING", 60, 142, K.cyan, 15, 700); text("ORDER", 60, 162, K.cyan, 15, 700); spacing(0);
    const lines = wrap(R.header.standing_order, 740, 19, 600);
    const y0 = lines.length > 1 ? 144 : 157;
    lines.slice(0, 2).forEach((l, k) => text(l, 178, y0 + k * 25, K.ink, 19, 600));
  }

  function drawHud(w) {
    const a = w.agent, y = 204, rules = R.rules;
    spacing(1.5); text("HEALTH", 40, y + 22, K.muted, 14, 700); spacing(0);
    const bx = 112, bw = 130; rrect(bx, y + 10, bw, 14, 7); ctx.fillStyle = K.track; ctx.fill();
    const hc = a.health > 60 ? K.mint : a.health > 30 ? K.amber : K.red;
    if (a.health > 0) { rrect(bx, y + 10, Math.max(14, bw * a.health / rules.health), 14, 7); ctx.fillStyle = hc; ctx.fill(); }
    text(String(a.health), bx + bw + 8, y + 23, K.ink, 18, 700, MONO);
    // dash: a charge bar that fills over the recharge ticks
    spacing(1.5); text("DASH", 312, y + 22, K.muted, 14, 700); spacing(0);
    const dx = 362, dw = 70, ready = !a.dash;
    rrect(dx, y + 10, dw, 14, 7); ctx.fillStyle = K.track; ctx.fill();
    const frac = ready ? 1 : 1 - a.dash / rules.dash_recharge;
    if (frac > 0) { rrect(dx, y + 10, Math.max(14, dw * frac), 14, 7); ctx.fillStyle = ready ? K.dash : K.slate; ctx.fill(); }
    text(ready ? "READY" : `${a.dash}`, dx + dw + 8, y + 22, ready ? K.mint : K.amber, 14, 700);
    gemGlyph(532, y + 16, 9); text(`${a.gems}`, 548, y + 23, K.ink, 18, 700, MONO);
    spacing(1.5); text("GEMS", 574, y + 22, K.muted, 14, 700); spacing(0);
    if (a.has_key) keyGlyph(660, y + 16, 15); else { ctx.globalAlpha = 0.35; keyGlyph(660, y + 16, 15); ctx.globalAlpha = 1; }
    spacing(1.5); text(a.has_key ? "KEY CARRIED" : "NO KEY", 680, y + 22, a.has_key ? K.amber : K.muted, 14, 700); spacing(0);
    text(`${w.seen.length}/9`, 952, y + 23, K.ink, 18, 700, MONO, "right");
    spacing(1.5); text("ROOMS", 906, y + 22, K.muted, 13, 700, UI, "right"); spacing(0);
  }

  function drawMap({ i, f, tick: t, world, after, p }) {
    const map = R.map, now = performance.now() / 1000;
    ctx.drawImage(R.base, MX, MY);
    const seen = new Set(world.seen);
    const e = ease(f);
    const it = world.items;  // drawn under the fog: the viewer sees them, the agent may not
    for (const [x, y] of it.gems) gemGlyph(cx(x), cy(y), 9);
    for (const [x, y] of it.potions) potionGlyph(cx(x), cy(y), 10);
    if (it.key) keyGlyph(cx(it.key[0]), cy(it.key[1]), 12);
    const escaped = R.end && R.end.outcome === "escaped" && i === R.n - 1 && f > 0.85;
    exitGlyph(cx(map.exit[0]), cy(map.exit[1]), 12, !world.agent.has_key, world.agent.has_key || escaped);
    const a0 = t.world.agent.pos, a1 = after.agent.pos;
    const ax = lerp(cx(a0[0]), cx(a1[0]), e), ay = lerp(cy(a0[1]), cy(a1[1]), e);
    const next = new Map(after.ghouls.map((g) => [g.id, g]));
    for (const g of t.world.ghouls) {
      const ng = next.get(g.id) || g;
      ghoulGlyph(lerp(cx(g.pos[0]), cx(ng.pos[0]), e), lerp(cy(g.pos[1]), cy(ng.pos[1]), e), 10, f < 0.5 ? g : ng, now);
    }
    for (const r of map.rooms) {  // fog: rooms and corridors the agent has not seen
      if (seen.has(r.id)) continue;
      ctx.fillStyle = "rgba(4,7,10,0.64)";
      ctx.fillRect(MX + (r.x0 - 1) * CS, MY + (r.y0 - 1) * CS, (r.w + 2) * CS, (r.h + 2) * CS);
    }
    for (const c of map.corridors) {
      if (seen.has(c.rooms[0]) || seen.has(c.rooms[1])) continue;
      ctx.fillStyle = "rgba(4,7,10,0.64)";
      for (const [x, y] of c.cells.concat(...c.doors)) ctx.fillRect(MX + x * CS, MY + y * CS, CS, CS);
    }
    for (const r of map.rooms) {
      spacing(1.3);
      const label = r.name.toUpperCase() + (seen.has(r.id) ? "" : " · UNSEEN"), lx = MX + r.x0 * CS + 2, ly = MY + r.y0 * CS - 8;
      font(12, 700); const lw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(4,7,10,0.82)"; rrect(lx - 5, ly - 13, lw + 10, 18, 4); ctx.fill();
      text(label, lx, ly, seen.has(r.id) ? K.muted : K.faint, 12, 700);
      spacing(0);
    }
    const trail = [];  // recent path, so loops show
    for (let k = Math.max(0, i - 14); k <= i; k++) trail.push(R.ticks[k].world.agent.pos);
    trail.forEach(([x, y], k) => {
      ctx.fillStyle = alpha(K.mint, 0.08 + 0.35 * (k / trail.length));
      ctx.beginPath(); ctx.arc(cx(x), cy(y), 3.5, 0, 7); ctx.fill();
    });
    if (t.target) {
      const tx = cx(t.target[0]), ty = cy(t.target[1]);
      ctx.setLineDash([6, 6]); ctx.strokeStyle = alpha(K.cyan, 0.8); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
      const pr = 12 + 3 * Math.sin(now * 5);
      ctx.strokeStyle = K.cyan; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(tx, ty, pr, 0, 7); ctx.stroke();
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        ctx.beginPath(); ctx.moveTo(tx + dx * (pr - 4), ty + dy * (pr - 4)); ctx.lineTo(tx + dx * (pr + 5), ty + dy * (pr + 5)); ctx.stroke();
      }
    }
    if (t.events.some((ev) => ev.kind === "dash")) dashTrail(cx(a0[0]), cy(a0[1]), ax, ay, f);
    const hurt = recentAge(i, f, "hit", 1.2);
    agentGlyph(ax, ay, 11, STEP[t.move] || null, hurt);
    for (let k = Math.max(0, i - 3); k <= i; k++) {  // events of tick k appear as its move lands
      const age = p - (k + 0.85);
      if (age < 0 || age > 2.5) continue;
      for (const ev of R.ticks[k].events) popup(ev, age, R.ticks[k + 1] ? R.ticks[k + 1].world.agent.pos : a1);
    }
    if (hurt > 0) {
      const w = map.width * CS, h = map.height * CS;
      const g = ctx.createRadialGradient(MX + w / 2, MY + h / 2, 230, MX + w / 2, MY + h / 2, 640);
      g.addColorStop(0, "rgba(255,90,90,0)"); g.addColorStop(1, `rgba(255,90,90,${0.35 * hurt})`);
      ctx.fillStyle = g; ctx.fillRect(MX, MY, w, h);
    }
    let py = MY + 24;
    if (t.stuck_streak >= STUCK) { pill(`NO PROGRESS FOR ${t.stuck_streak} TICKS`, MX + map.width * CS - 10, py, K.void, K.red); py += 32; }
    if ((t.idle || 0) >= R.loop) pill(`GOING IN CIRCLES: ${t.idle} TICKS`, MX + map.width * CS - 10, py, K.void, K.orange);
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
    const labels = { gem: ["+1 GEM", K.violet], potion: [`+${ev.gain} HEALTH`, K.pink], key: ["KEY!", K.amber],
      hit: [`−${ev.damage} HEALTH`, K.red], exit_locked: ["LOCKED: NO KEY", K.amber], bump: ["BUMP", K.faint],
      blocked: ["BLOCKED", K.faint], dash: ["DASH", K.dash] };
    let where = pos, label, col;
    const room = (id) => R.map.rooms[id], mid = (r) => [r.x0 + (r.w - 1) / 2, r.y0 + (r.h - 1) / 2];
    const ghoulAt = (id) => { const g = R.ticks[Math.min(R.n - 1, Run.locate(R, clock)[0])].world.ghouls.find((x) => x.id === id); return g ? g.pos : pos; };
    if (ev.kind === "room_seen") { where = mid(room(ev.room)); label = room(ev.room).name.toUpperCase() + " DISCOVERED"; col = K.cyan; }
    else if (ev.kind === "ghouls_woke") { where = mid(room(ev.room)); label = ev.ghouls.length > 1 ? "GHOULS WAKE" : "A GHOUL WAKES"; col = K.ghoul; }
    else if (ev.kind === "ghoul_tired") { where = ghoulAt(ev.ghoul); label = "TIRED"; col = K.muted; }
    else if (ev.kind === "ghoul_slept") { where = ghoulAt(ev.ghoul); label = "ASLEEP"; col = K.faint; }
    else if (labels[ev.kind]) [label, col] = labels[ev.kind]; else return;
    const a = age < 0.2 ? age / 0.2 : 1 - (age - 0.2) / 2.3;
    ctx.globalAlpha = Math.max(0, a);
    spacing(1.5);
    const big = ["room_seen", "ghouls_woke"].includes(ev.kind);
    font(big ? 20 : 17, 800);
    ctx.lineWidth = 5; ctx.strokeStyle = K.void; ctx.textAlign = "center";
    const y = cy(where[1]) - 22 - age * 16;
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
      let status, bg;
      if (d.method === "only_option") { status = "ONLY OPTION · NO MODEL CALL"; bg = K.muted; }
      else if (d.tournament && !d.tournament.done) { status = "DECIDING · TOURNAMENT"; bg = K.amber; }
      else if (tier === "move") { status = `DECIDED · TICK ${t.tick}`; bg = K.mint; }
      else {
        const prev = i > 0 ? (R.ticks[i - 1].goals[tier] || {}).choice : null;
        const changed = prev !== goal.choice;
        status = changed ? `NEW · TICK ${t.tick}` : `SAME ANSWER · TICK ${t.tick}`;
        bg = changed ? K.mint : K.muted;
      }
      return { d, at: i, fresh: true, status, color: K.void, bg };
    }
    const k = R.last[tier][i];
    if (goal.waiting) return { d: Run.held(R, tier, i), at: k, fresh: false, status: "DECIDING…", color: K.void, bg: K.amber, blink: true };
    if (k < 0) return { d: null, at: -1, fresh: false, status: "NOT DECIDED YET", color: K.ink, bg: K.slate };
    return { d: Run.held(R, tier, i), at: k, fresh: false, status: `HELD · SINCE TICK ${goal.since ?? R.ticks[k].tick}`, color: K.ink, bg: K.slate };
  }

  function drawCard(tier, i) {
    const r = CARDS[tier], info = cardInfo(tier, i), t = R.ticks[i];
    const meta = R.header.tiers.find((x) => x.name === tier) || { every: 1, kind: "control" };
    panel(r.x, r.y, r.w, r.h, info.fresh ? K.panel : K.panelDim);
    spacing(2.5); text(TIER_LABEL[tier].toUpperCase(), r.x + 20, r.y + 30, K.cyan, 19, 800); spacing(0);
    font(19, 800); spacing(2.5); const lw = ctx.measureText(TIER_LABEL[tier].toUpperCase()).width; spacing(0);
    const cadence = meta.kind === "plan" ? "PLAN · held until reached, gone or stalled" : "CONTROL HEAD · every tick";
    text(cadence, r.x + 34 + lw, r.y + 30, K.muted, 14, 500);
    const blinkOff = info.blink && playing && Math.floor(performance.now() / 450) % 2 === 1;
    pill(info.status, r.x + r.w - 16, r.y + 24, info.color, blinkOff ? alpha(K.amber, 0.55) : info.bg);
    const d = info.d;
    if (!d) {
      text(meta.question || "", r.x + 20, r.y + 58, K.muted, 16);
      text("Waiting for the first decision.", r.x + 20, r.y + 88, K.faint, 16);
      return;
    }
    ctx.globalAlpha = info.fresh ? 1 : 0.7;
    let y = r.y + 56;
    const [q, qs] = fit(d.question, r.w - 40, 16, 500);
    text(q, r.x + 20, y, K.ink, qs, 500);
    let note = null;
    const prog = (t.tournaments || {})[tier];
    if (info.fresh && d.tournament) {
      const tr = d.tournament;
      note = tr.done ? `Tournament final over ${tr.options} options: the winner below is committed.`
        : `Tournament over ${tr.options} options` + (prog ? ` · round ${prog.round}, ${prog.resolved}/${prog.contested} groups done` : "") + ` · this group: ${d.options.length} options`;
    } else if (info.blink) {
      note = `A new answer is due on a later tick (at most ${R.header.plan_budget ?? "∞"} planning decision per tick). Last answer shown dimmed.`;
    }
    if (note) {
      const g = ctx.globalAlpha; ctx.globalAlpha = 1;
      text(fit(note, r.w - 40, 14, 600)[0], r.x + 20, y + 19, K.amber, 14, 600);
      ctx.globalAlpha = g; y += 20;
    }
    const top = y + 8, room = Math.floor((r.y + r.h - 28 - top) / ROW);
    let rows = d.options.map((_, k) => k);
    if (rows.length > room) {  // keep the chosen option visible; the rest are in the inspector
      rows = rows.slice(0, room - 1);
      if (!rows.includes(d.choice)) rows[rows.length - 1] = d.choice;
    }
    rows.forEach((k, row) => {
      const opt = d.options[k], yy = top + row * ROW, chosen = k === d.choice, pr = d.probs[k];
      if (chosen) { rrect(r.x + 12, yy, r.w - 24, ROW - 2, 5); ctx.fillStyle = alpha(K.mint, 0.13); ctx.fill(); }
      text(d.labels[k] || "·", r.x + 32, yy + 16, chosen ? K.mint : K.faint, 14, 700, MONO, "center");
      const [s, sz] = fit(opt, 560, 16, chosen ? 700 : 400);
      text(s, r.x + 52, yy + 16, chosen ? K.mint : "#d3e0e0", sz, chosen ? 700 : 400);
      const bx = r.x + 628, bw = 180;
      rrect(bx, yy + 7, bw, 8, 4); ctx.fillStyle = K.track; ctx.fill();
      if (pr > 0.002) { rrect(bx, yy + 7, Math.max(8, bw * pr), 8, 4); ctx.fillStyle = chosen ? K.mint : K.slate; ctx.fill(); }
      if (d.orders) for (const o of d.orders) {
        ctx.fillStyle = chosen ? K.ink : K.muted; ctx.fillRect(bx + bw * o[k] - 1, yy + 4, 2, 14);
      }
      const pct = pr >= 0.995 ? "100%" : pr < 0.005 ? "<1%" : Math.round(pr * 100) + "%";
      text(pct, r.x + r.w - 18, yy + 16, chosen ? K.mint : K.muted, 15, 700, MONO, "right");
    });
    if (rows.length < d.options.length) text(`+${d.options.length - rows.length} more options below the stage`, r.x + 52, top + rows.length * ROW + 14, K.faint, 13);
    const src = R.ticks[info.at] || t;
    let foot;
    if (d.method === "only_option") foot = `committed without a model call at tick ${src.tick}`;
    else {
      const b = src.batch;
      foot = `answer ${d.labels[d.choice]} · p ${d.probs[d.choice].toFixed(2)}` + (d.orders ? " (mean of 2 orders; ticks mark each)" : "") +
        ` · tick ${src.tick} batch: ${b.decisions} decision${b.decisions > 1 ? "s" : ""}, ${Math.round(b.forward_ms)} ms · ${d.prompt_tokens} prompt tokens`;
    }
    const [fs, fsz] = fit(foot, r.w - 40, 13, 400, MONO, 11);
    text(fs, r.x + 20, r.y + r.h - 11, K.muted, fsz, 400, MONO);
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- timeline

  const EVENT_COL = { key: K.amber, hit: K.red, died: K.red, escaped: K.mint, potion: K.pink, gem: K.violet, room_seen: K.cyan, dash: K.dash };

  function drawTimelineBase(run) {
    const off = document.createElement("canvas"); off.width = TL.w; off.height = TL.h;
    const g = off.getContext("2d");
    const n = run.n, bw = TL.w / n, bot = TL.h - 16, ph = bot - 30;
    const cap = Math.max(400, Math.ceil(run.fwMax / 100) * 100);
    g.fillStyle = K.panel; g.fillRect(0, 0, TL.w, TL.h);
    g.strokeStyle = K.faint; g.setLineDash([3, 4]);
    const ref = bot - ph * 200 / cap;
    g.beginPath(); g.moveTo(0, ref); g.lineTo(TL.w, ref); g.stroke(); g.setLineDash([]);
    run.ticks.forEach((t, k) => {
      const x = k * bw, ms = t.batch.forward_ms;
      const plan = t.decisions.some((d) => d.kind === "plan" && d.method !== "only_option");
      const h = ph * Math.min(ms, cap) / cap;
      g.fillStyle = plan ? K.cyan : K.slate; g.fillRect(x, bot - h, Math.max(1, bw - (bw > 3 ? 1 : 0)), h);
      if (t.stuck_streak >= STUCK) { g.fillStyle = K.red; g.fillRect(x, bot + 2, Math.max(1, bw), 4); }
      if ((t.idle || 0) >= run.loop) { g.fillStyle = K.orange; g.fillRect(x, bot + 7, Math.max(1, bw), 3); }
      for (const ev of t.events) {
        const col = EVENT_COL[ev.kind];
        if (!col) continue;
        const big = ["key", "hit", "died", "escaped"].includes(ev.kind);
        g.fillStyle = col; g.beginPath(); g.arc(x + bw / 2, big ? 11 : 23, big ? 4 : 2.8, 0, 7); g.fill();
      }
    });
    g.font = `400 11px ${MONO}`; g.fillStyle = K.muted; g.textAlign = "right";
    g.fillText("200 ms", TL.w - 4, ref - 3);
    g.fillText(`tick ${n - 1}`, TL.w - 4, TL.h - 3); g.textAlign = "left"; g.fillText("tick 0", 4, TL.h - 3);
    return off;
  }

  function drawTimeline(p) {
    spacing(2); text("FORWARD PASS PER TICK", TL.x, TL.y - 8, K.cyan, 13, 700); spacing(0);
    text(`median ${Math.round(R.fwMedian)} · p90 ${Math.round(R.fwP90)} · max ${Math.round(R.fwMax)} ms`,
      TL.x + TL.w, TL.y - 8, K.muted, 13, 400, MONO, "right");
    ctx.save(); rrect(TL.x, TL.y, TL.w, TL.h, 8); ctx.clip(); ctx.drawImage(R.timeline, TL.x, TL.y); ctx.restore();
    rrect(TL.x, TL.y, TL.w, TL.h, 8); ctx.strokeStyle = K.edge; ctx.lineWidth = 1; ctx.stroke();
    const x = TL.x + TL.w * (p / R.n);
    ctx.fillStyle = K.ink; ctx.fillRect(x - 1, TL.y, 2, TL.h);
    const legend = [["plan + control", K.cyan], ["control only", K.slate], ["hit", K.red], ["key", K.amber], ["dash", K.dash],
      ["no progress 6+", K.red], [`in circles ${R.loop}+`, K.orange]];
    let lx = TL.x;
    const ly = TL.y + TL.h + 18;
    for (const [s, c] of legend) {
      ctx.fillStyle = c;
      if (["hit", "key", "dash"].includes(s)) { ctx.beginPath(); ctx.arc(lx + 5, ly - 5, 4, 0, 7); ctx.fill(); }
      else ctx.fillRect(lx, ly - 10, 10, 10);
      text(s, lx + 15, ly, K.muted, 12, 500); font(12, 500); lx += 28 + ctx.measureText(s).width;
    }
  }

  function drawOutcome(i, f) {
    if (!R.end || i < R.n - 1 || f < 0.9) return;
    const o = R.end.outcome, s = R.end.summary || {};
    const [word, col] = o === "escaped" ? ["ESCAPED", K.mint] : o === "died" ? ["CAUGHT BY A GHOUL", K.red]
      : o === "timeout" ? ["OUT OF TIME", K.amber] : ["STOPPED", K.muted];
    const bx = MX + 96, by = MY + 290, bw = 700, bh = 150;
    ctx.fillStyle = "rgba(4,7,10,0.9)"; rrect(bx, by, bw, bh, 12); ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
    spacing(4); text(word, bx + bw / 2, by + 70, col, o === "died" ? 42 : 50, 800, UI, "center"); spacing(0);
    text(`tick ${R.end.tick} · ${s.gems ?? 0} gems · ${s.hits ?? 0} hits taken · ${s.dashes ?? 0} dashes · ${s.rooms_seen ?? "?"}/9 rooms`,
      bx + bw / 2, by + 112, K.ink, 20, 500, UI, "center");
  }

  // ---------------------------------------------------------------- About this run (the shell renders these)

  function verdict(run) {
    const e = run.end;
    return !e ? ["Unfinished", "amber", "the trace has no end record"]
      : e.outcome === "escaped" ? ["Escaped", "mint", `at tick ${e.tick}`]
      : e.outcome === "died" ? ["Died", "red", `· caught by a ghoul at tick ${e.tick}`]
      : e.outcome === "timeout" ? ["Out of time", "amber", `stopped at tick ${e.tick}`]
      : [e.outcome, "amber", `at tick ${e.tick}`];
  }

  function facts(run) {
    const s = (run.end && run.end.summary) || {}, m = run.map, v = (x) => x ?? "?";
    const room = (k) => { const n = m.rooms[m[k]].name; return n[0].toUpperCase() + n.slice(1); };
    return [
      ["Map", [["Seed", m.seed], ["Start", room("start_room")], ["Key room", room("key_room")], ["Exit", room("exit_room")]]],
      ["Progress", [
        ["Key", s.key_tick != null ? `picked up at tick ${s.key_tick}` : "never picked up"],
        ["Rooms seen", `${v(s.rooms_seen)} of ${m.rooms.length}`],
        ["Gems", v(s.gems), "a bonus; not needed to escape"],
        ["Stuck ticks", v(s.stuck_ticks), "ticks in streaks of 6+ without getting closer to the same target"],
        ["Loop ticks", v(s.loop_ticks), `ticks in streaks of ${run.loop}+ without a new cell or any progress; longest ${v(s.longest_idle)}`],
      ]],
      ["Ghouls", [
        ["Hits taken", v(s.hits), `${run.rules.ghoul_damage} health each`],
        ["Risky moves", v(s.avoidable_risky_moves), "moves into a ghoul's reach when a safe move existed"],
        ["Hits after a safe move", v(s.hits_after_safe_move), "a move labelled safe that was not"],
        ["Dashes", v(s.dashes), `offered on ${v(s.dashes_offered)} move decisions`],
        ["Potions drunk", v(s.potions_drunk)],
      ]],
      ["Model", [
        ["Model decisions", v(s.model_decisions), `${v(s.planning_decisions)} of them planning`],
        ["Only-option commits", v(s.only_option_commits), "no model call needed"],
        ["Forward pass", `${Math.round(run.fwMedian)} ms median`, `per tick; p90 ${Math.round(run.fwP90)} ms, max ${Math.round(run.fwMax)} ms`],
        ["Order averaging", run.header.order_debias ? "on" : "off", run.header.order_debias ? "each decision read in 2 option orders" : null],
        ["Captured", String(run.header.created || "?").replace("T", " ")],
      ]],
    ];
  }

  function notes(run) {
    const r = run.rules;
    return [
      ["Reading the map", [
        "Dimmed rooms: not seen by the agent yet. You see the whole map; the agent does not.",
        "Green wisps are ghouls. Grey with a z: asleep until the agent steps into their room. Glowing: chasing. Dim: tired or backing off.",
        `A ghoul hits for ${r.ghoul_damage} when next to the agent. It never leaves its room and tires after ${r.ghoul_chase} ticks of chasing.`,
        `A white streak is a dash: up to ${r.dash_cells} cells in one tick, then ${r.dash_recharge} ticks to recharge (the DASH bar).`,
      ]],
      ["Reading the decisions", [
        "Every decision was made by the model while the run was captured. This page only replays the recorded trace.",
        "Probabilities are the model's next-token softmax over the option labels (averaged over two orders when order averaging is on). They are not calibrated.",
        "\"closer\" is measured along routes that keep out of every ghoul's reach. Moves into a ghoul's reach are offered only when nothing is safe.",
        "\"Only option\": one choice was possible, so it was committed without a model call.",
      ]],
    ];
  }

  Games.add({
    scenario: "dungeon",
    title: "Dungeon",
    tiers: Object.keys(CARDS),
    tierLabels: TIER_LABEL,
    inspectNote: "Each decision below was one row in that tick's batched forward pass (two rows with order averaging). " +
      "The prompt is rebuilt from the recorded parts in the engine's layout; the exact chat template is under Prompt template.",
    exampleFrom: "move head",
    prepare, drawStage, pane, drawPane, verdict, facts, notes,
  });
})();
