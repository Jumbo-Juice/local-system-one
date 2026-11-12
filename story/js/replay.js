// A recorded game run, drawn tick by tick. The scroll moves time; nothing here plays the game.
(function () {
  "use strict";
  const S = window.S, D = S.D, esc = S.esc;

  const TEXT = {
    bullet: {
      title: "The larger model in the shooter, as recorded",
      label: "A map of nine rooms. The blue dot is the agent; red shapes are enemies and their bullets.",
      key: [["round", "#2563eb", "the agent"], ["round", "#b91c1c", "gunner"], ["", "#b91c1c", "brute"], ["round hollow", "#b91c1c", "asleep"], ["", "#b45309", "key"], ["", "#15803d", "exit"]],
    },
    escape: {
      title: "The larger model in the dungeon, as recorded",
      label: "A map of nine rooms. The blue dot is the agent; red circles are ghouls.",
      key: [["round", "#2563eb", "the agent"], ["round", "#b91c1c", "ghoul, awake"], ["round hollow", "#b91c1c", "ghoul, asleep"], ["", "#b45309", "key"], ["", "#15803d", "exit"]],
    },
  };

  S.build.replay = (slot) => {
    const name = slot.dataset.run, run = D.replays[name], map = run.map, frames = run.frames, text = TEXT[name];
    const cell = ([x, y]) => `<rect class="floor" x="${x}" y="${y}" width="1" height="1"/>`;
    slot.innerHTML = `<p class="fig-title">${text.title}</p>
      <svg class="replay-map" viewBox="0 0 ${map.w} ${map.h}" role="img" aria-label="${text.label}">
        ${map.rooms.map(([x, y, w, h]) => `<rect class="floor" x="${x}" y="${y}" width="${w}" height="${h}"/>`).join("")}
        ${map.halls.map(cell).join("")}${map.doors.map(cell).join("")}
        <rect class="exit" x="${map.exit[0] + 0.1}" y="${map.exit[1] + 0.1}" width="0.8" height="0.8" rx="0.12"/>
        <polyline class="trail" data-trail/>
        <g data-things></g>
        <circle class="warn" r="0.95" data-warn/>
        <circle class="you" r="0.48" data-you/>
      </svg>
      <div class="ruler" aria-hidden="true">${(run.warned_ticks || []).map((w) => `<b style="--at:${(w / (frames.length - 1)).toFixed(4)}"></b>`).join("")}<i></i></div>
      <div class="replay-hud"><span>Tick <b data-t></b> of ${run.end_tick}</span><span>Health <b data-hp></b><span class="hp"><i></i></span></span><span data-carry></span></div>
      <div class="replay-card"><p class="chose" data-chose></p><p class="others" data-others></p></div>
      <ul class="key">${text.key.map(([cls, c, label]) => `<li><i class="${cls}" style="--c:${c}"></i>${label}</li>`).join("")}</ul>`;
    const q = (sel) => slot.querySelector(sel);
    const you = q("[data-you]"), warn = q("[data-warn]"), trail = q("[data-trail]"), things = q("[data-things]"), hp = q(".hp"), ruler = q(".ruler");
    const warned = new Set(run.warned_ticks || []);
    const token = (v) => (v === "moment" ? run.moment.tick : v === "end" ? frames.length - 1 : Number(v));
    let shown = -1;

    const setTick = (i) => {
      i = Math.max(0, Math.min(frames.length - 1, i));
      if (i === shown) return;
      shown = i;
      const f = frames[i], last = i === frames.length - 1;
      const health = last && f.hp_after != null ? f.hp_after : f.hp;
      you.setAttribute("cx", f.a[0] + 0.5);
      you.setAttribute("cy", f.a[1] + 0.5);
      you.classList.toggle("hit", f.ev.includes("hit"));
      warn.setAttribute("cx", f.a[0] + 0.5);
      warn.setAttribute("cy", f.a[1] + 0.5);
      warn.classList.toggle("on", warned.has(f.t));
      ruler.style.setProperty("--now", (i / (frames.length - 1)).toFixed(4));
      trail.setAttribute("points", frames.slice(0, i + 1).map((g) => `${g.a[0] + 0.5},${g.a[1] + 0.5}`).join(" "));
      things.innerHTML =
        (f.k ? `<path class="keyitem" d="M${f.k[0] + 0.5} ${f.k[1] + 0.1}l0.4 0.4-0.4 0.4-0.4-0.4z"/>` : "") +
        f.e.map(([x, y, kind, , awake]) => kind === "b"
          ? `<rect class="foe${awake ? "" : " asleep"}" x="${x + 0.14}" y="${y + 0.14}" width="0.72" height="0.72"/>`
          : `<circle class="foe${awake ? "" : " asleep"}" cx="${x + 0.5}" cy="${y + 0.5}" r="0.38"/>`).join("") +
        (f.b || []).map(([x, y, owner]) => `<circle class="shot${owner === "e" ? "" : " own"}" cx="${x + 0.5}" cy="${y + 0.5}" r="0.2"/>`).join("");
      q("[data-t]").textContent = last ? run.end_tick : f.t;
      q("[data-hp]").textContent = health;
      hp.style.setProperty("--v", (health / 100).toFixed(2));
      hp.classList.toggle("low", health <= 40);
      q("[data-carry]").textContent = f.key ? "Carrying the key" : "";

      const chose = q("[data-chose]"), others = q("[data-others]"), m = f.m;
      if (last) {
        chose.className = "chose replay-end";
        chose.textContent = run.outcome === "died" ? `Killed by a ${run.cause} on tick ${run.end_tick}, after ${run.hits} hits.` : `Escaped on tick ${run.end_tick}, with ${run.hits === 0 ? "no hits" : run.hits + " hits"}.`;
        others.textContent = "";
      } else if (m) {
        const pick = m.o[m.c], danger = /BULLET|ENEMY/.test(pick), safe = m.o.filter((o, j) => j !== m.c && /safe/.test(o));
        chose.className = "chose" + (danger ? " is-danger" : "");
        chose.innerHTML = `It chose: <b>${esc(pick)}</b>, score ${m.p[m.c].toFixed(2)}`;
        others.textContent = m.o.length > 1 ? (danger && safe.length ? "A safe move was on the list: " + (safe.find((o) => /on the target|closer/.test(o)) || safe[0]) : "Also on the list: " + m.o.filter((_, j) => j !== m.c).slice(0, 2).join("; ")) : "It was the only move on the list.";
      } else {
        chose.className = "chose";
        chose.textContent = "No move question this tick.";
        others.textContent = "";
      }
    };

    setTick(0);
    S.hooks[slot.closest(".panel").dataset.panel] = {
      enter: (panel, step) => { if (step.dataset.tick != null) setTick(Number(step.dataset.tick)); },
      scrub: (panel, p, step) => {
        const [a, b] = step.dataset.scrub.split(":").map(token);
        setTick(Math.round(a + (b - a) * p));
      },
    };
  };
})();
