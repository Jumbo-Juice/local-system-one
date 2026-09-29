// What every game viewer shares: the registry of game viewers, reading a trace.jsonl into a run,
// and turning playback time into a tick.
"use strict";

// One entry per game demo, keyed by the trace header's "scenario" (the same name as its runs/<game>/
// pool). A game viewer is one file in viewer/games/ that calls Games.add({...}); viewer/README.md
// lists what it must provide.
const Games = {
  byScenario: {},
  add(game) { this.byScenario[game.scenario] = game; },
  get(scenario) { return this.byScenario[scenario] || null; },
};

const Run = (() => {
  const MIN_TICK_S = 0.05;  // ticks shorter than this (the mock backend) are shown for 50 ms

  // trace.jsonl text -> a run ready to draw: the records plus playback times, latency figures and,
  // per decision tier, the last tick at or before each tick that decided that tier.
  function parse(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim()).map((l, k) => {
      try { return JSON.parse(l); } catch { throw new Error(`line ${k + 1} is not JSON; is this a trace.jsonl?`); }
    });
    const header = lines.find((l) => l.type === "header");
    const ticks = lines.filter((l) => l.type === "tick");
    const end = lines.find((l) => l.type === "end") || null;
    if (!header || !ticks.length) throw new Error("this file has no header or no ticks; is it a trace.jsonl?");
    const game = Games.get(header.scenario);
    if (!game) throw new Error(`there is no viewer for the "${header.scenario}" demo yet (add viewer/games/${header.scenario}.js)`);
    const n = ticks.length, start = new Float64Array(n + 1), dur = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const next = i + 1 < n ? ticks[i + 1].t : ticks[i].t + ticks[i].batch.tick_ms / 1000;
      dur[i] = Math.max(MIN_TICK_S, next - ticks[i].t);
      start[i + 1] = start[i] + dur[i];
    }
    const last = {};
    for (const tier of game.tiers) {
      const arr = new Int32Array(n).fill(-1);
      let k = -1;
      for (let i = 0; i < n; i++) { if (ticks[i].decisions.some((d) => d.tier === tier)) k = i; arr[i] = k; }
      last[tier] = arr;
    }
    const fw = ticks.filter((t) => t.batch.decisions).map((t) => t.batch.forward_ms).sort((a, b) => a - b);
    const q = (p) => fw.length ? fw[Math.min(fw.length - 1, Math.floor(p * fw.length))] : 0;
    const run = { game, header, ticks, end, n, start, dur, total: start[n], last, map: header.map,
      fwMedian: q(0.5), fwP90: q(0.9), fwMax: fw.length ? fw[fw.length - 1] : 0 };
    game.prepare(run);
    return run;
  }

  // Playback time -> [tick index, fraction of that tick played].
  function locate(run, clock) {
    const s = run.start;
    let lo = 0, hi = run.n - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (s[mid] <= clock) lo = mid; else hi = mid - 1; }
    return [lo, Math.min(1, Math.max(0, (clock - s[lo]) / run.dur[lo]))];
  }

  // Everything a frame needs about the moment `clock`: the tick, how far into it, and the world
  // before and after it (the move lands at 85% of the tick).
  function moment(run, clock) {
    const [i, f] = locate(run, clock);
    const tick = run.ticks[i], next = run.ticks[i + 1] || null;
    const after = next ? next.world : (run.end ? run.end.world : tick.world);
    return { i, f, tick, after, world: f < 0.85 ? tick.world : after, p: i + f };
  }

  // The decision of `tier` in force at tick index i (made at i or earlier), or null.
  function held(run, tier, i) {
    const k = run.last[tier] ? run.last[tier][i] : -1;
    if (k < 0) return null;
    const ds = run.ticks[k].decisions.filter((d) => d.tier === tier);
    return ds[ds.length - 1];
  }

  const fmtTime = (s) => { const m = Math.floor(s / 60); return String(m).padStart(2, "0") + ":" + (s - 60 * m).toFixed(1).padStart(4, "0"); };
  const modelName = (run) => String((run.header.backend || {}).model || (run.header.backend || {}).kind || "model").split("/").pop();

  return { parse, locate, moment, held, fmtTime, modelName };
})();
