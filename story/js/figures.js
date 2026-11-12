// The figures the page can't write as plain markup: each builder fills a [data-build] slot from data.js.
// S.hooks[panel] holds what a panel does when its step arrives ({ enter, scrub }).
(function () {
  "use strict";
  const S = window.S, D = S.D, esc = S.esc;
  const LETTERS = "ABCDEFGH";
  S.build = {};
  S.hooks = {};

  // ---- a question and its lettered options, with one bar per score ----
  function optionsList(options, scores, opts = {}) {
    const top = scores.indexOf(Math.max(...scores));
    return `<ol class="opts">${options.map((text, i) => {
      const danger = /BULLET|ENEMY/.test(text);
      return `<li class="opt${i === top ? " is-top" : ""}${danger ? " is-danger" : ""}">
        <span class="opt-letter">${(opts.letters || LETTERS)[i]}</span><span class="opt-text">${esc(text)}</span>
        <span class="opt-score">${scores[i].toFixed(2)}</span>
        <span class="row-track"><i class="row-fill" style="--v:${scores[i].toFixed(4)}"></i></span></li>`;
    }).join("")}</ol>`;
  }

  S.build.look = (slot) => {
    const d = D.games.look;
    slot.innerHTML = `<p class="ask">${esc(d.question)}</p>${optionsList(d.options, d.scores)}`;
  };

  // ---- Trial 2: long lists ----
  S.build.tournament = (slot) => {
    const rows = D.engine.tournament, W = 360, H = 230;
    const x = (n) => 44 + ((n - 10) / 70) * 290, y = (v) => 190 - v * 165;
    const line = (key) => rows.map((r, i) => `${i ? "L" : "M"}${x(r.options).toFixed(1)} ${y(r[key]).toFixed(1)}`).join(" ");
    const dots = (key, cls) => rows.map((r) => `<circle class="${cls}" cx="${x(r.options).toFixed(1)}" cy="${y(r[key]).toFixed(1)}" r="3.5"/>`).join("");
    const last = rows[rows.length - 1];
    slot.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Asked all at once, right answers fall from ${S.fmt.pct(rows[0].at_once)} with 10 options to ${S.fmt.pct(last.at_once)} with 80. In a tournament they stay at ${S.fmt.pct(last.tournament)} or above.">
      ${[0, 0.5, 1].map((v) => `<line class="grid" x1="44" x2="340" y1="${y(v)}" y2="${y(v)}"/><text x="38" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("")}
      ${[10, 26, 40, 80].map((n) => `<text x="${x(n)}" y="208" text-anchor="middle">${n}</text>`).join("")}
      <text x="192" y="226" text-anchor="middle">options in the list</text>
      <path class="ln ln-ink" pathLength="1" d="${line("at_once")}"/><path class="ln ln-alt" d="${line("tournament")}"/>
      ${dots("at_once", "pt-ink")}${dots("tournament", "pt-alt")}
      <text class="lab-ink" x="${x(80)}" y="${y(last.tournament) + 17}" text-anchor="end">tournament: ${S.fmt.pct(last.tournament)}</text>
      <text class="lab-ink" x="${x(80)}" y="${y(last.at_once) + 17}" text-anchor="end">all at once: ${S.fmt.pct(last.at_once)}</text>
    </svg>`;
  };
  S.hooks.lists = {
    enter(panel) {
      S.from(panel.querySelectorAll(".ln-ink"), { strokeDasharray: 1, strokeDashoffset: 1 }, { strokeDashoffset: 0 }, 1.1);
      S.from(panel.querySelectorAll(".ln-alt, .pt-alt"), { autoAlpha: 0 }, { autoAlpha: 1, delay: 0.5 }, 0.8);
    },
  };

  // ---- Trial 2: flip the order (a thing to try) ----
  S.build.flip = (slot) => {
    const d = D.games.flip, n = d.options.length;
    const views = {
      listed: { scores: d.listed, order: d.options.map((_, i) => i), note: "“Hold fire” is listed last." },
      reversed: { scores: d.reversed, order: d.options.map((_, i) => n - 1 - i), note: "The same options, in the opposite order." },
      averaged: { scores: d.averaged, order: d.options.map((_, i) => i), note: "Both looks, averaged. This is what the run acted on." },
    };
    slot.innerHTML = `<p class="ask">${esc(d.question)}</p><p class="ask-state" data-note></p><div data-opts></div>
      <div class="try"><fieldset class="seg"><legend>Try it: change the order of the options</legend>
        <label><input type="radio" name="flip" value="listed" checked><span>Hold fire last</span></label>
        <label><input type="radio" name="flip" value="reversed"><span>Hold fire first</span></label>
        <label><input type="radio" name="flip" value="averaged"><span>Both, averaged</span></label>
      </fieldset></div>`;
    const set = (name, animate) => {
      const v = views[name];
      slot.querySelector("[data-note]").textContent = v.note;
      slot.querySelector("[data-opts]").innerHTML = optionsList(v.order.map((i) => d.options[i]), v.order.map((i) => v.scores[i]));
      slot.querySelector(`input[value="${name}"]`).checked = true;
      if (animate) S.from(slot.querySelectorAll(".row-fill"), { scaleX: 0 }, { scaleX: (i, el) => Number(el.style.getPropertyValue("--v")) , clearProps: "transform" }, 0.6);
    };
    slot.addEventListener("change", (e) => e.target.name === "flip" && set(e.target.value, true));
    set("listed", false);
    S.hooks.flip = { enter: () => set("listed", false) };
  };

  // ---- Trial 4: one recorded payment decision ----
  S.build.payment = (slot) => {
    const d = D.fraud.paysim.decision;
    const order = ["approve", "review", "decline"];
    const kind = d.type === "TRANSFER" ? "A transfer" : "A cash-out";
    slot.innerHTML = `<p class="fig-title">One look at one simulated payment</p>
      <p class="ask">${kind} of ${S.n(d.amount)} to another customer, at hour ${d.hour}. Which action?</p>
      ${optionsList(order, order.map((k) => d.scores[k]))}
      <p class="cap">The model leans to ${esc(d.action)}. The payment was ${d.truth ? "a fraud" : "genuine"}. Amounts in PaySim are in its own simulated currency.</p>`;
  };

  // ---- Trials 4 and 5: one square per decision ----
  const GRIDS = {
    declined: () => {
      const m = D.manual.paysim_frozen;
      return { n: m.declined, special: m.frauds, base: "#f87171", mark: "#7f1d1d", cols: 76 };
    },
    match: () => {
      const p = D.fraud.paysim;
      return { n: p.decisions, special: p.differ, base: "#cbd5e1", mark: "#b45309", cols: 76 };
    },
  };
  function drawGrid(canvas, spec, shown) {
    const rows = Math.ceil(spec.n / spec.cols);
    const cssW = canvas.clientWidth || 600, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cell = (cssW * dpr) / spec.cols;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cell * rows);
    canvas.style.aspectRatio = `${spec.cols} / ${rows}`;
    const ctx = canvas.getContext("2d");
    const gap = Math.max(1, cell * 0.16), special = [];
    ctx.fillStyle = spec.base;
    for (let i = 0; i < Math.min(shown, spec.n); i++) {
      // the marked squares are spread evenly: the same picture every time
      if (Math.floor(((i + 1) * spec.special) / spec.n) > Math.floor((i * spec.special) / spec.n)) { special.push(i); continue; }
      ctx.fillRect((i % spec.cols) * cell, Math.floor(i / spec.cols) * cell, cell - gap, cell - gap);
    }
    ctx.fillStyle = spec.mark;
    const grow = spec.special < 50 ? cell * 0.9 : 0; // a handful of squares must still be findable
    special.forEach((i) => ctx.fillRect((i % spec.cols) * cell - grow / 2, Math.floor(i / spec.cols) * cell - grow / 2, cell - gap + grow, cell - gap + grow));
  }
  S.grids = () => {
    document.querySelectorAll("[data-grid]").forEach((canvas) => {
      const spec = GRIDS[canvas.dataset.grid]();
      const draw = (shown) => drawGrid(canvas, spec, shown);
      draw(spec.n);
      canvas._draw = draw;
      S.hooks[canvas.closest(".panel").dataset.panel] = {
        enter() {
          if (!S.hasGsap || S.reduced) return draw(spec.n);
          const state = { shown: 0 };
          window.gsap.to(state, { shown: spec.n, duration: 1.2, ease: "expo.out", overwrite: true, onUpdate: () => draw(Math.round(state.shown)) });
        },
      };
    });
    let timer;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => document.querySelectorAll("[data-grid]").forEach((c) => c._draw(Infinity)), 150); });
  };

  // ---- Trial 5: the rule that weighs the amount (a thing to try) ----
  S.build.rule = (slot) => {
    const c = D.fraud.costs, lambda = D.fraud.lambda;
    slot.innerHTML = `<p class="fig-title">What each action is expected to cost</p>
      <div class="slider"><label for="rule-amount">The payment’s amount</label><output for="rule-amount" data-out="amount"></output>
        <input id="rule-amount" type="range" min="5" max="1500" step="5" value="60"></div>
      <div class="slider"><label for="rule-p">Fraud estimate for this payment</label><output for="rule-p" data-out="p"></output>
        <input id="rule-p" type="range" min="0" max="60" step="1" value="8">
        <small>How many payments like this one, in 100, turned out to be fraud. A fit made on earlier payments, used only to weigh costs.</small></div>
      <div class="rows">
        ${["approve", "review", "decline"].map((k) => `<div class="row" data-action="${k}"><span class="row-label">${k[0].toUpperCase() + k.slice(1)}</span><span class="row-val"></span><span class="row-track"><i class="row-fill"></i></span></div>`).join("")}
      </div>
      <p class="pick" aria-live="polite">Cheapest: <b data-out="pick"></b></p>`;
    const amount = slot.querySelector("#rule-amount"), p = slot.querySelector("#rule-p");
    const out = (k) => slot.querySelector(`[data-out="${k}"]`);
    const update = () => {
      const a = Number(amount.value), share = Number(p.value) / 100;
      const cost = {
        approve: share * a,
        review: c.review_fee + lambda,
        decline: (1 - share) * Math.max(c.friction_rate * a, c.friction_min),
      };
      const max = Math.max(...Object.values(cost), 20);
      const best = Object.keys(cost).reduce((x, y) => (cost[y] < cost[x] ? y : x));
      [amount, p].forEach((el) => el.style.setProperty("--p", (((el.value - el.min) / (el.max - el.min)) * 100).toFixed(1) + "%"));
      out("amount").textContent = S.n(a) + " USD";
      out("p").textContent = p.value + " in 100";
      out("pick").textContent = best;
      slot.querySelectorAll("[data-action]").forEach((row) => {
        const v = cost[row.dataset.action];
        row.classList.toggle("is-model", row.dataset.action === best);
        row.querySelector(".row-val").textContent = v.toFixed(2) + " USD";
        row.style.setProperty("--v", (v / max).toFixed(4));
      });
    };
    amount.addEventListener("input", update);
    p.addEventListener("input", update);
    update();
  };

  // ---- Trial 6: the final test, window by window ----
  S.build.race = (slot) => {
    const s = D.fraud.final.setups;
    const lines = [
      { key: "approve_all", name: "Approve everything", cls: "ln-quiet", lab: "lab-quiet" },
      { key: "rich_amt", name: "The model", cls: "ln-model", lab: "lab-model" },
      { key: "baseline", name: "Logistic regression", cls: "ln-ink", lab: "lab-ink" },
    ];
    const cum = (arr) => arr.reduce((acc, v) => (acc.push((acc[acc.length - 1] || 0) + v), acc), []);
    const series = lines.map((l) => ({ ...l, pts: [0, ...cum(s[l.key].windows)] }));
    const n = series[0].pts.length - 1, end = (l) => l.pts[n], top = Math.max(...series.map(end));
    const W = 360, H = 232, x = (i) => 40 + (i / n) * 212, y = (v) => 196 - (v / top) * 178;
    const ticks = [0, 50000, 100000].filter((v) => v <= top * 1.05);
    // each line is named where it ends; of the two that end close together, the dearer is labelled above its end
    const close = series.filter((l) => l.key !== "approve_all").sort((a, b) => end(b) - end(a));
    const dy = (l) => (l === close[0] ? -17 : l === close[1] ? 13 : 4);
    slot.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Running cost over the 20 windows. ${series.map((l) => `${l.name} ends at ${S.n(end(l))} USD`).join("; ")}.">
        <defs><clipPath id="race-clip"><rect x="0" y="0" width="${W}" height="${H}"/></clipPath></defs>
        ${ticks.map((v) => `<line class="grid" x1="40" x2="252" y1="${y(v)}" y2="${y(v)}"/><text x="34" y="${y(v) + 4}" text-anchor="end">${v ? v / 1000 + "k" : "0"}</text>`).join("")}
        ${[1, 5, 10, 15, 20].map((i) => `<text x="${x(i)}" y="214" text-anchor="middle">${i}</text>`).join("")}
        <text x="146" y="230" text-anchor="middle">window</text>
        <g clip-path="url(#race-clip)">${series.map((l) => `<path class="ln ${l.cls}" d="${l.pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")}"/>`).join("")}</g>
        <g data-ends>${series.map((l) => `<text class="${l.lab}" x="${x(n) + 8}" y="${(y(end(l)) + dy(l)).toFixed(1)}">${l.name}</text><text class="${l.lab}" x="${x(n) + 8}" y="${(y(end(l)) + dy(l) + 12).toFixed(1)}">${S.n(end(l))} USD</text>`).join("")}</g>
      </svg>
      <p class="cap">The dashed line approves every payment. The total at the end of each line is the cost over all 20 windows.</p>`;
    const rect = slot.querySelector("#race-clip rect"), ends = slot.querySelector("[data-ends]");
    const reveal = (p) => {
      rect.setAttribute("width", (40 + p * 214).toFixed(1));
      ends.style.opacity = p > 0.97 ? 1 : 0.25;
    };
    reveal(1);
    S.hooks.race = { scrub: (panel, p) => reveal(p) };
  };

  S.build.forest = (slot) => {
    const diffs = D.fraud.final.diffs, LO = -4200, HI = 1400, at = (v) => ((v - LO) / (HI - LO)).toFixed(4);
    const row = (key, id, label, colour) => {
      const d = diffs[key];
      return `<div class="forest-row" data-row="${id}" style="--c:${colour};--zero:${at(0)};--lo:${at(d.ci[0])};--hi:${at(d.ci[1])};--at:${at(d.mean)}">
        <p class="forest-label">${label}</p>
        <div class="forest-track" role="img" aria-label="Difference ${S.n(d.mean)} USD per window, interval from ${S.n(d.ci[0])} to ${S.n(d.ci[1])}">
          <span class="forest-zero"></span><span class="forest-ci"></span><span class="forest-dot"></span></div>
        <div class="forest-ends"><span>the model costs less</span><span>costs more</span></div>
        <p class="forest-said">${S.n(Math.abs(d.mean))} USD ${d.mean < 0 ? "less" : "more"} per window. The interval ${d.distinguishable ? "leaves out zero: distinguishable." : "crosses zero: not distinguishable."}</p>
      </div>`;
    };
    slot.innerHTML = `<p class="fig-title">The model’s cost minus the other’s, with its 95% interval</p>
      <div class="forest" data-state="a">${row("rich_amt_approve", "a", "Against approving everything", "#2563eb")}${row("rich_amt_baseline", "b", "Against logistic regression", "#2563eb")}</div>`;
    S.hooks.forest = { enter: (panel, step) => panel.querySelector(".forest").setAttribute("data-state", step.dataset.state || "b") };
  };
})();
