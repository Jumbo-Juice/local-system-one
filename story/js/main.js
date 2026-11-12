// Wiring: build the figures, bind the numbers, and let the scroll drive them.
// A figure's picture is a function of the step in view, so forward, backward and a reload land the same.
(function () {
  "use strict";
  const S = window.S;
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  $$("[data-build]").forEach((slot) => S.build[slot.dataset.build](slot));
  S.bind(document);
  S.grids();

  // ---- the stamps: one mark per trial, in the trial, the bar and the tally ----
  const stampOf = {};
  $$(".stamp").forEach((el) => {
    stampOf[el.dataset.trial] = el.dataset.stamp;
    el.querySelector(".stamp-mark").innerHTML = S.mark(el.dataset.stamp);
  });
  $$(".tally-row").forEach((row) => (row.querySelector(".tally-mark").innerHTML = S.mark(row.dataset.stamp)));
  const slots = {};
  $$(".slot").forEach((a) => {
    const n = a.dataset.slot;
    slots[n] = a;
    a.dataset.stamp = stampOf[n];
    a.insertAdjacentHTML("beforeend", S.mark(stampOf[n]));
    a.setAttribute("aria-label", `Trial ${n}: ${a.dataset.q}`);
  });
  const setSlot = (n, on) => {
    const a = slots[n];
    a.classList.toggle("is-stamped", on);
    a.setAttribute("aria-label", `Trial ${n}: ${a.dataset.q}` + (on ? ` Stamped ${S.STAMP_WORD[stampOf[n]]}.` : ""));
  };

  const growRows = (root) =>
    S.from($$(".row-fill", root), { scaleX: 0 },
      { scaleX: (i, el) => parseFloat(getComputedStyle(el).getPropertyValue("--v")) || 0, clearProps: "transform" }, 0.8, 0.05);

  // ---- a step arrives: show its panel, then let the panel do its thing ----
  const show = (fig, step, animate) => {
    const name = step.dataset.panel;
    let changed = false, panel = null;
    $$(".panel", fig).forEach((p) => {
      const on = p.dataset.panel === name;
      if (on) { panel = p; changed = !p.classList.contains("is-on"); }
      p.classList.toggle("is-on", on);
    });
    const hook = S.hooks[name];
    if (hook && hook.enter && (animate || !changed)) hook.enter(panel, step);
    else if (hook && hook.enter) hook.enter(panel, step);
    if (changed && animate) growRows(panel);
    return panel;
  };

  if (!S.hasGsap) {
    // Without the motion library the page is still the whole story: every panel, stacked.
    document.documentElement.classList.remove("js");
    $$(".slot").forEach((a) => setSlot(a.dataset.slot, true));
    return;
  }

  const gsap = window.gsap, ScrollTrigger = window.ScrollTrigger;
  gsap.registerPlugin(ScrollTrigger);
  const narrow = window.matchMedia("(max-width: 59.99rem)");
  const line = () => (narrow.matches ? "78%" : "62%");

  $$(".scrolly").forEach((scrolly) => {
    const fig = scrolly.querySelector(".fig"), steps = $$(".step", scrolly);
    const activate = (step, animate) => {
      steps.forEach((s) => s.classList.toggle("is-on", s === step));
      return show(fig, step, animate);
    };
    activate(steps[0], false);
    steps.forEach((step) => {
      const scrub = step.dataset.scrub != null;
      ScrollTrigger.create({
        trigger: step,
        start: () => `top ${line()}`,
        end: () => `bottom ${line()}`,
        invalidateOnRefresh: true,
        // onEnter and onEnterBack also fire for a step a jump skips over, in order,
        // so a link from the bar lands on the same panel as scrolling there would.
        onEnter: () => activate(step, true),
        onEnterBack: () => activate(step, true),
        onUpdate: scrub
          ? (self) => { const hook = S.hooks[step.dataset.panel]; if (hook && hook.scrub) hook.scrub(null, self.progress, step); }
          : undefined,
      });
    });
  });

  // ---- the stamp lands, and takes its place in the bar ----
  $$(".stamp").forEach((el) => {
    const n = el.dataset.trial, mark = el.querySelector(".stamp-mark");
    ScrollTrigger.create({
      trigger: el,
      start: "top 82%",
      onEnter: () => {
        setSlot(n, true);
        S.from(mark, { scale: 1.9, rotate: -14, autoAlpha: 0 }, { scale: 1, rotate: 0, autoAlpha: 1 }, 0.7);
        S.from(mark.querySelector("path"), { strokeDasharray: 60, strokeDashoffset: 60 }, { strokeDashoffset: 0, delay: 0.18 }, 0.6);
        S.from(el.querySelector(".stamp-line"), { x: -18, autoAlpha: 0 }, { x: 0, autoAlpha: 1, delay: 0.08 }, 0.7);
        S.from(slots[n], { scale: 1.7 }, { scale: 1, delay: 0.25, clearProps: "transform" }, 0.6);
      },
      onLeaveBack: () => setSlot(n, false),
    });
  });

  // ---- how far through the story you are ----
  const progress = document.querySelector(".bar-progress i");
  ScrollTrigger.create({ start: 0, end: "max", onUpdate: (self) => (progress.style.transform = `scaleX(${self.progress.toFixed(4)})`) });

  // ---- the opening figure draws once ----
  growRows(document.getElementById("fig-look"));

  // On a narrow screen a long step's text sticks just under the figure, so it needs the figure's height.
  const measure = () => $$(".scrolly").forEach((sc) => sc.style.setProperty("--fig-h", sc.querySelector(".fig-wrap").offsetHeight + "px"));
  const refresh = () => { measure(); ScrollTrigger.refresh(); };
  measure();
  window.addEventListener("resize", measure);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(refresh);
  $$("details.where").forEach((d) => d.addEventListener("toggle", refresh));
  narrow.addEventListener("change", refresh);
})();
