// The story's shared pieces: data lookup, number formats, the comparison sentence, the stamp marks.
(function () {
  "use strict";
  const D = window.STORY;
  const S = (window.S = { D });

  S.get = (path) => path.split(".").reduce((node, key) => (node == null ? node : node[key]), D);

  const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  S.n = (v) => nf.format(Math.round(v));
  S.fmt = {
    int: (v) => S.n(v),
    usd: (v) => S.n(v) + " USD",
    absusd: (v) => S.n(Math.abs(v)) + " USD",
    ms: (v) => S.n(v) + " ms",
    pct: (v) => Math.round(v * 100) + "%",
    x: (v) => String(v),
    score2: (v) => v.toFixed(2),
    raw: (v) => String(v),
  };

  // A comparison as one plain sentence: the first setup minus the second, as the console and the deck word it.
  S.compare = (d, first, second, unit) => {
    const m = Math.round(d.mean);
    const bound = (v) => (Math.round(v) < 0 ? S.n(-v) + " less" : Math.round(v) > 0 ? S.n(v) + " more" : "0");
    const head = m === 0
      ? `${first} cost the same per ${unit} as ${second} to the dollar`
      : `${first} cost ${S.n(Math.abs(m))} USD ${m < 0 ? "less" : "more"} per ${unit} than ${second}`;
    return `${head}, ${d.distinguishable || m === 0 ? "and" : "but"} the 95% interval of the difference runs from ${bound(d.ci[0])} to ${bound(d.ci[1])}, so the two are ${d.distinguishable ? "" : "not "}distinguishable.`;
  };

  S.bind = (root) => {
    root.querySelectorAll("[data-bind]").forEach((el) => {
      const [path, f = "int"] = el.dataset.bind.split("|");
      el.textContent = S.fmt[f](S.get(path));
    });
    root.querySelectorAll("[data-fill]").forEach((el) => {
      const v = S.get(el.dataset.fill) / Number(el.dataset.max);
      el.style.setProperty("--v", Math.max(0, Math.min(1, v)).toFixed(4));
    });
    root.querySelectorAll("[data-compare]").forEach((el) => {
      el.textContent = S.compare(S.get(el.dataset.compare), el.dataset.first, el.dataset.second, el.dataset.unit);
    });
    root.querySelectorAll("[data-dots]").forEach((el) => {
      const on = S.get(el.dataset.dots), of = S.get(el.dataset.of);
      el.setAttribute("role", "img");
      el.setAttribute("aria-label", `${on} of ${of}`);
      el.innerHTML = Array.from({ length: of }, (_, i) => `<li${i < on ? ` class="${el.dataset.kind || "on"}"` : ""}></li>`).join("");
    });
  };

  S.h = (html) => {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  };
  S.esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const MARKS = {
    yes: "M5 12.5l4.5 4.5L19 7.5",
    no: "M6.5 6.5l11 11M17.5 6.5l-11 11",
    help: "M12 5.5v13M5.5 12h13",
    cant: "M8.6 9.2a3.4 3.4 0 1 1 5.7 2.5c-1.3 1.1-2.3 1.8-2.3 3.3M12 18.7v.1",
  };
  S.STAMP_WORD = { yes: "yes", no: "no", help: "only with help", cant: "can’t tell" };
  S.mark = (stamp) => `<svg class="mark" viewBox="0 0 24 24" aria-hidden="true"><path d="${MARKS[stamp]}"/></svg>`;

  S.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches || /[?&]reduced=1\b/.test(location.search);
  S.hasGsap = typeof window.gsap !== "undefined";
  if (S.reduced) document.documentElement.classList.add("reduced");

  // One way to move anything: the same ease everywhere, or the final state at once.
  S.to = (target, vars, duration = 0.7) => {
    if (!S.hasGsap) return;
    if (S.reduced) window.gsap.set(target, vars);
    else window.gsap.to(target, { ...vars, duration, ease: "expo.out", overwrite: "auto" });
  };
  S.from = (target, from, to, duration = 0.8, stagger = 0) => {
    if (!S.hasGsap || S.reduced) return;
    window.gsap.fromTo(target, from, { ...to, duration, ease: "expo.out", stagger, overwrite: "auto" });
  };
})();
