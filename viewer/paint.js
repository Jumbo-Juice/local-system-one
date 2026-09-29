// Canvas helpers shared by the game viewers. A game viewer calls Paint.use(ctx, palette) before it
// draws; the helpers then draw on that context with that palette.
"use strict";

const Paint = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const UI = css("--ui"), MONO = css("--mono");
  let ctx = null, K = null;

  function use(context, palette) { ctx = context; K = palette; }
  function font(size, weight = 400, family = UI) { ctx.font = `${weight} ${size}px ${family}`; }
  function spacing(px) { if ("letterSpacing" in ctx) ctx.letterSpacing = px + "px"; }
  function text(str, x, y, color, size, weight = 400, family = UI, align = "left") {
    font(size, weight, family); ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(str, x, y);
  }
  // Shrink the font down to `min`, then cut with an ellipsis, until `str` fits in maxW. Returns [text, size].
  function fit(str, maxW, size, weight = 400, family = UI, min = 13) {
    let s = size; font(s, weight, family);
    while (ctx.measureText(str).width > maxW && s > min) font(--s, weight, family);
    if (ctx.measureText(str).width > maxW) {
      while (str.length > 1 && ctx.measureText(str + "…").width > maxW) str = str.slice(0, -1);
      str += "…";
    }
    return [str, s];
  }
  function wrap(str, maxW, size, weight = 400, family = UI) {
    font(size, weight, family);
    const words = str.split(" "), lines = [];
    let line = "";
    for (const w of words) {
      const t = line ? line + " " + w : w;
      if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
    }
    if (line) lines.push(line);
    return lines;
  }
  function rrect(x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function panel(x, y, w, h, fill = K.panel) {
    rrect(x, y, w, h, 10); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = K.edge; ctx.lineWidth = 1; ctx.stroke();
  }
  function pill(str, xRight, yMid, color, bg) {
    font(14, 700); spacing(1.5);
    const w = ctx.measureText(str).width + 20;
    rrect(xRight - w, yMid - 13, w, 26, 13); ctx.fillStyle = bg; ctx.fill();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.fillText(str, xRight - w + 10, yMid + 5);
    spacing(0);
  }

  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  const alpha = (hex, a) => hex + Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");
  function mix(a, b, t) {
    const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const [x, y] = [p(a), p(b)];
    return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",") + ")";
  }

  return { UI, MONO, use, font, spacing, text, fit, wrap, rrect, panel, pill, lerp, ease, alpha, mix };
})();
