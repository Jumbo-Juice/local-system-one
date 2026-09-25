"""Real-time 2D demo: every tick, all agents' due decisions go through the engine in ONE batch.

    python -m demo.sim                                   # window, model from config/default.toml
    python -m demo.sim --config config/mock.toml         # window, no model (random decisions)
    python -m demo.sim --headless --ticks 100            # text only; prints a summary
    python -m demo.sim --headless --ticks 100 --no-goals # flat control: no strategy/target tiers

Window keys: space = pause/resume, Esc = quit. Visuals are deliberately minimal.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time

from system_one import Decision, load_config, make_engine

from .brain import Controller
from .world import MOVES, World

CELL = 34
PANEL_W = 430


def build_controller(args) -> Controller:
    engine = make_engine(load_config(args.config))
    engine.decide_batch([Decision("warm-up", ("a", "b"))])  # first XPU pass compiles kernels (~3 s)
    world = World(n_agents=args.agents, n_gems=args.gems, n_food=args.food, seed=args.seed)
    return Controller(world, engine, use_goals=not args.no_goals, group_size=args.group_size)


def agent_view(ctrl: Controller, display: dict) -> list[dict]:
    out = []
    for b in ctrl.brains:
        a = b.agent
        tours = {name: t for name, t in b.stack.tournaments.items()}
        out.append({
            "id": a.id, "pos": a.pos, "colour": a.colour, "score": a.score, "energy": a.energy,
            "health": a.health, "deaths": a.deaths, "target_cell": b.target_cell(),
            "current": dict(b.stack.current), "prob": dict(b.stack.probability),
            "tournaments": {n: (len(t.rounds) + 1, len(t.alive), len(t.options)) for n, t in tours.items()},
            "last": display.get(a.id, {}),
        })
    return out


def remember(display: dict, report) -> None:
    """Keep the latest result per agent and tier for the side panel."""
    for agent_id, tier, r, tour in report.updates:
        entry = {"choice": r.choice, "probs": dict(zip(r.decision.options, r.probs)),
                 "outside": r.outside_mass, "method": r.method, "in_tournament": tour is not None}
        display.setdefault(agent_id, {})[tier] = entry


def run_headless(ctrl: Controller, ticks: int, verbose: bool = True) -> dict:
    reports = []
    for _ in range(ticks):
        rep = ctrl.tick()
        reports.append(rep)
        if verbose:
            acts = " ".join(f"{aid}:{r.choice.replace('move ', '')}" for aid, tier, r, _ in rep.updates if tier == "action")
            print(f"tick {rep.tick:4} decisions {rep.decisions:3} forward {1000 * rep.forward_s:7.1f} ms  {acts}", flush=True)
    fw = [1000 * r.forward_s for r in reports if r.decisions]
    agents = ctrl.world.agents
    summary = {
        "ticks": ticks, "agents": len(agents), "use_goals": ctrl.brains[0].use_goals,
        "gems": sum(a.score for a in agents), "deaths": sum(a.deaths for a in agents),
        "hazard_hits": sum(r.events["hits"] for r in reports), "food_eaten": sum(r.events["food"] for r in reports),
        "decisions_per_tick_mean": statistics.mean(r.decisions for r in reports),
        "forward_ms_median": statistics.median(fw) if fw else 0.0,
        "forward_ms_p90": statistics.quantiles(fw, n=10)[-1] if len(fw) >= 10 else max(fw, default=0.0),
        "forward_ms_max": max(fw, default=0.0),
        "tick_ms_median": 1000 * statistics.median(r.total_s for r in reports),
        "tournament_decisions": sum(1 for r in reports for u in r.updates if u[3] is not None),
        "action_counts": {m: sum(1 for r in reports for u in r.updates if u[1] == "action" and u[2].choice == m)
                          for m in MOVES},
    }
    return summary


class App:
    """Minimal tkinter view. The engine runs in a worker thread; the UI polls snapshots."""

    def __init__(self, ctrl: Controller, title: str, close_after: float = 0.0):
        import tkinter as tk

        self.tk, self.ctrl = tk, ctrl
        w = ctrl.world
        self.root = tk.Tk()
        self.root.title(title)
        # Status lines go below both the grid and the agent panel (~125 px per agent).
        self.footer_y = max(w.height * CELL, 8 + 125 * len(w.agents)) + 8
        self.canvas = tk.Canvas(self.root, width=w.width * CELL + PANEL_W, height=self.footer_y + 40,
                                bg="white", highlightthickness=0)
        self.canvas.pack()
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.running = threading.Event()
        self.running.set()
        self.stop = threading.Event()
        self.display: dict = {}
        self.rates: list[float] = []
        self.root.bind("<space>", lambda e: self.running.clear() if self.running.is_set() else self.running.set())
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self.work, daemon=True).start()
        self.root.after(30, self.poll)
        if close_after:
            self.root.after(int(1000 * close_after), self.close)

    def close(self):
        self.stop.set()
        self.root.destroy()

    def work(self):
        while not self.stop.is_set():
            if not self.running.is_set():
                time.sleep(0.05)
                continue
            rep = self.ctrl.tick()
            remember(self.display, rep)
            snap = {"report": rep, "agents": agent_view(self.ctrl, self.display),
                    "gems": list(self.ctrl.world.gems), "food": list(self.ctrl.world.food),
                    "hazards": list(self.ctrl.world.hazards)}
            self.queue.put(snap)  # blocks if the UI is behind: the sim never outruns the view

    def poll(self):
        try:
            snap = None
            while True:
                snap = self.queue.get_nowait()
        except queue.Empty:
            pass
        if snap is not None:
            self.draw(snap)
        if not self.stop.is_set():
            self.root.after(30, self.poll)

    def draw(self, s):
        c, w = self.canvas, self.ctrl.world
        c.delete("all")
        for x in range(w.width):
            for y in range(w.height):
                fill = "#555" if (x, y) in w.walls else "#f4f4f4"
                c.create_rectangle(x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL, fill=fill, outline="#ddd")
        for x, y in s["food"]:
            c.create_oval(x * CELL + 11, y * CELL + 11, x * CELL + 23, y * CELL + 23, fill="#2e9e44", outline="")
        for x, y in s["gems"]:
            cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
            c.create_polygon(cx, cy - 8, cx + 7, cy, cx, cy + 8, cx - 7, cy, fill="#1aa3c9", outline="")
        for x, y in s["hazards"]:
            c.create_rectangle(x * CELL + 5, y * CELL + 5, (x + 1) * CELL - 5, (y + 1) * CELL - 5, fill="#d62728", outline="")
        for a in s["agents"]:
            x, y = a["pos"]
            if a["target_cell"]:
                tx, ty = a["target_cell"]
                c.create_line(x * CELL + CELL // 2, y * CELL + CELL // 2, tx * CELL + CELL // 2, ty * CELL + CELL // 2,
                              fill=a["colour"], dash=(3, 3))
            c.create_oval(x * CELL + 4, y * CELL + 4, (x + 1) * CELL - 4, (y + 1) * CELL - 4, fill=a["colour"], outline="black")
            c.create_text(x * CELL + CELL // 2, y * CELL + CELL // 2, text=str(a["id"]), fill="white", font=("Segoe UI", 10, "bold"))
        self.draw_panel(s)

    def draw_panel(self, s):
        c, w = self.canvas, self.ctrl.world
        x0, y = w.width * CELL + 12, 8
        for a in s["agents"]:
            c.create_rectangle(x0, y + 2, x0 + 12, y + 14, fill=a["colour"], outline="")
            c.create_text(x0 + 18, y, anchor="nw", font=("Segoe UI", 9, "bold"),
                          text=f"Agent {a['id']}  gems {a['score']}  energy {a['energy']}  health {a['health']}  deaths {a['deaths']}")
            y += 17
            for tier in ("strategy", "target"):
                if tier in a["tournaments"]:
                    rnd, alive, total = a["tournaments"][tier]
                    txt = f"{tier}: tournament round {rnd}, {alive} of {total} options left"
                elif tier in a["current"]:
                    txt = f"{tier} (p={a['prob'].get(tier, 0):.2f}): {a['current'][tier]}"
                else:
                    continue
                c.create_text(x0, y, anchor="nw", font=("Segoe UI", 8), text=txt[:72])
                y += 14
            act = a["last"].get("action")
            if act:
                for m in MOVES:
                    p = act["probs"].get(m, 0.0)
                    bold = m == act["choice"]
                    c.create_text(x0, y, anchor="nw", font=("Consolas", 8, "bold" if bold else "normal"), text=f"{m:11}")
                    c.create_rectangle(x0 + 80, y + 3, x0 + 80 + int(200 * p), y + 11, fill=a["colour"] if bold else "#bbb", outline="")
                    c.create_text(x0 + 285, y, anchor="nw", font=("Consolas", 8), text=f"{p:.2f}")
                    y += 12
                c.create_text(x0, y, anchor="nw", font=("Consolas", 8), fill="#666",
                              text=f"outside allowed tokens: {act['outside']:.3f}")
                y += 14
            y += 6
        rep = s["report"]
        now = time.perf_counter()
        self.rates = [t for t in self.rates if now - t < 5] + [now]
        tps = (len(self.rates) - 1) / max(1e-9, self.rates[-1] - self.rates[0]) if len(self.rates) > 1 else 0.0
        per = 1000 * rep.forward_s / rep.decisions if rep.decisions else 0.0
        backend = self.ctrl.engine.backend.info()
        c.create_text(8, self.footer_y, anchor="nw", font=("Consolas", 9),
                      text=(f"tick {rep.tick}  |  {rep.decisions} decisions in one batch  |  forward "
                            f"{1000 * rep.forward_s:.0f} ms ({per:.1f} ms/decision)  |  {tps:.1f} ticks/s"
                            + ("" if self.running.is_set() else "  |  PAUSED")
                            + f"\nbackend {backend.get('kind')} {str(backend.get('model', '')).split('/')[-1]} "
                            f"{backend.get('device', '')}  |  space: pause  esc: quit"))

    def run(self):
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--gems", type=int, default=24)
    ap.add_argument("--food", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=8, help="tournament group size for large choice sets")
    ap.add_argument("--no-goals", action="store_true", help="flat control: action tier only")
    ap.add_argument("--json", default=None, help="headless: write the summary to this file")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--close-after", type=float, default=0.0, help="window: close after N seconds (smoke tests)")
    args = ap.parse_args()
    ctrl = build_controller(args)
    if args.headless:
        summary = run_headless(ctrl, args.ticks, verbose=not args.quiet)
        summary["backend"] = ctrl.engine.backend.info()
        print(json.dumps(summary, indent=1))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=1)
    else:
        App(ctrl, "System One-style decision demo (not Jev)", args.close_after).run()


if __name__ == "__main__":
    main()
