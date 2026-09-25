"""Tournament sampling vs one full decision on the same large choice sets.

Task: "Which of these is a <category>?" with exactly one member of the category hidden among
n-1 words from other categories. n <= 255, so a single full decision is also possible
(26+ options use two-letter labels). Reports accuracy, agreement and latency.

Usage: python -m bench.tournament_compare --config config/default.toml --sizes 40 80 --problems 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from system_one import Decision, load_config, make_engine
from system_one.tournament import Tournament, run_tournaments

from .hwinfo import host_info

WORDS = {
    "animal": "dog cat horse cow sheep lion tiger elephant giraffe zebra wolf fox bear rabbit deer "
              "camel monkey kangaroo otter beaver",
    "fruit": "apple banana cherry grape lemon mango orange peach pear plum kiwi melon apricot papaya "
             "lime fig guava lychee coconut pineapple",
    "vehicle": "car bus truck bicycle motorcycle train tram van taxi scooter ambulance tractor lorry "
               "minibus jeep limousine forklift bulldozer caravan trolleybus",
    "tool": "hammer screwdriver wrench pliers saw drill chisel spanner shovel rake axe crowbar "
            "sandpaper tape-measure file mallet trowel clamp level hacksaw",
    "musical instrument": "piano violin guitar drum flute trumpet cello harp clarinet saxophone oboe "
                          "banjo accordion tuba trombone ukulele harmonica xylophone bassoon mandolin",
    "colour": "red blue green yellow purple orange-colour pink brown black white grey violet indigo "
              "turquoise magenta beige maroon navy teal crimson",
    "country": "France Spain Germany Italy Japan China India Brazil Canada Mexico Kenya Egypt Norway "
               "Sweden Peru Chile Vietnam Thailand Laos Ghana",
    "sport": "football tennis cricket rugby golf hockey baseball basketball volleyball badminton "
             "swimming boxing cycling rowing fencing archery skiing surfing wrestling judo",
    "piece of furniture": "chair table sofa bed wardrobe desk bookshelf stool bench cupboard dresser "
                          "armchair ottoman cabinet sideboard futon hammock crib recliner nightstand",
    "vegetable": "carrot potato onion cabbage lettuce spinach broccoli cauliflower pea bean celery "
                 "cucumber radish turnip leek garlic beetroot courgette pumpkin asparagus",
}
WORDS = {k: v.split() for k, v in WORDS.items()}
OUT = Path(__file__).parent / "results"


def problems(n: int, count: int, seed: int) -> list[tuple[str, list[str], str]]:
    rng = random.Random(seed)
    out = []
    cats = list(WORDS)
    for _ in range(count):
        cat = rng.choice(cats)
        answer = rng.choice(WORDS[cat])
        others = [w for c in cats if c != cat for w in WORDS[c]]
        opts = rng.sample(others, n - 1) + [answer]
        rng.shuffle(opts)
        out.append((cat, opts, answer))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[40, 80])
    ap.add_argument("--groups", type=int, nargs="+", default=[10, 26])
    ap.add_argument("--problems", type=int, default=30)
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    engine.decide_batch([Decision("warm-up", ("a", "b"))])
    report = {"host": host_info(), "backend": engine.backend.info(), "args": vars(args), "runs": []}

    for n in args.sizes:
        probs = problems(n, args.problems, seed=n)
        make = lambda cat: (lambda opts: Decision(f"Which of these is a {cat}?", tuple(opts)))
        full_ds = [make(cat)(opts) for cat, opts, _ in probs]
        t = time.perf_counter()
        full = []
        for d in full_ds:  # one decision per problem, timed like a single call
            full.append(engine.decide(d))
        full_s = (time.perf_counter() - t) / len(probs)
        full_choice = [r.choice for r in full]
        row = {"n": n, "full": {
            "accuracy": sum(c == a for c, (_, _, a) in zip(full_choice, probs)) / len(probs),
            "latency_ms_per_problem": 1000 * full_s,
            "prompt_tokens_mean": statistics.mean(r.prompt_tokens for r in full),
            "mean_chosen_prob": statistics.mean(r.prob for r in full),
            "mean_outside_mass": statistics.mean(r.outside_mass for r in full),
        }, "tournaments": []}
        print(f"n={n:3} full       acc={row['full']['accuracy']:.0%} {row['full']['latency_ms_per_problem']:.0f} ms/problem "
              f"prompt~{row['full']['prompt_tokens_mean']:.0f} tok p={row['full']['mean_chosen_prob']:.2f}", flush=True)
        for g in args.groups:
            times, winners, rounds, survived = [], [], [], []
            for cat, opts, answer in probs:  # one problem at a time, rounds batched
                tour = Tournament(tuple(opts), g, make(cat))
                t = time.perf_counter()
                run_tournaments(engine.decide_batch, [tour])
                times.append(time.perf_counter() - t)
                winners.append(tour.winner)
                rounds.append(len(tour.rounds))
                survived.append(answer in [tour.options[i] for i in tour.rounds[0].winners])
            acc = sum(w == a for w, (_, _, a) in zip(winners, probs)) / len(probs)
            agree = sum(w == f for w, f in zip(winners, full_choice)) / len(probs)
            tr = {"group_size": g, "accuracy": acc, "agreement_with_full": agree,
                  "latency_ms_per_problem": 1000 * statistics.median(times),
                  "latency_ms_range": [1000 * min(times), 1000 * max(times)], "rounds_mean": statistics.mean(rounds),
                  # Did the right answer win its first-round group? Separates group errors from final errors.
                  "answer_survived_round1": sum(survived) / len(probs)}
            row["tournaments"].append(tr)
            print(f"n={n:3} tour g={g:3} acc={acc:.0%} agree-with-full={agree:.0%} "
                  f"survived-r1={tr['answer_survived_round1']:.0%} {tr['latency_ms_per_problem']:.0f} ms/problem "
                  f"rounds={tr['rounds_mean']:.1f}", flush=True)
        report["runs"].append(row)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"tournament_compare_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
