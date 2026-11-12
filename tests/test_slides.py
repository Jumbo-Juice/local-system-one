"""The final deck (slides/): data.js rebuilds from the committed files and says what the handoff's tables say,
every binding on a slide resolves, nothing reaches the network, the deck is static, and the talk fits 15 minutes."""

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

SLIDES = Path(__file__).resolve().parent.parent / "slides"
spec = importlib.util.spec_from_file_location("build_data", SLIDES / "build_data.py")
build_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_data)

MAX_TALK_S = 12 * 60  # 15 minutes everything included (owner, 2026-11-04), so 3:00 is left for questions
TIMING = re.compile(r"\[(\w+) · (\d+):(\d\d)\]")


@pytest.fixture(scope="module")
def data():
    return build_data.build()


@pytest.fixture(scope="module")
def html():
    return (SLIDES / "index.html").read_text(encoding="utf-8")


def sections(html):
    return re.findall(r"<section\b[^>]*>.*?</section>", html, flags=re.S)


def test_data_js_is_what_the_build_writes():
    assert (SLIDES / "data.js").read_text(encoding="utf-8") == build_data.render()


@pytest.mark.parametrize("key,mean,low,high,auc,stopped,through,declined,reviews", [
    ("signal_amt", 1943, 1651, 2338, .883, 404, 23562, 928, 702),
    ("signal_cust", 1992, 1683, 2383, .883, 335, 29338, 247, 804),
    ("baseline", 2106, 1946, 2292, .853, 384, 26105, 1051, 729),
    ("rich_amt", 2427, 2023, 2896, .859, 375, 30161, 1081, 895),
    ("approve_all", 5233, 4312, 6121, None, 0, 104651, 0, 0),
])
def test_final_setups_match_the_handoff(data, key, mean, low, high, auc, stopped, through, declined, reviews):
    s = data["final"]["setups"][key]
    assert round(s["mean"]) == mean and [round(v) for v in s["ci"]] == [low, high]
    assert s["auc"] is None if auc is None else round(s["auc"], 3) == auc
    assert (s["frauds_stopped"], round(s["fraud_let_through"]), s["good_declined"], s["reviews"]) == (stopped, through, declined, reviews)
    assert len(s["windows"]) == 20


def test_readout_matches_the_handoff(data):
    s = data["final"]["setups"]["rich_cust"]
    assert (round(s["mean"]), s["good_declined"], s["good_touched"], s["frauds_stopped"], round(s["fraud_let_through"]), s["reviews"]) == (2417, 444, 1327, 311, 33508, 974)
    told = data["final"]["told"]
    assert (round(told["rich_cust"]), round(told["baseline"]), round(told["diff"]), [round(v) for v in told["ci"]]) == (2639, 2632, 7, [-405, 481])


@pytest.mark.parametrize("key,mean,low,high,distinguishable", [
    ("rich_amt_baseline", 321, -74, 774, False),
    ("signal_amt_baseline", -163, -462, 238, False),
    ("signal_cust_baseline", -115, -391, 273, False),
    ("rich_amt_signal_amt", 484, 103, 972, True),
    ("signal_amt_approve", -3290, -4127, -2422, True),
    ("rich_cust_baseline", 311, -75, 746, False),
    ("rich_cust_rich_amt", -10, -91, 87, False),
    ("rich_cust_approve", -2816, -3631, -1986, True),
    ("rich_amt_approve", -2806, -3631, -1970, True),
])
def test_final_differences_are_first_minus_second(data, key, mean, low, high, distinguishable):
    d = data["final"]["diffs"][key]
    assert round(d["mean"]) == mean and [round(v) for v in d["ci"]] == [low, high] and d["distinguishable"] is distinguishable
    assert d["ci"][0] < d["mean"] < d["ci"][1]


@pytest.mark.parametrize("key,mean,auc,declined,stopped", [
    ("frozen", 3697, .7540, 9586, 1808), ("phase12", 2288, .8795, 3644, 2169), ("seed1", 2186, .8815, 3253, 2194),
    ("rich", 2112, .8932, 3459, 2324), ("logreg", 2125, .8882, 4003, 2347), ("signal", 1920, .9102, 3241, 2410),
    ("signal_alone", 1853, .9108, 3052, 2452), ("trees", 1830, .9139, 3003, 2445),
])
def test_ladder_matches_the_handoff(data, key, mean, auc, declined, stopped):
    s = data["dev"]["setups"][key]
    assert (round(s["mean"]), round(s["auc"], 4), s["good_declined"], s["frauds_stopped"]) == (mean, auc, declined, stopped)


@pytest.mark.parametrize("key,mean,low,high,distinguishable", [
    ("frozen_phase12", 1409, 1188, 1641, True), ("rich_phase12", -176, -281, -58, True), ("rich_logreg", -13, -142, 125, False),
    ("signal_signal_alone", 67, -33, 166, False), ("rich_trees", 282, 151, 423, True), ("seed1_phase12", -102, -205, 6, False),
])
def test_dev_differences_are_first_minus_second(data, key, mean, low, high, distinguishable):
    d = data["dev"]["diffs"][key]
    assert round(d["mean"]) == mean and [round(v) for v in d["ci"]] == [low, high] and d["distinguishable"] is distinguishable


def test_training_on_cost_matches_the_handoff(data):
    c = data["cost"]
    assert c["lambda"] == 5
    d = c["diffs"]["weighted_rich"]
    assert (round(d["mean"]), [round(v) for v in d["ci"]], d["first_cheaper"], d["distinguishable"]) == (53, [-39, 142], 46, False)
    assert round(c["diffs"]["weighted_logreg"]["mean"]) == 40 and round(c["diffs"]["weighted_trees"]["mean"]) == 335
    w = c["waterfall"]
    assert (round(w["start"]), round(w["fraud"]), round(w["declines"]), round(w["reviews"]), round(w["end"])) == (2112, 124, -61, -10, 2165)
    assert w["start"] + w["fraud"] + w["declines"] + w["reviews"] == pytest.approx(w["end"], abs=.02)
    assert (c["setups"]["rich"]["big_stopped"], c["setups"]["weighted"]["big_stopped"], c["big_frauds"]) == (258, 259, 275)


def test_bands_add_up_to_the_setups(data):
    rows = data["manual"]["bands"]["rows"]
    assert sum(r["frauds"] for r in rows) == data["cost"]["frauds"]
    assert sum(r["rich"] for r in rows) == data["cost"]["setups"]["rich"]["frauds_stopped"]
    assert sum(r["weighted"] for r in rows) == data["cost"]["setups"]["weighted"]["frauds_stopped"]


def test_earlier_phases_and_latency(data):
    p14, p18 = data["rule"]["phase14"], data["rule"]["phase18"]
    assert (round(p14["lines"]), round(p14["rule"]), round(p14["diff"]["mean"]), [round(v) for v in p14["diff"]["ci"]]) == (2752, 1926, -825, [-1307, -464])
    assert (p18["base"], p18["customers"], round(p18["diff"]["mean"])) == (476, 153, 119)
    assert round(data["final"]["latency"]["p90"]) == 56 and data["final"]["latency"]["bar"] == 250
    assert round(data["final"]["rank_corr"], 3) == .984 and round(data["dev"]["echo"]["rank_corr"], 3) == .984  # no double rounding
    assert round(p14["latency"]["p90"]) == 128


def test_only_paysim_rows_reach_the_deck(data):
    assert data["paysim"]["model_n"] == data["paysim"]["model_declined"] == 257
    # window-level IEEE-CIS only: row ranges and per-window costs, never a row's amount
    assert not re.search(r"TransactionAmt|TransactionID|card1", json.dumps(data))
    assert {r["type"] for r in data["paysim"]["replay"]} <= {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    assert not re.search(r"oldbalance|newbalance", json.dumps(data))


def test_every_binding_resolves(data, html):
    paths = re.findall(r"\{\{\s*([\w.]+)", html)
    assert paths
    for path in paths:
        node = data
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        assert node is not None, path


def test_no_network_at_show_time(html):
    files = [SLIDES / "index.html", SLIDES / "deck.css", *SLIDES.glob("js/*.js")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        text = re.sub(r"<aside class=\"notes\">.*?</aside>", "", text, flags=re.S)
        assert not re.search(r"(?:src|href)=\"https?:|url\(\s*[\"']?https?:|@import|fetch\(", text), f
    for ref in re.findall(r"(?:src|href)=\"([^\"#]+)\"", html):
        assert ref.startswith("data:") or (SLIDES / ref).exists(), ref


def test_reveal_is_pinned_and_licensed():
    assert (SLIDES / "vendor/reveal/LICENSE").exists() and (SLIDES / "vendor/fonts/archivo/OFL.txt").exists()
    assert "6.0.2" in (SLIDES / "README.md").read_text(encoding="utf-8")


def test_every_slide_has_notes_and_the_talk_fits(html):
    """15 presented slides, each with notes and a [Name · m:ss] timing, and one references slide with no timing."""
    total = 0
    leaves = [s for s in sections(html) if 'class="page' in s]
    assert len(leaves) == 16
    refs = [s for s in leaves if 'class="refs' in s]
    assert len(refs) == 1 and not TIMING.search(refs[0]), "one references slide, not presented"
    for s in leaves:
        assert '<aside class="notes">' in s, s[:120]
        if s in refs:
            continue
        m = TIMING.search(s)
        assert m, s[:120]
        total += int(m[2]) * 60 + int(m[3])
    assert 0 < total <= MAX_TALK_S, total


def test_no_scenes(html):
    """The deck is static: no scene machinery, no Auto-Animate, no backup sub-slides."""
    assert "data-scene" not in html and "scenes.js" not in html
    assert not (SLIDES / "js" / "scenes.js").exists()
    assert "data-auto-animate" not in html and "data-print-skip" not in html
    assert not re.search(r"<section\b[^>]*>\s*<section\b", html), "no vertical stacks: no backup sub-slides"
    for f in SLIDES.glob("js/*.js"):
        assert "scenes" not in f.read_text(encoding="utf-8"), f


def visible_text(html):
    """What the room reads: no notes, comments, tags or {{bindings}} (a binding is a number from data, not prose)."""
    text = re.sub(r"<aside class=\"notes\">.*?</aside>", " ", html, flags=re.S)
    text = re.sub(r"<!--.*?-->|<head>.*?</head>|<script\b.*?</script>", " ", text, flags=re.S)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    return re.sub(r"<[^>]+>", " ", text)


def rule_breaks(text):
    """Phase numbers, setup ids (case-sensitive, whole words), chain and window ids, file paths, 'econometric'."""
    banned = [r"(?i)\bphase \d+", r"(?i)econometric", r"fraud/", r"\.py\b", r"\.jsonl?\b", r"\b(?:dev|test)\d+\b"]
    banned += [rf"(?<![\w-]){re.escape(s)}(?![\w-])" for s in build_data.NAMES]
    return [m[0] for b in banned for m in [re.search(b, text)] if m]


def test_visible_text_follows_the_rules(html):
    for bad in ["Phase 14", "econometrics", "see fraud/SUMMARY.md", "finetune.py", "a .jsonl trace", "window dev0", "final-rich-amt"]:
        assert rule_breaks(bad), bad
    assert not rule_breaks("a rule-based baseline on the test windows, the final test, data.js, final-rich-amtx")
    assert rule_breaks(visible_text(html)) == []


def test_print_shows_every_step():
    """?print-pdf is one page per slide in its final state: core.js shows every [data-from] element when printing."""
    core = (SLIDES / "js" / "core.js").read_text(encoding="utf-8")
    assert re.search(r"const PRINT = /print-pdf/\.test\(location\.search\)", core)
    assert re.search(r"if \(PRINT\)[^\n]*querySelectorAll\('\[data-from\]'\)[^\n]*classList\.add\('shown'\)", core)
    main = (SLIDES / "js" / "main.js").read_text(encoding="utf-8")
    assert "pdfSeparateFragments: false" in main


@pytest.mark.parametrize("banned", [r"slightly (?:cheaper|better)", r"beats? (?:logistic regression|logreg)", r"\d+ ?% probability", r"probability of fraud"])
def test_the_words_the_handoff_rules_out(html, banned):
    visible = re.sub(r"<aside class=\"notes\">.*?</aside>", "", html, flags=re.S)
    assert not re.search(banned, visible, flags=re.I)


def test_earlier_phases_carry_their_own_window_counts(data):
    """Each earlier phase ran on its own test windows, so the backup table says how many and binds every number."""
    r = data["rule"]
    assert (r["phase14"]["windows_n"], r["phase17"]["windows_n"], r["phase18"]["windows_n"], r["phase20"]["windows_n"]) == (20, 8, 10, 10)
    assert r["phase14"]["diff"]["first_cheaper"] == 18 and r["phase14"]["diff"]["windows"] == 20
    p17, p20 = r["phase17"], r["phase20"]
    assert (round(p17["diff"]["mean"]), [round(v) for v in p17["diff"]["ci"]], p17["diff"]["distinguishable"]) == (1002, [405, 1870], True)
    assert (round(p17["dev"]["mean"]), [round(v) for v in p17["dev"]["ci"]], p17["dev"]["windows"]) == (626, [475, 780], 88)
    assert (round(p20["diff"]["mean"]), [round(v) for v in p20["diff"]["ci"]], p20["diff"]["distinguishable"]) == (-97, [-385, 182], False)
    assert (round(p20["dev"]["mean"]), [round(v) for v in p20["dev"]["ci"]]) == (-295, [-411, -173])
    assert [round(v) for v in r["phase18"]["diff"]["ci"]] == [-2, 245] and r["phase18"]["diff"]["distinguishable"] is False


def test_operating_points_and_the_sign_test_in_the_notes(data):
    f = data["final"]
    rich, base = f["setups"]["rich_amt"], f["setups"]["baseline"]
    assert (round(rich["recall"], 3), round(rich["precision"], 3)) == (.595, .165)
    assert (round(base["recall"], 3), round(base["precision"], 3)) == (.610, .184)
    assert base["frauds_stopped"] > rich["frauds_stopped"] and base["precision"] > rich["precision"]  # the slide says "no", on both counts
    assert [round(v, 3) for v in rich["auc_ci"]] == [.831, .885] and [round(v, 3) for v in base["auc_ci"]] == [.821, .880]
    sign = f["sign"]
    assert (round(sign["signal_amt_baseline"]["p"], 3), sign["signal_amt_baseline"]["first_cheaper"]) == (.041, 15)
    assert (round(sign["rich_amt_baseline"]["p"], 2), sign["rich_amt_baseline"]["first_cheaper"]) == (.50, 8)
    assert round(sign["final12_gap"]) == 2951 and f["dearest"]["window"] == "final12"
    assert round(f["prompt_tokens"]) == 345



def test_every_format_on_a_slide_exists(html):
    core = (SLIDES / "js" / "core.js").read_text(encoding="utf-8")
    block = core[core.index("const formats = {"):core.index("const get =")]
    known = set(re.findall(r"^\s+(\w+)(?::|,)", block, flags=re.M)) | {"int", "signed"}
    used = set(re.findall(r"\{\{\s*[\w.]+\|(\w+)", html))
    assert used and used <= known, used - known



def all_diffs(data):
    """Every comparison in data.js with its unit, as the console words it: chains for the dev-only evals."""
    for block, unit in (("final", "window"), ("dev", "chain"), ("cost", "chain")):
        for key, d in data[block]["diffs"].items():
            yield f"{block}.{key}", d, unit
    for phase, p in data["rule"].items():
        yield f"rule.{phase}", p["diff"], "window"


def test_the_deck_says_each_comparison_in_the_consoles_own_sentence(data):
    """build_data.sentence() is the Python twin of the console's sentence() (fraud/viewer/console.js): run the console's
    own code in node on every comparison in data.js, and on every name, and require the same text."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = (SLIDES.parent / "fraud" / "viewer" / "console.js").read_text(encoding="utf-8")
    names = js[js.index("const SETUP_NAME = {"):js.index("function setupLegend")]
    money = re.search(r"^const money = .*$", js, flags=re.M)[0]
    say = js[js.index("function sentence(c, unit)"):js.index("function sentences(")]
    cases = [{"key": k, "c": {"first": d["first"], "second": d["second"], "mean": d["mean"], "lo": d["ci"][0], "hi": d["ci"][1], "dist": d["distinguishable"]},
              "unit": u, "said": d["said"]} for k, d, u in all_diffs(data)]
    t = data["final"]["told"]
    cases.append({"key": "final.told", "c": {"first": "final-rich-cust", "second": "final-logreg-amt", "mean": t["diff"], "lo": t["ci"][0], "hi": t["ci"][1],
                                             "dist": t["ci"][0] > 0 or t["ci"][1] < 0}, "unit": "window", "said": t["said"]})
    script = ("const el = (tag, attrs, ...kids) => kids.join('');\n" + names + "\n" + money + "\n" + say +
              "\nconst inp = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "\nprocess.stdout.write(JSON.stringify({said: inp.cases.map((x) => sentence(x.c, x.unit).join('')), names: inp.ids.map(shortName)}));")
    ids = sorted(build_data.NAMES)
    out = json.loads(subprocess.run([node, "-e", script], input=json.dumps({"cases": cases, "ids": ids}), capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["names"] == [build_data.NAMES[s] for s in ids]
    assert len(cases) > 20
    for case, console in zip(cases, out["said"], strict=True):
        assert case["said"] == console, case["key"]


def test_every_comparison_runs_in_its_registered_order(data):
    """The deck states every comparison as the console does: the registered first minus the second. build_data.diff()
    refuses the other way round, so a reversed comparison fails the build, and every key names the first setup first."""
    entry = {"a": "lm-frozen-amt", "b": "lm-phase12-amt", "better": "lm-phase12-amt", "other": "lm-frozen-amt",
             "cheaper": 20, "dearer": 0, "ties": 0, "mean_diff": -1409.0, "mean_diff_ci95": [-1641.0, -1188.0], "distinguishable": True}
    assert build_data.diff(entry, "lm-frozen-amt", "chain")["mean"] == 1409
    with pytest.raises(ValueError, match="registered order"):
        build_data.diff(entry, "lm-phase12-amt", "chain")
    for key, d, _ in all_diffs(data):
        assert "registered_order" not in d, key
    assert {"frozen_phase12", "rich_phase12"} <= set(data["dev"]["diffs"]) and "rich_amt_signal_amt" in data["final"]["diffs"]




def test_worked_payment_is_paysim_and_its_orders_disagree(data):
    w = data["worked"]
    assert w["type"] == "TRANSFER"
    assert w["top"]["listed"] != w["top"]["reversed"]
    for k in ("listed", "reversed"):
        assert abs(sum(w["orders"][k]) - 1) < 1e-6
        assert max(w["logits"][k]) == 0
        e = [math.exp(x) for x in w["logits"][k]]
        s = sum(e)
        assert all(abs(a / s - b) < 1e-9 for a, b in zip(e, w["orders"][k]))


def test_worked_rule_raises_when_nothing_matches(monkeypatch, tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"type": "header"}) + "\n" + json.dumps({"type": "tx", "by": "filter", "tx": {"type": "TRANSFER"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(build_data, "TRACE", p)
    with pytest.raises(LookupError):
        build_data.worked_block()


def test_approve_all_accuracy(data):
    f = data["final"]
    assert f["approve_all_accuracy"] == 1 - f["frauds"] / f["rows"]
    assert 0.9 < f["approve_all_accuracy"] < 1


def test_cost_stress_edge_is_in_the_manual_data(data):
    s = data["manual"]["stress"]
    assert (s["flat_low"], s["flat_high"]) == (50, 60)
    assert "RESULTS-COST-STRESS" in s["_source"]


def test_illustration_numbers_are_labelled(html):
    """Rules, logistic regression and trees recorded no score for the worked payment, so their numbers are made up
    (manual_data.json → illustration), and every slide that shows one says "illustration" where the room can read it."""
    showing = [s for s in sections(html) if "manual.illustration." in s]
    assert len(showing) >= 3, "the rule, logistic regression and trees slides show the mechanism on illustration numbers"
    for s in showing:
        assert re.search(r"\billustration\b", visible_text(s), flags=re.I), s[:120]


def test_every_manual_block_names_its_source(data):
    for key, block in data["manual"].items():
        if isinstance(block, dict):
            assert block.get("_source"), key
    assert "Implementation choice" in data["manual"]["illustration"]["_source"]
    assert "[1]" in data["manual"]["nilson"]["_source"]


@pytest.mark.parametrize("n,name,time", [
    (1, "Juji", "0:25"), (2, "Juji", "0:45"), (3, "Juji", "1:00"), (4, "Juji", "0:45"),
    (5, "Swan", "0:45"), (6, "Swan", "0:50"), (7, "Swan", "0:45"),
])
def test_the_first_slides_follow_the_running_order(html, n, name, time):
    """Slides 01–07 as the spec's running order: who presents, for how long, on the tag and in the notes."""
    s = next((s for s in sections(html) if re.match(rf'<section id="s{n:02d}" data-n="{n}"', s)), None)
    assert s, n
    assert f'<span class="tag">{n} · {name}</span>' in s
    assert f"[{name} · {time}]" in s
