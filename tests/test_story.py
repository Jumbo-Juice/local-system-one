"""The story (story/): data.js rebuilds from the committed results and pinned runs and says what the
write-ups say, the page reaches no network, shows no IEEE-CIS row, and keeps to the glossary's words."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STORY = ROOT / "story"
spec = importlib.util.spec_from_file_location("story_build_data", STORY / "build_data.py")
build_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_data)


@pytest.fixture(scope="module")
def data():
    return build_data.build()


def page_files():
    return [p for p in STORY.rglob("*") if p.suffix in {".html", ".css", ".js"} and "vendor" not in p.parts and p.name != "data.js"]


def test_data_js_is_what_the_build_writes():
    assert (STORY / "data.js").read_text(encoding="utf-8") == build_data.render()


def test_engine_numbers_match_the_readme(data):
    speed = data["engine"]["speed"]
    assert (speed["look_ms"], speed["write_ms"], speed["times"]) == (99.8, 358.0, 3.6)
    table = {r["options"]: (round(r["at_once"] * 100), round(r["tournament"] * 100)) for r in data["engine"]["tournament"]}
    assert table[10] == (93, 93) and table[26] == (87, 93) and table[40] == (47, 100) and table[80] == (17, 80)


def test_game_numbers_match_the_readme(data):
    games = data["games"]
    assert (games["grid"]["gems_times"], games["grid"]["food_times"]) == (2.3, 5.4)
    assert games["escape"] == {"escaped": 40, "runs": 40}


def test_the_order_flip_is_a_recorded_decision_whose_top_option_changes(data):
    flip = data["games"]["flip"]
    top = lambda scores: flip["options"][scores.index(max(scores))]
    assert top(flip["listed"]) == "hold fire" and top(flip["reversed"]).startswith("shoot")
    for scores in (flip["listed"], flip["reversed"], flip["averaged"]):
        assert len(scores) == len(flip["options"]) and abs(sum(scores) - 1) < 0.01
    assert [round((a + b) / 2, 2) for a, b in zip(flip["listed"], flip["reversed"])] == [round(p, 2) for p in flip["averaged"]]


def test_replays_are_the_pinned_runs(data):
    bullet, escape = data["replays"]["bullet"], data["replays"]["escape"]
    assert (bullet["outcome"], bullet["cause"], bullet["end_tick"], bullet["kills"]) == ("died", "gunner", 48, 4)
    assert (escape["outcome"], escape["end_tick"], escape["hits"]) == ("escaped", 45, 0)
    assert bullet["warned_ticks"], "the run has moves into a labelled bullet path taken while a safe move was offered"
    for tick in bullet["warned_ticks"]:
        move = next(f["m"] for f in bullet["frames"] if f["t"] == tick)
        assert "BULLET" in move["o"][move["c"]] and any("safe" in option for option in move["o"])
    for name in ("flip", "bullet", "escape"):
        assert build_data.SOURCES[name].endswith(".pinned.jsonl") and (ROOT / build_data.SOURCES[name]).exists()


def test_fraud_numbers_are_the_decks(data):
    fraud = data["fraud"]
    means = {k: round(v["mean"]) for k, v in fraud["final"]["setups"].items()}
    assert means == {"baseline": 2106, "rich_amt": 2427, "rich_cust": 2417, "approve_all": 5233}
    against = fraud["final"]["diffs"]["rich_amt_baseline"]
    assert (round(against["mean"]), [round(v) for v in against["ci"]], against["distinguishable"]) == (321, [-74, 774], False)
    saving = fraud["final"]["diffs"]["rich_amt_approve"]
    assert round(saving["mean"]) == -2806 and saving["distinguishable"]
    trained = fraud["dev"]["diffs"]["frozen_phase12"]
    assert round(trained["mean"]) == 1409 and trained["distinguishable"]
    assert round(fraud["rule"]["diff"]["mean"]) == -825 and fraud["rule"]["diff"]["distinguishable"]
    assert fraud["final"]["latency"]["p90"] < fraud["final"]["latency"]["bar"]
    assert (fraud["paysim"]["matched"], fraud["paysim"]["decisions"]) == (3853, 3862)
    assert (fraud["final"]["setups"]["rich_amt"]["good_declined"], fraud["final"]["setups"]["rich_cust"]["good_declined"]) == (1081, 444)


def test_every_manual_block_names_its_source(data):
    for key, block in data["manual"].items():
        if not key.startswith("_"):
            assert block["_source"], key


def test_no_ieee_row_reaches_the_story(data):
    """Window-level results only: no IEEE-CIS column name or transaction id in data.js."""
    text = (STORY / "data.js").read_text(encoding="utf-8")
    for column in ("TransactionID", "TransactionAmt", "TransactionDT", "card1", "addr1", "P_emaildomain", "isFraud"):
        assert column not in text
    assert set(data["replays"]) == {"bullet", "escape"}


def test_gsap_is_the_vendored_checked_copy():
    source = (STORY / "vendor" / "gsap" / "SOURCE.txt").read_text(encoding="utf-8")
    import hashlib
    for name in ("gsap.min.js", "ScrollTrigger.min.js"):
        digest = hashlib.sha256((STORY / "vendor" / "gsap" / name).read_bytes()).hexdigest()
        assert digest in source, name


def test_the_page_reaches_no_network():
    for path in page_files():
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<a\b[^>]*>", "", text)  # links a reader may follow are not fetches
        text = text.replace("http://www.w3.org/", "")
        assert not re.search(r"""(?:src|href|url\()\s*=?\s*["'(]?\s*(?:https?:)?//""", text), path.name
        assert "fetch(" not in text and "XMLHttpRequest" not in text, path.name


def test_the_page_keeps_to_the_glossary():
    """The story's own words (GLOSSARY.md, The story) and none of a lab notebook's."""
    html = (STORY / "index.html").read_text(encoding="utf-8")
    prose = re.sub(r"<script\b.*?</script>|<style\b.*?</style>|<[^>]+>", " ", html, flags=re.S)
    assert "Can a model that never writes be trusted to decide?" in prose
    for stamp in ("yes", "no", "only with help", "can’t tell"):
        assert stamp in prose.lower(), stamp
    for banned in (r"\bPhase \d", r"\blm-[a-z]", r"\bfinal-[a-z]+-", r"\bprobabilit", r"\bsignificant", r"\bverdict"):
        assert not re.search(banned, prose), banned
    assert "not Jev" in prose or "is not Jev" in prose


def test_every_binding_on_the_page_resolves(data):
    html = (STORY / "index.html").read_text(encoding="utf-8")
    paths = set(re.findall(r'data-(?:bind|fill|compare|dots|of)="([^"|]+)', html))
    assert paths, "the page binds its numbers to data.js"
    for path in paths:
        node = data
        for part in path.split("."):
            node = len(node) if part == "length" else node[int(part)] if isinstance(node, list) else node[part]
        assert node is not None, path
