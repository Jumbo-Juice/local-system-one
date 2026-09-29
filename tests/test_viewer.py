"""The Master Viewer (python -m viewer): pool listing, the server's file rules, bundled pages, and
one game viewer per game demo."""

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from demo.runs import RUNS, to_jsonl
from viewer import server

HEADER = {"type": "header", "scenario": "shooter", "created": "2026-09-29T13:33:06",
          "backend": {"kind": "hf", "model": "Qwen/Qwen2.5-3B-Instruct"}, "map": {"seed": 4}}


def write(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_jsonl(list(records)), encoding="utf-8")
    return path


def test_pool_lists_every_run_newest_first_with_its_outcome(tmp_path):
    write(tmp_path / "shooter" / "20260929-133306_qwen2.5-3b_seed4.jsonl", HEADER, {"type": "tick"},
          {"type": "end", "outcome": "died", "tick": 98})
    write(tmp_path / "shooter" / "20260927-002350_classic_qwen2.5-1.5b_seed0.pinned.jsonl",
          {**HEADER, "created": "2026-09-27T00:23:50"}, {"type": "tick"})  # never finished
    write(tmp_path / "dungeon" / "20260926-162605_qwen2.5-3b_seed0.jsonl",
          {**HEADER, "scenario": "dungeon", "created": "2026-09-26T16:26:05"}, {"type": "end", "outcome": "escaped", "tick": 7})
    listing = server.Pool(tmp_path).listing()
    assert listing["pools"] == ["dungeon", "shooter"]
    runs = listing["runs"]
    assert [r["created"][:10] for r in runs] == ["2026-09-29", "2026-09-27", "2026-09-26"]
    newest, pinned = runs[0], runs[1]
    assert newest == {"path": "runs/shooter/20260929-133306_qwen2.5-3b_seed4.jsonl", "game": "shooter", "scenario": "shooter",
                      "label": "qwen2.5-3b_seed4", "created": "2026-09-29T13:33:06", "model": "Qwen2.5-3B-Instruct",
                      "seed": 4, "pinned": False, "outcome": "died", "tick": 98}
    assert pinned["pinned"] and pinned["label"] == "classic_qwen2.5-1.5b_seed0" and pinned["outcome"] is None


def test_a_broken_trace_is_skipped_not_fatal(tmp_path):
    write(tmp_path / "shooter" / "20260929-133306_ok.jsonl", HEADER, {"type": "end", "outcome": "escaped", "tick": 3})
    (tmp_path / "shooter" / "20260929-140000_broken.jsonl").write_text("not json\n", encoding="utf-8")
    assert [r["label"] for r in server.Pool(tmp_path).listing()["runs"]] == ["ok"]


@pytest.fixture
def served():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(server.Pool(), {}, verbose=False))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/"
    httpd.shutdown()
    httpd.server_close()


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as res:
        return res.status, res.read().decode("utf-8")


def test_server_serves_the_page_the_scripts_the_pools_and_the_listing(served):
    status, page = fetch(served)
    assert status == 200 and "<title>System One Runs</title>" in page
    for src in re.findall(r'<script src="([^"]+)"></script>', page):
        assert fetch(served + src)[0] == 200, src
    listing = json.loads(fetch(served + "api/runs")[1])
    pinned = [r for r in listing["runs"] if r["pinned"]]
    assert pinned and fetch(served + pinned[0]["path"])[1].startswith('{"type":"header"')


@pytest.mark.parametrize("path", ["server.py", "__main__.py", "runs/../demo/runs.py", "runs/shooter/../../README.md",
                                  "runs/README.md", "..%2Fdemo%2Fruns.py", "runs/nope/missing.jsonl"])
def test_server_serves_nothing_else(served, path):
    with pytest.raises(urllib.error.HTTPError) as err:
        fetch(served + path)
    assert err.value.code == 404


def test_bundle_inlines_every_script_and_embeds_the_trace_safely(tmp_path):
    trace = write(tmp_path / "run.jsonl", {**HEADER, "note": "</script><b>"}, {"type": "end", "outcome": "died", "tick": 1})
    page = server.bundle(trace, tmp_path / "run.html").read_text(encoding="utf-8")
    assert '<script src="' not in page, "a bundled page must open from disk with no other files"
    assert "</script><b>" not in page
    slot = re.search(r'<script id="trace" type="application/x-ndjson" data-name="run.jsonl">(.*?)</script>', page, re.S)
    assert [json.loads(x) for x in slot.group(1).splitlines()][0]["note"] == "</script><b>"
    assert page.count('id="trace"') == 1


def test_every_game_demo_has_its_own_viewer():
    """A game demo's runs pool into runs/<game>/ and it draws itself from viewer/games/<game>.js,
    which the page loads. The pinned runs show which game demos exist."""
    page = (server.HERE / "index.html").read_text(encoding="utf-8")
    games = sorted(p.stem for p in (server.HERE / "games").glob("*.js"))
    for game in games:
        assert f'<script src="games/{game}.js"></script>' in page, f"index.html does not load games/{game}.js"
        code = (server.HERE / "games" / f"{game}.js").read_text(encoding="utf-8")
        assert f'scenario: "{game}"' in code, f"games/{game}.js must register scenario {game!r}"
    for pool in sorted(p.name for p in RUNS.iterdir() if p.is_dir() and any(p.glob("*.pinned.jsonl"))):
        assert pool in games, f"runs/{pool}/ has no viewer: add viewer/games/{pool}.js (see viewer/README.md)"
