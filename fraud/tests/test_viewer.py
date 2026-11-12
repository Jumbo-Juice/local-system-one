from pathlib import Path
import re
import json

from fraud.viewer import server

HEADER = {"type": "header", "app": "fraud", "setup": "rules", "window": {"id": "test100", "split": "test"},
          "backend": {}, "created": "2026-10-03T05:22:00"}
END = {"type": "end", "summary": {"cost": 1100.0, "recall": 1.0, "total_ms": 0.2, "model_decisions": 0}}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_only_console_files_runs_and_results_are_served(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), (results / "eval").mkdir(parents=True), (results / "eval_d3ft").mkdir(), (results / "eval_d1ft").mkdir(), (results / "eval_d3").mkdir()
    write(runs / "a.jsonl", [HEADER, END])
    for d in ("eval", "eval_d3ft", "eval_d1ft", "eval_d3"):
        (results / d / "e.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    ok = lambda route: server.resolve(route, runs, results)
    assert ok("") == server.HERE / "index.html" and ok("console.js") == server.HERE / "console.js"
    assert ok("runs/a.jsonl") == runs / "a.jsonl" and ok("results/eval/e.json") == results / "eval" / "e.json"
    assert ok("results/eval_d3ft/e.json") == results / "eval_d3ft" / "e.json"
    assert ok("results/eval_d1ft/e.json") == results / "eval_d1ft" / "e.json"
    for bad in ("runs/../secret.txt", "runs/..", "secret.txt", "server.py", "runs/a.json", "results/eval/../../secret.txt",
                "runs/missing.jsonl", "../fraud/data/paysim.npz", "results/e.json", "results/eval_d3/e.json",
                "results/../secret.txt", "results/eval/sub/e.json"):
        assert ok(bad) is None, bad


def test_listing_reads_header_and_end(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), (results / "eval").mkdir(parents=True), (results / "eval_d3ft").mkdir()
    write(runs / "20261003-052158_eval_rules_test100.jsonl", [HEADER, {"type": "tx"}, END])
    write(runs / "20261003-060000_hybrid_mock_dev0.jsonl", [dict(HEADER, setup="hybrid", window={"id": "dev0"})])
    ev = lambda i, t: json.dumps({"eval": i, "created": t, "complete": True, "bars": {}, "totals": {}})
    (results / "eval" / "20261003-052158.json").write_text(ev("20261003-052158", "2026-10-03T06:00:00"), encoding="utf-8")
    (results / "eval_d3ft" / "20261021-155222.json").write_text(ev("20261021-155222", "2026-10-21T16:16:00"), encoding="utf-8")
    (results / "eval_d3ft" / "20261021-130745_repro.json").write_text(json.dumps({"check": "repro"}), encoding="utf-8")
    lst = server.Listing(runs, results)
    by_file = {r["file"]: r for r in lst.run_list()["runs"]}
    ev = by_file["20261003-052158_eval_rules_test100.jsonl"]
    assert ev["eval"] == "20261003-052158" and ev["finished"] and ev["cost"] == 1100.0 and ev["window"] == "test100"
    assert not by_file["20261003-060000_hybrid_mock_dev0.jsonl"]["finished"]
    evals = lst.eval_list()["evals"]
    assert [e["file"] for e in evals] == ["eval_d3ft/20261021-155222.json", "eval/20261003-052158.json"]  # newest first, no repro


def test_page_loads_its_own_files_only():
    page = (server.HERE / "index.html").read_text(encoding="utf-8")
    assert 'src="console.js"' in page and 'href="console.css"' in page
    assert "shell.js" not in page and "games/" not in page
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert "uncalibrated" in js and "fetch(" in js


def test_ieee_evals_are_listed_and_served_and_other_files_in_their_folders_are_not_evals(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), (results / "eval_rule").mkdir(parents=True), (results / "eval_final").mkdir()
    (results / "eval_rule" / "20261103-211736.json").write_text(json.dumps({"eval": "20261103-211736", "created": "2026-11-03T21:18:00", "setups": {}, "script": "fraud/eval_rule.py"}), encoding="utf-8")
    (results / "eval_rule" / "dev_look.json").write_text(json.dumps({"rows": {}}), encoding="utf-8")
    evals = server.Listing(runs, results).eval_list()["evals"]
    assert [e["file"] for e in evals] == ["eval_rule/20261103-211736.json"] and evals[0]["ieee"] and evals[0]["script"] == "fraud/eval_rule.py"
    assert server.resolve("results/eval_rule/20261103-211736.json", runs, results) == results / "eval_rule" / "20261103-211736.json"


def test_a_chain_windows_trace_is_read_as_an_eval_run(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write(runs / "20261103-211736_eval_rule-base_rule3.jsonl", [dict(HEADER, setup="rule-base", window={"id": "rule3", "split": "test"}), END])
    d = server.describe(runs / "20261103-211736_eval_rule-base_rule3.jsonl")
    assert d["eval"] == "20261103-211736" and d["window"] == "rule3" and d["finished"]


def test_the_run_page_shows_the_rule_and_says_scores_are_uncalibrated():
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert "function ruleCard" in js and "Amount-aware" in js and "Customer line" in js and "Two lines on the score" in js
    assert "probability" not in js.split("function ruleCard")[1].split("function about")[0].replace("not probabilities", "")


def test_every_final_and_dev_setup_has_a_console_name():
    js = (Path(__file__).resolve().parents[1] / "viewer" / "console.js").read_text(encoding="utf-8")
    for setup in ("final-logreg-amt", "final-signal-amt", "final-signal-cust", "final-rich-amt", "lm-signal-cust", "lm-rich+trees-amt"):
        assert f'"{setup}"' in js


def test_the_dev_evals_and_the_final_test_are_listed():
    from fraud.viewer.server import EVAL_DIRS

    assert {"eval_lm_nuc", "eval_combos", "eval_final"} <= set(EVAL_DIRS)


def test_the_ieee_eval_page_pairs_costs_with_the_windows_that_were_scored():
    """Phase 11's result lists its 10 dev windows beside the 20 test windows its costs are for. The page keeps only the scored ones."""
    import re

    results = Path(__file__).resolve().parents[1] / "results"
    r = json.loads((results / "eval_ieee" / "20261102-183431.json").read_text(encoding="utf-8"))
    scored = len(next(iter(r["setups"].values()))["windows"])
    assert len(r["windows"]) == 30 and scored == 20
    kept = [w for w in r["windows"] if len(r["windows"]) == scored or not re.match(r"^dev\d+$", w)][-scored:]  # the page's rule, in console.js ieeeEval
    assert kept == [f"test{k}" for k in range(20)]
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert r"all.length === scored || !/^dev\d+$/.test(w)" in js and ".slice(-scored)" in js


def test_every_committed_ieee_setup_has_a_name_on_the_console():
    import re

    results = Path(__file__).resolve().parents[1] / "results"
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert "function setupName" in js
    for d in server.EVAL_DIRS:
        for f in sorted((results / d).glob("2*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            for setup in r.get("setups", {}) if isinstance(r.get("setups"), dict) else ():
                patterned = re.match(r"^audit-(all|no[A-Z]+)-(amt|lines)$", setup) or re.match(r"^feat-(all|rich|uid|nocdv|nocdv-rich)-(amt|lines)$", setup)
                assert f'"{setup}"' in js or patterned, (f.name, setup)


def test_the_final_test_page_shows_the_owners_verdict_and_the_readout_as_a_readout():
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert "rich_cust_readout.json" in js and "extras.json" in js and '"final-rich-cust"' in js
    assert "read-out, not a test" in js and "The registered result above stands as it came out" in js
    assert "held-out dev chain" in js  # a dev eval's chart doesn't call its chains test windows
    results = Path(__file__).resolve().parents[1] / "results"
    readout = json.loads((results / "eval_final" / "rich_cust_readout.json").read_text(encoding="utf-8"))
    final = json.loads((results / "eval_final" / "20261104-090649.json").read_text(encoding="utf-8"))
    assert readout["eval"] == final["eval"] and set(readout["table"]) >= {"final-rich-cust", "final-rich-amt", "final-logreg-amt"}
    assert server.resolve("results/eval_final/rich_cust_readout.json") and server.resolve("results/eval_final/extras.json")


def test_the_cost_map_is_served_and_listed_for_its_page_but_not_as_an_eval(tmp_path):
    """Phase 16's cost-stress file has no totals or setups: the #/costs page finds it through /api/costs, the eval list leaves it out."""
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), (results / "eval_cost_stress").mkdir(parents=True)
    (results / "eval_cost_stress" / "20261103-210247.json").write_text(json.dumps({"run": "20261103-210247", "created": "2026-11-03T21:02:49", "families": {}, "stops": {}}), encoding="utf-8")
    (results / "eval_cost_stress" / "20261101-000000.json").write_text(json.dumps({"run": "20261101-000000", "families": {}}), encoding="utf-8")
    lst = server.Listing(runs, results)
    assert lst.eval_list()["evals"] == []
    assert lst.cost_maps()["files"] == ["eval_cost_stress/20261103-210247.json", "eval_cost_stress/20261101-000000.json"]  # newest first
    assert server.resolve("results/eval_cost_stress/20261103-210247.json", runs, results) == results / "eval_cost_stress" / "20261103-210247.json"
    assert server.resolve("results/eval_cost_stress/20261103-210247.md", runs, results) is None


def test_the_committed_cost_map_is_found_and_has_six_families_and_the_page_has_its_route():
    lst = server.Listing()
    files = lst.cost_maps()["files"]
    assert files and all(e["file"] not in files for e in lst.eval_list()["evals"])
    r = json.loads((server.RESULTS / files[0]).read_text(encoding="utf-8"))
    assert len(r["families"]) == 6 and "review fee" in r["families"]
    for rows in r["families"].values():
        for x in rows:
            assert {"logreg_deployed", "logreg_repicked", "adapter_deployed", "adapter_repicked"} <= set(x)
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    page = (server.HERE / "index.html").read_text(encoding="utf-8")
    assert 'h === "costs"' in js and 'getJSON("api/costs")' in js and 'href="#/costs"' in page
    assert "not corrected for multiple comparisons" in js and "Phase 12's adapter, not a final one" in js


def test_every_comparison_on_an_eval_page_is_the_first_setup_minus_the_second():
    """The eval files store mean_diff as the cheaper setup minus the dearer one. The console's compare() turns every committed
    comparison back into the registered order, a minus b (the deck does the same in slides/build_data.py diff())."""
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    results = Path(__file__).resolve().parents[1] / "results"
    entries = []
    for path in sorted(results.glob("*/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            continue
        for v in (d.get("verdict") or {}).values():
            entries += v if isinstance(v, list) else [v]
        entries += d.get("against") or [] if isinstance(d.get("against"), list) else []
    assert len(entries) > 100
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    body = js[js.index("function compare(p)"):js.index("const questionTag")]
    script = body + "\nprocess.stdout.write(JSON.stringify(JSON.parse(require('fs').readFileSync(0, 'utf8')).map(compare)));"
    out = json.loads(subprocess.run([node, "-e", script], input=json.dumps(entries), capture_output=True, text=True, check=True).stdout)
    for e, c in zip(entries, out, strict=True):
        flip = e["better"] != e["a"]
        low, high = e["mean_diff_ci95"]
        assert (c["first"], c["second"]) == (e["a"], e["b"])
        assert c["mean"] == pytest.approx(-e["mean_diff"] if flip else e["mean_diff"])
        assert [c["lo"], c["hi"]] == pytest.approx([-high, -low] if flip else [low, high])
        assert c["firstCheaper"] == (e["dearer"] if flip else e["cheaper"])
        assert c["dist"] == e["distinguishable"] and c["lo"] <= c["mean"] <= c["hi"]


def test_the_logreg_timing_is_served_for_the_final_page_but_not_listed_as_an_eval():
    """fraud/time_logreg.py's result feeds the final test's Cost and speed panel; it has no setups or totals, so it is no eval."""
    results = Path(__file__).resolve().parents[1] / "results"
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    name = re.search(r'^const LOGREG_TIMING = "results/time_logreg/([\w-]+\.json)";$', js, flags=re.M)[1]
    assert server.resolve(f"results/time_logreg/{name}") == results / "time_logreg" / name
    t = json.loads((results / "time_logreg" / name).read_text(encoding="utf-8"))
    assert t["setup"] == "final-logreg-amt" and 0 < t["latency_ms"]["p90"] < 1 and "dev" in t["rows_from"]
    assert not any(e["file"].startswith("time_logreg/") for e in server.Listing().eval_list()["evals"])
