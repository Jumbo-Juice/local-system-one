import json

from fraud.viewer import server

HEADER = {"type": "header", "app": "fraud", "setup": "rules", "window": {"id": "test100", "split": "test"},
          "backend": {}, "created": "2026-10-03T05:22:00"}
END = {"type": "end", "summary": {"cost": 1100.0, "recall": 1.0, "total_ms": 0.2, "model_decisions": 0}}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_only_console_files_runs_and_results_are_served(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), results.mkdir()
    write(runs / "a.jsonl", [HEADER, END])
    (results / "e.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    ok = lambda route: server.resolve(route, runs, results)
    assert ok("") == server.HERE / "index.html" and ok("console.js") == server.HERE / "console.js"
    assert ok("runs/a.jsonl") == runs / "a.jsonl" and ok("results/eval/e.json") == results / "e.json"
    for bad in ("runs/../secret.txt", "runs/..", "secret.txt", "server.py", "runs/a.json", "results/eval/../../secret.txt",
                "runs/missing.jsonl", "../fraud/data/paysim.npz", "results/e.json"):
        assert ok(bad) is None, bad


def test_listing_reads_header_and_end(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir(), results.mkdir()
    write(runs / "20261003-052158_eval_rules_test100.jsonl", [HEADER, {"type": "tx"}, END])
    write(runs / "20261003-060000_hybrid_mock_dev0.jsonl", [dict(HEADER, setup="hybrid", window={"id": "dev0"})])
    (results / "20261003-052158.json").write_text(json.dumps({"eval": "20261003-052158", "complete": True, "bars": {}}), encoding="utf-8")
    lst = server.Listing(runs, results)
    by_file = {r["file"]: r for r in lst.run_list()["runs"]}
    ev = by_file["20261003-052158_eval_rules_test100.jsonl"]
    assert ev["eval"] == "20261003-052158" and ev["finished"] and ev["cost"] == 1100.0 and ev["window"] == "test100"
    assert not by_file["20261003-060000_hybrid_mock_dev0.jsonl"]["finished"]
    assert lst.eval_list()["evals"][0]["id"] == "20261003-052158"


def test_page_loads_its_own_files_only():
    page = (server.HERE / "index.html").read_text(encoding="utf-8")
    assert 'src="console.js"' in page and 'href="console.css"' in page
    assert "shell.js" not in page and "games/" not in page
    js = (server.HERE / "console.js").read_text(encoding="utf-8")
    assert "uncalibrated" in js and "fetch(" in js
