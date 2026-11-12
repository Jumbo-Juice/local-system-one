"""The paper's numbers (paper/data.json) come from the deck's build, formatted as the deck formats them."""

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("paper_build_data", ROOT / "paper" / "build_data.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

TIES = [0.125, 0.0625, 0.8125, 0.03125, 1.005, 2.675, 0.015, -0.4, 2.5, -2.5]
INPUTS = {
    "int": [-115.4, 0.4, 1051, 2106.5, 305369.89] + TIES,
    "signed": [-115.4, 0.4, 1051, 2106.5, 305369.89] + TIES,
    "ci": [[-391.2, 273.8]],
    "range": [[-391.2, 273.8]],
    "auc": [0.8834, 0.0315] + [x for x in TIES if x > 0],
    "auc4": [0.8834, 0.0315] + [x for x in TIES if x > 0],
    "aucrange": [[0.8834, 0.0315]],
    "num": [1051, 0.8834],
    "abs": [-115.4, 0.4, 1051, 2106.5, 305369.89] + TIES,
    "pct": [0.8834, 0.0315] + TIES,
    "d2": [-0.004, -115.4, 0.0315, 0.8834] + TIES,
    "usd2": [305369.89, 0.4, 1051] + TIES,
    "dist": [{"distinguishable": False}, {"distinguishable": True}],
}


def test_formats_match_the_deck():
    """Run the deck's own format functions (slides/js/core.js) in node and require the same text."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = (ROOT / "slides" / "js" / "core.js").read_text(encoding="utf-8")
    block = js[js.index("const MINUS"):js.index("const get =")]
    script = block + "\nconst inp = JSON.parse(require('fs').readFileSync(0, 'utf8'));" \
        "\nprocess.stdout.write(JSON.stringify(Object.fromEntries(Object.entries(inp).map(([f, xs]) => [f, xs.map(formats[f])]))));"
    out = json.loads(subprocess.run([node, "-e", script], input=json.dumps(INPUTS), capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert set(paper.FORMATS) == set(out)
    for name, xs in INPUTS.items():
        assert [paper.FORMATS[name](x) for x in xs] == out[name], name
    assert paper.FORMATS["signed"](-115.4) == "\u2212115"


def test_data_json_is_what_the_build_writes():
    assert (ROOT / "paper" / "data.json").read_text(encoding="utf-8") == paper.render()


def test_every_key_in_the_paper_resolves():
    typ = (ROOT / "paper" / "main.typ").read_text(encoding="utf-8")
    written = json.loads((ROOT / "paper" / "data.json").read_text(encoding="utf-8"))
    keys = paper.keys(typ)
    assert keys
    for k in keys:
        assert k in written and written[k] != "None", k


def test_unresolved_key_raises():
    with pytest.raises(KeyError, match="final.nope"):
        paper.resolve({"final": {}}, "final.nope|int")
    with pytest.raises(KeyError, match="final.x"):
        paper.resolve({"final": {"x": None}}, "final.x|int")


def test_keys_reads_v_calls():
    assert paper.keys('a #v("x.y|int") b #v("z") #v("x.y|int")') == ["x.y|int", "z"]
