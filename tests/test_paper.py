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


# ---- the paper itself (paper/main.typ) ----------------------------------------------------------------------

TYPST_WINGET = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Typst.Typst_Microsoft.Winget.Source_8wekyb3d8bbwe/typst-x86_64-pc-windows-msvc/typst.exe"
MAIN = ROOT / "paper" / "main.typ"
FOOTNOTE = re.compile(r"// code-and-data: start\n(.*?)// code-and-data: end\n", re.S)
REFS = re.compile(r"// references: start\n(.*?)// references: end\n", re.S)

slides_spec = importlib.util.spec_from_file_location("slides_build_data", ROOT / "slides" / "build_data.py")
slides = importlib.util.module_from_spec(slides_spec)
slides_spec.loader.exec_module(slides)


def typst() -> str | None:
    """Typst on PATH, or where winget put it (agent shells don't have it on PATH)."""
    return shutil.which("typst") or (str(TYPST_WINGET) if TYPST_WINGET.exists() else None)


def body_text(typ: str) -> str:
    """The prose a reader sees, as source: no "Code and results" footnote, no v() keys, no import or data-file line, no comments."""
    typ = FOOTNOTE.sub(" ", typ)
    typ = re.sub(r'#?v\(\s*"[^"]*"\s*\)', " ", typ)
    typ = re.sub(r'(?m)^#import .*$|json\("data\.json"\)', " ", typ)
    return re.sub(r"(?m)//.*$", " ", typ)


def rule_breaks(text: str) -> list[str]:
    """The Global Constraints: phase numbers, setup ids and chain or window ids, file paths, 'econometric';
    and the words the handoff rules out."""
    banned = [r"(?i)\bphase \d+", r"(?i)econometric", r"fraud/", r"\.py\b", r"\.jsonl?\b", r"\b(?:dev|test)\d+\b",
              r"(?i)slightly (?:cheaper|better)", r"(?i)beats? (?:logistic regression|logreg)", r"(?i)\d+ ?% probability",
              r"(?i)probability of fraud"]
    banned += [rf"(?<![\w-]){re.escape(s)}(?![\w-])" for s in slides.NAMES]
    return [m[0] for b in banned for m in [re.search(b, text)] if m]


def test_paper_compiles_to_a_main_text_and_appendices(tmp_path):
    """About 8 IEEE pages of main text plus appendices (REVISION_PROMPT.md, Stage 4)."""
    exe = typst()
    if not exe:
        pytest.skip("typst is not installed")
    out = subprocess.run([exe, "compile", str(MAIN), str(tmp_path / "p{p}.png")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    pages = sorted(tmp_path.glob("p*.png"))
    assert 8 <= len(pages) <= 14, len(pages)


def test_paper_has_its_verified_references():
    """About 25-30 references, listed by key in order of first citation, each cited in the text; System One only in a footnote.
    Each was checked against a primary source (REVISION_LOG.md, Stage 4)."""
    typ = MAIN.read_text(encoding="utf-8")
    order = re.findall(r'"([a-z0-9]+)"', re.search(r"#let REFS = \(([^)]*)\)", typ)[1])
    block = REFS.findall(typ)
    assert len(block) == 1
    listed = re.findall(r'(?m)^#bibk\("([a-z0-9]+)"\)\[', block[0])
    assert listed == order and 25 <= len(order) <= 30
    body = REFS.sub(" ", typ)
    cited = [k for args in re.findall(r"#cite-k\(([^)]*)\)", body) for k in re.findall(r'"([a-z0-9]+)"', args)]
    assert set(cited) == set(order)
    prose = body[body.index("// ---- the paper"):]
    in_text = [k for args in re.findall(r"#cite-k\(([^)]*)\)", prose) for k in re.findall(r'"([a-z0-9]+)"', args)]
    assert list(dict.fromkeys(in_text)) == order, "references are numbered in order of first citation in the text"
    text = dict(re.findall(r'(?m)^#bibk\("([a-z0-9]+)"\)\[(.*)\]\s*$', block[0]))
    for key, words in {"pet": "Schick", "lmbff": "Gao", "lift": "Dinh", "robinson": "Robinson", "pezeshkpour": "Pezeshkpour", "elkan": "Elkan",
                       "bahnsen": "Correa Bahnsen", "nosek": "Nosek", "dror": "Dror", "efron": "Efron", "kunsch": "Künsch", "grinsztajn": "Grinsztajn",
                       "tabllm": "TabLLM", "tan": "Tan", "d1omni": "d1-Omni-2B"}.items():
        assert words in text[key], key
    assert "System One" not in block[0] and "System One" in typ


def test_paper_text_follows_the_rules():
    typ = MAIN.read_text(encoding="utf-8")
    for bad in ["Phase 14", "econometrics", "see fraud/SUMMARY.md", "finetune.py", "a .jsonl trace", "window dev0", "final-rich-amt",
                "slightly cheaper", "a 94% probability"]:
        assert rule_breaks(bad), bad
    assert not rule_breaks("a rule-based baseline, the final test, data, final-rich-amtx, a logistic regression")
    footnotes = FOOTNOTE.findall(typ)
    assert len(footnotes) == 1 and "footnote" in footnotes[0] and "Code and results" in footnotes[0]
    body = body_text(typ)
    assert rule_breaks(body) == []
    flat = re.sub(r"\s+", " ", body)
    # the settled framing (REVISION_PROMPT.md, Stage 4)
    assert "Can a Small Fine-Tuned LLM Make Fraud Decisions? A Locked, Cost-Based Evaluation" in flat
    assert "no detectable saving" in flat and "not a fairness audit" in flat and "This is the part that is new" not in flat
    assert "Bayes minimum-risk" in flat and "Platt scaling" in flat and "operating point" in flat and "verbalizer" in flat
    assert "option-order bias" in flat and "post-hoc" in flat and "Two flips" not in flat
    for gone in [r"Student ID", r"Group 13", r"Frog Watch", r"\bproposal\b", r"\bdeck\b"]:
        assert not re.search(gone, typ), gone
    assert "Siam University" in typ and "course project" in flat


def test_generated_block_is_what_the_generator_writes(tmp_path):
    """paper/gen_tables.py writes the generated block; the committed main.typ must match it."""
    spec = importlib.util.spec_from_file_location("gen_tables", ROOT / "paper" / "gen_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    copy = tmp_path / "main.typ"
    copy.write_bytes(MAIN.read_bytes())
    gen.MAIN = copy
    gen.main()
    assert copy.read_text(encoding="utf-8") == MAIN.read_text(encoding="utf-8")
