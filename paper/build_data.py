"""The paper's numbers: the deck's build, formatted as the deck's slides/js/core.js `formats` format them.

paper/main.typ reads paper/data.json through v("<path>|<fmt>"). The path is into slides/build_data.py's build()
output; no "|fmt" means String(value), as the deck's bindings do. Run `python paper/build_data.py` to rewrite it."""

import importlib.util
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINUS = "\u2212"


def deck() -> dict:
    """The deck's build, loaded by path: slides/ is a folder of files, not a package."""
    spec = importlib.util.spec_from_file_location("deck_build_data", ROOT / "slides" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(), module.js_round


_DECK, js_round = None, None


def _load():
    global _DECK, js_round
    if _DECK is None:
        _DECK, js_round = deck()
    return _DECK


def _round(x: float) -> int:
    _load()
    return js_round(x)


def to_fixed(x: float, digits: int) -> str:
    """JS Number.toFixed: the exact binary value, ties away from zero (Python's :.Nf rounds ties to even)."""
    return f"{Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP):.{digits}f}"


def group(x: float, digits: int = 0) -> str:
    """toLocaleString('en-US', fixed digits): Intl rounds half-expand on the shortest decimal, i.e. repr(x)."""
    d = Decimal(repr(float(abs(x)))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{d:,.{digits}f}"


def _int(x): return (MINUS if _round(x) < 0 else "") + group(_round(x))
def _signed(x): return ("+" if _round(x) > 0 else MINUS if _round(x) < 0 else "") + group(_round(x))
def _auc(x, d=3): return re.sub(r"^0", "", to_fixed(x, d))


def _str(x) -> str:
    """JS String(x) for what the data holds: numbers without a trailing .0, booleans lower-case."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


FORMATS: dict[str, Callable] = {
    "int": _int,
    "signed": _signed,
    "ci": lambda a: f"{_int(a[0])} to {_int(a[1])}",
    "range": lambda a: f"{_int(a[0])}\u2013{_int(a[1])}",
    "auc": _auc,
    "auc4": lambda x: _auc(x, 4),
    "aucrange": lambda a: f"{_auc(a[0])}\u2013{_auc(a[1])}",
    "num": _str,
    "abs": lambda x: group(_round(x)),
    "pct": lambda x: f"{_round(x * 100)}%",
    "d2": lambda x: (MINUS if to_fixed(x, 2).startswith("-") else "") + to_fixed(abs(x), 2),
    "usd2": lambda x: group(x, 2),
    "dist": lambda d: "distinguishable" if d["distinguishable"] else "not distinguishable",
}

KEY = re.compile(r'\bv\(\s*"([^"]+)"\s*\)')


def keys(typ: str) -> list[str]:
    """Every distinct v("path|fmt") key in the Typst source, in order of first use."""
    return list(dict.fromkeys(KEY.findall(typ)))


def resolve(data: dict, key: str) -> str:
    path, _, fmt = key.partition("|")
    node = data
    try:
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (KeyError, IndexError, ValueError, TypeError):
        raise KeyError(key) from None
    if node is None:
        raise KeyError(f"{key} (value is None)")
    if fmt and fmt not in FORMATS:
        raise KeyError(f"{key} (no format {fmt})")
    return FORMATS[fmt](node) if fmt else _str(node)


def render() -> str:
    typ = (ROOT / "paper" / "main.typ").read_text(encoding="utf-8")
    data = _load()
    out = {k: resolve(data, k) for k in sorted(keys(typ))}
    return json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    (ROOT / "paper" / "data.json").write_text(render(), encoding="utf-8", newline="\n")
