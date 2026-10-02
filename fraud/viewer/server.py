"""The fraud analyst console: replay recorded fraud runs and read the eval in the browser.

    python -m fraud.viewer                                  # start page: runs, auto-demo, eval
    python -m fraud.viewer fraud/runs/<run>.jsonl           # open one run
    python -m fraud.viewer --port 8766 --no-browser

The page is fraud/viewer/index.html with console.js and console.css. This server serves those
files, the traces fraud/runs/*.jsonl, the eval results fraud/results/eval/*.json, and two listings
(/api/runs, /api/evals). It only reads files: nothing here runs a model. It binds to 127.0.0.1
unless --host says otherwise. It shares no code with the game viewer (fraud/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from ..runs import RUNS

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "eval"
PAGES = ("index.html", "console.js", "console.css")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".jsonl": "application/x-ndjson; charset=utf-8", ".json": "application/json; charset=utf-8"}
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_STAMP = re.compile(r"^\d{8}-\d{6}_")
_EVAL = re.compile(r"^(\d{8}-\d{6})_eval_(.+)_(test\d+|dev\d+)$")


def read_ends(path: Path) -> tuple[dict, dict | None]:
    """A trace's header (first line) and end record (None if the run did not finish)."""
    with path.open("rb") as f:
        header = json.loads(f.readline())
        size = f.seek(0, 2)
        f.seek(max(0, size - 65_536))
        tail = f.read().splitlines()
    for line in reversed(tail):
        if line.strip():
            try:
                rec = json.loads(line)
            except ValueError:
                return header, None
            return header, rec if rec.get("type") == "end" else None
    return header, None


def describe(path: Path) -> dict:
    header, end = read_ends(path)
    pinned = path.name.endswith(".pinned.jsonl")
    stem = path.name[: -len(".pinned.jsonl")] if pinned else path.stem
    m = _EVAL.match(stem)
    backend = header.get("backend") or {}
    summ = (end or {}).get("summary") or {}
    return {
        "file": path.name,
        "label": _STAMP.sub("", stem),
        "eval": m.group(1) if m else header.get("eval"),
        "setup": header.get("setup"),
        "window": (header.get("window") or {}).get("id"),
        "split": (header.get("window") or {}).get("split"),
        "created": header.get("created"),
        "model": str(backend.get("model") or backend.get("kind") or "").split("/")[-1],
        "pinned": pinned,
        "finished": end is not None,
        "cost": summ.get("cost"),
        "recall": summ.get("recall"),
        "total_ms": summ.get("total_ms"),
        "model_decisions": summ.get("model_decisions"),
    }


class Listing:
    """The runs and evals on disk, re-read only for files that changed."""

    def __init__(self, runs: Path = RUNS, results: Path = RESULTS):
        self.runs, self.results = runs, results
        self._seen: dict[Path, tuple[int, int, dict]] = {}
        self._lock = threading.Lock()

    def _cached(self, path: Path, read) -> dict | None:
        st = path.stat()
        hit = self._seen.get(path)
        if not hit or hit[:2] != (st.st_mtime_ns, st.st_size):
            try:
                hit = (st.st_mtime_ns, st.st_size, read(path))
            except (OSError, ValueError) as err:
                print(f"skipping {path.name}: {err}")
                return None
            self._seen[path] = hit
        return hit[2]

    def run_list(self) -> dict:
        with self._lock:
            files = sorted(self.runs.glob("*.jsonl")) if self.runs.is_dir() else []
            runs = [r for r in (self._cached(p, describe) for p in files) if r]
        runs.sort(key=lambda r: (r["created"] or "", r["file"]), reverse=True)
        return {"runs": runs}

    def eval_list(self) -> dict:
        def read(path: Path) -> dict:
            r = json.loads(path.read_text(encoding="utf-8"))
            return {"id": r.get("eval"), "file": path.name, "created": r.get("created"), "complete": r.get("complete"),
                    "bars": r.get("bars")}
        with self._lock:
            files = sorted(self.results.glob("*.json")) if self.results.is_dir() else []
            evals = [r for r in (self._cached(p, read) for p in files) if r]
        evals.sort(key=lambda r: r["created"] or "", reverse=True)
        return {"evals": evals}


def resolve(route: str, runs: Path = RUNS, results: Path = RESULTS) -> Path | None:
    """The file behind a route, or None. Only the console files, runs/<name>.jsonl and
    results/eval/<name>.json are reachable; no subfolders, no '..'."""
    if route in ("", "index.html"):
        return HERE / "index.html"
    if route in PAGES:
        return HERE / route
    head, _, name = route.rpartition("/")
    if not _NAME.match(name) or ".." in name:
        return None
    if head == "runs" and name.endswith(".jsonl"):
        path = runs / name
    elif head == "results/eval" and name.endswith(".json"):
        path = results / name
    else:
        return None
    return path if path.is_file() else None


def make_handler(listing: Listing, verbose: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            route = unquote(urlsplit(self.path).path).lstrip("/")
            if route == "api/runs":
                return self.send(json.dumps(listing.run_list(), ensure_ascii=False).encode("utf-8"), TYPES[".json"])
            if route == "api/evals":
                return self.send(json.dumps(listing.eval_list(), ensure_ascii=False).encode("utf-8"), TYPES[".json"])
            path = resolve(route, listing.runs, listing.results)
            if path is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send(path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

        def send(self, body: bytes, ctype: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args) -> None:
            if verbose:
                super().log_message(fmt, *args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="a trace in fraud/runs/ to open")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args()
    page = ""
    if args.trace:
        path = Path(args.trace).resolve()
        if path.parent != RUNS or not path.is_file():
            ap.error(f"{args.trace} is not a trace in {RUNS}")
        page = f"#/run/{quote(path.name)}"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Listing(), args.verbose))
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_port}/{page}"
    print(f"fraud console: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
