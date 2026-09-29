"""The Master Viewer: watch the runs of every demo in the browser.

    python -m viewer                                    # start page: load a trace, a random run, or auto-demo
    python -m viewer runs/shooter/<run>.jsonl           # open one run
    python -m viewer --host 0.0.0.0                     # reachable from other devices on the network
    python -m viewer --bundle runs/shooter/<run>.jsonl  # one self-contained page with that run, to share

The page is viewer/index.html; each game demo draws itself from viewer/games/<game>.js. This server
serves that folder, the run pools runs/<game>/*.jsonl and /api/runs (every pooled run, newest first).
It only reads files; nothing here runs a model.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from demo.runs import RUNS

HERE = Path(__file__).resolve().parent
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".jsonl": "application/x-ndjson; charset=utf-8", ".json": "application/json; charset=utf-8"}
_SCRIPT = re.compile(r'<script src="([^"]+)"></script>')
_SLOT = re.compile(r'<script id="trace" type="application/x-ndjson"></script>')
_STAMP = re.compile(r"^\d{8}-\d{6}_")


# ---------------------------------------------------------------- the pool listing

def read_ends(path: Path) -> tuple[dict, dict | None]:
    """A trace's header (first line) and end record (last line, None if the run did not finish),
    without reading the ticks in between."""
    with path.open("rb") as f:
        header = json.loads(f.readline())
        size = f.seek(0, 2)
        f.seek(max(0, size - 262_144))
        tail = f.read().splitlines()
    for line in reversed(tail):
        if line.strip():
            try:
                rec = json.loads(line)
            except ValueError:  # the window started mid-line: no end record that small
                return header, None
            return header, rec if rec.get("type") == "end" else None
    return header, None


def describe(path: Path) -> dict:
    header, end = read_ends(path)
    pinned = path.name.endswith(".pinned.jsonl")
    stem = path.name[: -len(".pinned.jsonl")] if pinned else path.stem
    backend = header.get("backend") or {}
    return {
        "path": f"runs/{path.parent.name}/{path.name}",
        "game": path.parent.name,
        "scenario": header.get("scenario"),
        "label": _STAMP.sub("", stem),
        "created": header.get("created"),
        "model": str(backend.get("model") or backend.get("kind") or "").split("/")[-1],
        "seed": (header.get("map") or {}).get("seed"),
        "pinned": pinned,
        "outcome": end.get("outcome") if end else None,
        "tick": end.get("tick") if end else None,
    }


class Pool:
    """Every run under runs/<game>/, newest first. Reads each file's header and end once, then
    again only if the file changes (an eval writes up to ~90 traces)."""

    def __init__(self, root: Path = RUNS):
        self.root = root
        self._seen: dict[Path, tuple[int, int, dict]] = {}
        self._lock = threading.Lock()

    def listing(self) -> dict:
        with self._lock:
            pools = sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []
            runs = []
            for game in pools:
                for path in (self.root / game).glob("*.jsonl"):
                    st = path.stat()
                    cached = self._seen.get(path)
                    if not cached or cached[:2] != (st.st_mtime_ns, st.st_size):
                        try:
                            cached = (st.st_mtime_ns, st.st_size, describe(path))
                        except (OSError, ValueError) as err:
                            print(f"skipping {path}: {err}")
                            continue
                        self._seen[path] = cached
                    runs.append(cached[2])
            runs.sort(key=lambda r: (r["created"] or "", r["path"]), reverse=True)
            return {"pools": pools, "runs": runs}


# ---------------------------------------------------------------- the server

def make_handler(pool: Pool, opened: dict[str, Path], verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            route = unquote(urlsplit(self.path).path).lstrip("/")
            if route == "api/runs":
                return self.send(json.dumps(pool.listing(), ensure_ascii=False).encode("utf-8"), TYPES[".json"])
            if route in opened:
                return self.send_file(opened[route])
            if route.startswith("runs/"):
                return self.send_file(RUNS / route[len("runs/"):], root=RUNS, suffixes={".jsonl"})
            return self.send_file(HERE / (route or "index.html"), root=HERE, suffixes={".html", ".js", ".css"})

        def send_file(self, path: Path, root: Path | None = None, suffixes: set[str] | None = None) -> None:
            path = path.resolve()
            if (root and not path.is_relative_to(root)) or (suffixes and path.suffix not in suffixes) or not path.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send(path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

        def send(self, body: bytes, ctype: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            if verbose:
                super().log_message(fmt, *args)

    return Handler


def serve(host: str, port: int, trace: Path | None, open_browser: bool, verbose: bool) -> None:
    opened: dict[str, Path] = {}
    target = ""
    if trace:
        trace = trace.resolve()
        if trace.parent.parent == RUNS.resolve():
            target = f"runs/{trace.parent.name}/{trace.name}"
        else:  # a trace outside the pools: served under its own name
            target = f"opened/{trace.name}"
            opened[target] = trace
    for p in range(port, port + 20):  # the next free port if this one is taken (another viewer running)
        try:
            server = ThreadingHTTPServer((host, p), make_handler(Pool(), opened, verbose))
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    shown = "localhost" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{shown}:{server.server_port}/" + (f"#run={quote(target)}" if target else "")
    print(f"Master Viewer on {url}  (Ctrl+C to stop)")
    if host in ("0.0.0.0", "::"):
        print(f"other devices on the network: http://<this machine's address>:{server.server_port}/")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------- one self-contained page

def bundle(trace: Path, out: Path) -> Path:
    """index.html with every script inlined and the trace embedded, so the page opens from disk with
    no server. '<' only occurs inside JSON strings, so escaping it as \\u003c keeps the JSON identical
    and stops '</script>' in a prompt from ending the tag."""
    page = (HERE / "index.html").read_text(encoding="utf-8")

    def inline(m: re.Match) -> str:
        code = (HERE / m.group(1)).read_text(encoding="utf-8")
        if "</script" in code.lower():
            raise ValueError(f"{m.group(1)} contains '</script' and cannot be inlined")
        return f"<script>\n{code}</script>"

    page = _SCRIPT.sub(inline, page)
    body = trace.read_text(encoding="utf-8").replace("<", "\\u003c")
    slot = f'<script id="trace" type="application/x-ndjson" data-name="{html.escape(trace.name)}">{body}</script>'
    page, n = _SLOT.subn(lambda _: slot, page, count=1)
    if n != 1:
        raise ValueError("index.html has no empty trace slot")
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m viewer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", type=Path, help="a trace.jsonl to open (default: the start page)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other devices (default: this machine only)")
    ap.add_argument("--port", type=int, default=8765, help="first port to try (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    ap.add_argument("--bundle", type=Path, metavar="TRACE", help="write one self-contained page with this run and exit")
    ap.add_argument("-o", "--out", type=Path, help="with --bundle: the page to write (default: the trace's name, .html)")
    args = ap.parse_args()
    if args.bundle:
        out = args.out or args.bundle.with_name(args.bundle.name.removesuffix(".jsonl") + ".html")
        print("wrote", bundle(args.bundle, out))
        return
    if args.trace and not args.trace.is_file():
        ap.error(f"no such file: {args.trace}")
    serve(args.host, args.port, args.trace, not args.no_browser, args.verbose)
