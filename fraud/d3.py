"""Decision 3.0 (vLLM Semantic Router's d3-nano / d3-mini) as the fraud brain, through llama.cpp.

A D3 model is a native decision model: llama-server's ``/v1/systemone`` endpoint answers a typed
``choice`` question with a probability per option, without generating text (Documented,
llama.cpp tools/server/README.md). It does not fit the engine's next-token contract, so it gets its
own decider here instead of a ``system_one`` backend.

- The question is prompt v4 of ``brain`` unchanged: state = CONTEXT + describe(), instructions =
  QUESTION, criteria = OPTIONS (no descriptions).
- The server reads a vllm-d3 choice in one option order only, the order of the request
  (Documented, server-decision.cpp). ``orders="both"`` asks the forward and the reversed order as
  two questions of one request and averages them, like the engine's order debias.
- One slot (``-np 1``) and one request at a time: llama.cpp issue #30402 (open) reports that a D3
  answer changes when another request shares its batch.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import brain
from .brain import ACTIONS, OPTIONS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
# GGUF Q8_0, paths under fraud/models/, sha256 from the Hugging Face file metadata.
# d3-*: ggml-org/vllm-d3-{nano,mini}-GGUF (Phase 6). d1-omni-2b: LiquidAI/d1-Omni-2B-GGUF (Phase 8),
# text only (no mmproj), needs llama.cpp b11960 or newer (Documented, its model card).
MODELS = {
    "d3-nano": ("d3/vllm-d3-nano-Q8_0.gguf", "4867d306fd22aec8c5ebd1851b7f1074f480702655e8000ceedf0c47c6a6f337"),
    "d3-mini": ("d3/vllm-d3-mini-Q8_0.gguf", "f9f99be46b7ce1e5a5f09ed620de89106447646b34f7d2993e7563b48feb2858"),
    "d1-omni-2b": ("lfm/d1-Omni-2B-Q8_0.gguf", "aedf1a7c341b4281979423dbc39e9c3d3ad1da8834156c8e75af5399ca3b0861"),
}
ORDERS = ("both", "fwd", "rev")
WINGET_SERVER = (Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
                 / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe" / "llama-server.exe")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gguf_path(model: str) -> Path:
    return MODELS_DIR / MODELS[model][0]


def find_server() -> str:
    exe = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if exe:
        return exe
    if WINGET_SERVER.exists():
        return str(WINGET_SERVER)
    raise FileNotFoundError("llama-server not found: `winget install llama.cpp`, or set LLAMA_SERVER")


def request(t, s, i: int, orders: str) -> dict:
    """The /v1/systemone body for transaction i: prompt v4 as one choice question per order."""
    if orders not in ORDERS:
        raise ValueError(f"orders must be one of {ORDERS}")
    d = brain.decision(t, s, i)
    q = lambda opts: {"type": "choice", "instructions": d.instruction, "criteria": {o: None for o in opts}}
    questions = {}
    if orders in ("both", "fwd"):
        questions["fwd"] = q(OPTIONS)
    if orders in ("both", "rev"):
        questions["rev"] = q(OPTIONS[::-1])
    return {"state": f"{d.context}\n\n{d.state}", "questions": questions}


class D3Server:
    """A llama-server process serving one D3 GGUF on 127.0.0.1."""

    def __init__(self, model: str, check_sha: bool = True, startup_timeout: float = 600.0, server: str | None = None):
        self.model, self.gguf = model, gguf_path(model)
        if not self.gguf.exists():
            raise FileNotFoundError(f"{self.gguf} missing; download {MODELS[model][0]} (RUN-GUIDE.md)")
        if check_sha and sha256(self.gguf) != MODELS[model][1]:
            raise RuntimeError(f"{self.gguf.name}: sha256 differs from the pinned one")
        self.exe = server or find_server()
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        args = [self.exe, "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(port), "-np", "1",
                "-ngl", "99", "-c", "4096", "-b", "4096", "-ub", "4096", "--no-webui"]
        fd, self.log_path = tempfile.mkstemp(prefix="llama-server-d3-", suffix=".log")
        self.proc = subprocess.Popen(args, stdout=os.fdopen(fd, "w"), stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self.url, t0 = f"http://127.0.0.1:{port}", time.time()
        while time.time() - t0 < startup_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; log: {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1.0)
        else:
            self.close()
            raise TimeoutError(f"llama-server did not become ready; log: {self.log_path}")
        self.version = subprocess.run([self.exe, "--version"], capture_output=True, text=True).stderr.strip()

    def post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("LLAMA_API_KEY"):  # llama-server reads the same variable and then requires it
            headers["Authorization"] = "Bearer " + os.environ["LLAMA_API_KEY"]
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        return {"kind": "llamacpp-systemone", "model": self.model, "gguf": self.gguf.name,
                "sha256": MODELS[self.model][1], "server": self.version.splitlines()[0] if self.version else "",
                "server_args": "-np 1 -ngl 99 -c 4096 -b 4096 -ub 4096"}

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class D3Decider:
    """approve / review / decline from a D3 model. ``post`` sends one /v1/systemone body.

    Without thresholds the action is the argmax of the (order-averaged) probabilities; with them,
    the decline probability against ``t_review`` / ``t_decline`` (as brain.ThresholdDecider).
    """

    by = "model"

    def __init__(self, post, orders: str = "both", thresholds: tuple[float, float] | None = None):
        if orders not in ORDERS:
            raise ValueError(f"orders must be one of {ORDERS}")
        self.post, self.orders, self.thresholds = post, orders, thresholds

    def prompt(self, t, s, i: int) -> str:
        return json.dumps(request(t, s, i, self.orders), indent=1)

    def decide(self, t, s, i: int) -> dict:
        body = request(t, s, i, self.orders)
        t0 = time.perf_counter()
        r = self.post(body)
        ms = (time.perf_counter() - t0) * 1000
        orders = [[float(r["answers"][q]["probabilities"][o]) for o in OPTIONS] for q in body["questions"]]
        p = [sum(col) / len(orders) for col in zip(*orders)]
        if self.thresholds:
            tr, td = self.thresholds
            action = "decline" if p[2] >= td else "review" if p[2] >= tr else "approve"
        else:
            action = ACTIONS[max(range(3), key=lambda k: p[k])]
        out = {"by": self.by, "action": action, "fallback": "approve" if p[0] >= p[2] else "decline",
               "scores": dict(zip(ACTIONS, p)), "orders": [dict(zip(ACTIONS, o)) for o in orders],
               "prompt_tokens": int(r.get("usage", {}).get("input_tokens", 0)), "ms": ms}
        if self.thresholds:
            out["thresholds"] = {"review": self.thresholds[0], "decline": self.thresholds[1]}
        return out
