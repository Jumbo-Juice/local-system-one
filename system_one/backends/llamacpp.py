"""llama.cpp backend: a local llama-server process serving a GGUF model.

For models that do not fit the GPU. A mixture-of-experts model such as Qwen3-30B-A3B keeps its
expert weights in system RAM (``cpu_moe``: llama-server --cpu-moe / --n-cpu-moe) and the rest on
the GPU; for large prompt batches llama.cpp streams the expert weights to the GPU. The server runs
on 127.0.0.1 only; no remote API is called.

How the engine's contract maps onto the server:
- Tokenizer and chat template come from the model's Hugging Face tokenizer (``tokenizer``), as in
  the HF backend, and prompts are sent as token ids, so the server never re-tokenises. At start-up
  the server's own tokenizer is compared with it on a sample prompt.
- One /completion request per chunk of the batch, holding every prompt of the chunk, with
  n_predict=1 and n_probs=N, greedy. The server runs the prompts in parallel slots, batched
  together, and returns per prompt the N most likely next tokens with their log-softmax over the
  full vocabulary. A candidate outside the top N gets the N-th value, an upper bound; such reads
  are counted in ``info()["clipped"]``.
- Each slot keeps its last prompt's keys/values and reuses the common prefix (cache_prompt).
  llama.cpp notes that this can change logits slightly (batch sizes differ), so repeated runs are
  not guaranteed to be bit-identical.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

from .base import Backend, NextTokenScores
from .hf import render_chat_template


def resolve_gguf(model: str) -> str:
    """A local .gguf path, or "<hf repo>:<file>" fetched into the Hugging Face cache."""
    if os.path.exists(model):
        return model
    repo, sep, name = model.partition(":")
    if not sep:
        raise FileNotFoundError(f"{model!r} is neither a file nor '<hf repo>:<file.gguf>'")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend(Backend):
    kind = "llamacpp"

    def __init__(
        self,
        model: str,
        tokenizer: str,
        server: str | None = None,
        url: str | None = None,
        ctx_size: int = 16384,
        parallel: int = 8,
        n_gpu_layers: int = 99,
        cpu_moe: bool | int = False,
        batch_size: int = 4096,
        ubatch_size: int = 4096,
        flash_attn: str = "on",
        n_probs: int = 100,
        threads: int | None = None,
        extra_args: list[str] | None = None,
        startup_timeout: float = 900.0,
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model_name = tokenizer
        self.n_probs = n_probs
        self.max_batch = parallel
        self.options = {"ctx_size": ctx_size, "parallel": parallel, "n_gpu_layers": n_gpu_layers, "cpu_moe": cpu_moe,
                        "batch_size": batch_size, "ubatch_size": ubatch_size, "flash_attn": flash_attn,
                        "threads": threads, "extra_args": list(extra_args or [])}
        self.clipped = self.reads = 0
        self.proc: subprocess.Popen | None = None
        self.log_path: str | None = None
        if url:
            self.url, self.gguf = url.rstrip("/"), model
        else:
            self.gguf = resolve_gguf(model)
            self.url = self._start(server, startup_timeout)
        self.props = self._get("/props")
        self._check_tokenizer()

    # -- server process ------------------------------------------------------------

    def _start(self, server: str | None, timeout: float) -> str:
        exe = server or os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
        if not exe or not (os.path.exists(exe) or shutil.which(exe)):
            raise FileNotFoundError("llama-server not found: install llama.cpp and put llama-server on PATH, "
                                    "or set LLAMA_SERVER or [backend] server")
        o, port = self.options, _free_port()
        args = [exe, "-m", self.gguf, "--host", "127.0.0.1", "--port", str(port), "-c", str(o["ctx_size"]),
                "-np", str(o["parallel"]), "-ngl", str(o["n_gpu_layers"]), "-b", str(o["batch_size"]),
                "-ub", str(o["ubatch_size"]), "-fa", o["flash_attn"], "--no-webui"]
        if o["cpu_moe"] is True:
            args.append("--cpu-moe")
        elif o["cpu_moe"]:
            args += ["--n-cpu-moe", str(int(o["cpu_moe"]))]
        if o["threads"]:
            args += ["-t", str(o["threads"])]
        args += o["extra_args"]
        fd, self.log_path = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
        log = os.fdopen(fd, "w")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        url, t0 = f"http://127.0.0.1:{port}", time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; log: {self.log_path}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return url
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1.0)
        self.close()
        raise TimeoutError(f"llama-server did not become ready in {timeout:.0f} s; log: {self.log_path}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            return json.loads(r.read())

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def _check_tokenizer(self) -> None:
        sample = self.render_chat("You are a test.", 'State:\nYou are at (3,4). Options:\nA: move north\nB: stay') + '{"choice": "'
        theirs = self._post("/tokenize", {"content": sample, "add_special": False, "parse_special": True})["tokens"]
        if theirs != self.encode(sample):
            raise RuntimeError("the GGUF's tokenizer differs from the Hugging Face tokenizer on a sample prompt")

    # -- tokenizer -----------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)

    def render_chat(self, system: str, user: str) -> str:
        return render_chat_template(self.tokenizer, system, user)

    # -- forward passes ------------------------------------------------------------

    def _top(self, batch: list[list[int]], n_probs: int) -> list[dict[int, float]]:
        """Per prompt: {token id: log-prob} of the ``n_probs`` most likely next tokens."""
        out: list[dict[int, float]] = []
        for a in range(0, len(batch), self.max_batch):
            chunk = batch[a:a + self.max_batch]
            res = self._post("/completion", {
                "prompt": chunk if len(chunk) > 1 else chunk[0], "n_predict": 1, "n_probs": n_probs,
                "temperature": -1, "cache_prompt": True, "post_sampling_probs": False})
            res = res if isinstance(res, list) else [res]
            res = sorted(res, key=lambda r: r.get("index", 0))
            if len(res) != len(chunk):
                raise RuntimeError(f"llama-server returned {len(res)} results for {len(chunk)} prompts")
            for r in res:
                top = r["completion_probabilities"][0]["top_logprobs"]
                out.append({int(t["id"]): float(t["logprob"]) for t in top})
        return out

    def next_token_scores(self, batch, candidates, prefix_lens=None):
        out = []
        for top, cand in zip(self._top(batch, self.n_probs), candidates):
            floor = min(top.values())
            self.reads += len(cand)
            self.clipped += sum(c not in top for c in cand)
            best = max(top, key=top.get)
            out.append(NextTokenScores(np.array([top.get(c, floor) for c in cand], dtype=np.float64), best, top[best]))
        return out

    def next_token_logits(self, batch: list[list[int]]) -> np.ndarray:
        """Log-probs, not raw logits (they differ by a constant per row, which no decision depends on).
        Tokens outside the top ``n_probs`` get the lowest value returned."""
        rows = np.zeros((len(batch), self.vocab_size), dtype=np.float32)
        for i, top in enumerate(self._top(batch, self.n_probs)):
            rows[i, :] = min(top.values())
            for t, lp in top.items():
                if t < self.vocab_size:
                    rows[i, t] = lp
        return rows

    def info(self) -> dict:
        gpu = None
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            pass
        name = os.path.basename(self.gguf)
        quant = next((q for q in ("Q2_K", "Q3_K_M", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0", "BF16", "F16")
                      if q in name), "unknown")
        return {
            "kind": self.kind, "model": self.model_name, "gguf": name, "quantisation": f"GGUF {quant}",
            "dtype": f"GGUF {quant}", "device": "cuda+cpu" if self.options["cpu_moe"] else "cuda", "device_name": gpu,
            "server_build": self.props.get("build_info"), "n_ctx_per_slot": self.props.get("default_generation_settings", {}).get("n_ctx"),
            "n_probs": self.n_probs, "max_batch": self.max_batch, "reads": self.reads, "clipped": self.clipped,
            **{k: v for k, v in self.options.items() if k != "extra_args"}, "extra_args": self.options["extra_args"],
        }
