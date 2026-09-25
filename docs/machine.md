# Machine inspection (Lenovo) and runtime choice

Inspected on 2026-09-26. All values are **Observed** on this machine.

## Hardware and OS

| Item | Value |
|------|-------|
| Machine | Lenovo 83HM |
| OS | Windows 11 Home 10.0.26200 |
| CPU | Intel Core Ultra 7 256V (Lunar Lake), 8 cores / 8 threads |
| RAM | 15.6 GB (shared with the iGPU) |
| GPU | Intel Arc 140V iGPU (Xe2, 64 EUs). PyTorch reports 8097 MB device memory, Level-Zero driver 1.6.34938, display driver 32.0.101.8132 |
| NPU | Intel AI Boost (present, not used) |
| NVIDIA / CUDA | none (`nvidia-smi` and `nvcc` not found) |
| Free disk (C:) | ~361 GB |

## Python and inference software found

| Environment | Python | Inference libraries |
|-------------|--------|---------------------|
| `anaconda3` base | 3.10.14 | torch 1.12.1 (CPU), transformers 4.24.0 (too old: no XPU, no `logits_to_keep`) |
| python.org | 3.13.9 | none (pygame only) |
| uv-managed | 3.11.16 | none |
| LM Studio | `lms.exe` present | no models downloaded |

No model weights were on disk (`~/.lmstudio/models`, `~/.ollama/models` and the HF cache were
empty; a profile-wide search for `*.gguf` / `*.safetensors` found none).

## Runtime choice

**Implementation choice: PyTorch 2.14 (XPU build) + transformers 5.17 in a project `.venv`
(Python 3.13).**

Reasons:

- The hard requirement is full next-token logits for batched inputs. A plain `model(...)` forward
  pass returns the logits tensor directly. Verified by test (`tests/test_logits.py`): one forward
  pass returns a `[B, 151936]` float32 array. Left-padded rows match unpadded runs: max |Δlogit|
  ≈ 5e-5 in float32 and ≤ 0.33 in bf16 (precision noise; argmax always equal).
- PyTorch's XPU backend runs on the Arc 140V (`torch.xpu.is_available() == True`; fp16 matmul
  ≈ 17.6 TFLOPS in a 2048³ micro-test).
- The same code runs on CUDA or CPU by changing `device` in the config. That covers the NUC 12
  whatever its GPU is.
- Rejected: **LM Studio / chat servers.** Their OpenAI-style APIs return at most top-k logprobs
  for generated tokens. They do not return log-probs for arbitrary tokens across a batch of
  prompts. **llama-cpp-python**: batched multi-sequence logits need the low-level API, and GPU
  builds for Intel on Windows must be compiled. **OpenVINO**: a good Intel runtime, but it adds
  a model-export step; not needed for a proof of concept (a possible future backend).

## Model choice

See `docs/research.md` → Observed → "Model selection" for the evaluation that picked the model.
