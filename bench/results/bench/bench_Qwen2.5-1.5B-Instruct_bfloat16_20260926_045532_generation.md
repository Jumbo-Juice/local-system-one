# Benchmark: Qwen/Qwen2.5-1.5B-Instruct (bfloat16) on Intel(R) Arc(TM) 140V GPU (8GB)

Host: Intel(R) Core(TM) Ultra 7 256V, 15.6 GB RAM, Windows-11-10.0.26200-SP0. Runtime: torch 2.14.0+xpu, transformers 5.17.0. Quantisation: none (weights in bfloat16). Started 2026-09-26 04:54:52.

Per-decision latency = call time / batch size. Median over repeats; range = min–max.

| prompt tokens (mean) | batch | mode | call median ms | per-decision ms | range | decisions/s |
|---:|---:|---|---:|---:|---|---:|

Text-generation baseline (greedy, no prefill, max 16 new tokens). style label: the model writes the label and stops; style json: it writes {"choice": "X"}.

| style | batch | per-decision ms | range | new tokens | parse failures | agreement with single-token |
|---|---:|---:|---|---:|---:|---:|
| label | 1 | 149.1 | 148.9–149.3 | 2.0 | 0/16 | 50% |
| label | 8 | 82.5 | 82.1–83.0 | 2.0 | 0/16 | 56% |
| json | 1 | 368.1 | 367.5–368.6 | 7.0 | 0/16 | 88% |
| json | 8 | 113.4 | 113.4–113.4 | 7.0 | 0/16 | 88% |
