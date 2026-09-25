# Benchmark: Qwen/Qwen2.5-1.5B-Instruct (bfloat16) on Intel(R) Arc(TM) 140V GPU (8GB)

Host: Intel(R) Core(TM) Ultra 7 256V, 15.6 GB RAM, Windows-11-10.0.26200-SP0. Runtime: torch 2.14.0+xpu, transformers 5.17.0. Quantisation: none (weights in bfloat16). Started 2026-09-26 04:43:54.

Per-decision latency = call time / batch size. Median over repeats; range = min–max.

| prompt tokens (mean) | batch | mode | call median ms | per-decision ms | range | decisions/s |
|---:|---:|---|---:|---:|---|---:|
| 160 | 1 | batched | 69.6 | 69.6 | 69.2–69.7 | 14.4 |
| 160 | 1 | sequential | 70.5 | 70.5 | 68.5–72.7 | 14.2 |
| 170 | 2 | batched | 111.9 | 55.9 | 55.6–56.3 | 17.9 |
| 170 | 2 | sequential | 139.6 | 69.8 | 69.6–71.2 | 14.3 |
| 166 | 4 | batched | 223.9 | 56.0 | 55.9–56.1 | 17.9 |
| 166 | 4 | sequential | 288.5 | 72.1 | 72.1–73.2 | 13.9 |
| 165 | 8 | batched | 379.8 | 47.5 | 47.4–47.6 | 21.1 |
| 165 | 8 | sequential | 586.8 | 73.4 | 72.9–75.6 | 13.6 |
| 164 | 16 | batched | 794.1 | 49.6 | 48.6–51.4 | 20.1 |
| 164 | 16 | sequential | 1150.6 | 71.9 | 71.8–76.7 | 13.9 |
| 165 | 32 | batched | 1395.8 | 43.6 | 43.4–43.7 | 22.9 |
| 165 | 32 | sequential | 2323.1 | 72.6 | 72.5–75.4 | 13.8 |
| 350 | 1 | batched | 112.5 | 112.5 | 112.5–113.2 | 8.9 |
| 350 | 1 | sequential | 113.4 | 113.4 | 113.1–113.8 | 8.8 |
| 352 | 2 | batched | 227.4 | 113.7 | 112.8–113.9 | 8.8 |
| 352 | 2 | sequential | 228.4 | 114.2 | 113.4–121.6 | 8.8 |
| 353 | 4 | batched | 391.1 | 97.8 | 97.7–97.9 | 10.2 |
| 353 | 4 | sequential | 465.5 | 116.4 | 114.6–134.9 | 8.6 |
| 354 | 8 | batched | 803.3 | 100.4 | 99.9–100.6 | 10.0 |
| 354 | 8 | sequential | 926.2 | 115.8 | 115.1–116.5 | 8.6 |
| 356 | 16 | batched | 1451.2 | 90.7 | 89.9–92.1 | 11.0 |
| 356 | 16 | sequential | 1875.6 | 117.2 | 115.0–119.0 | 8.5 |
| 356 | 32 | batched | 2848.9 | 89.0 | 88.4–90.0 | 11.2 |
| 356 | 32 | sequential | 3668.3 | 114.6 | 114.2–116.3 | 8.7 |
| 1125 | 1 | batched | 335.1 | 335.1 | 334.4–351.7 | 3.0 |
| 1125 | 1 | sequential | 336.0 | 336.0 | 335.6–336.0 | 3.0 |
| 1122 | 2 | batched | 625.6 | 312.8 | 311.8–313.6 | 3.2 |
| 1122 | 2 | sequential | 672.5 | 336.2 | 334.8–336.6 | 3.0 |
| 1122 | 4 | batched | 1267.1 | 316.8 | 302.9–325.7 | 3.2 |
| 1122 | 4 | sequential | 1359.4 | 339.8 | 338.3–341.9 | 2.9 |
| 1122 | 8 | batched | 2506.0 | 313.2 | 310.5–325.4 | 3.2 |
| 1122 | 8 | sequential | 2678.6 | 334.8 | 334.3–335.3 | 3.0 |
| 1122 | 16 | batched | 5068.7 | 316.8 | 313.5–330.5 | 3.2 |
| 1122 | 16 | sequential | 5401.4 | 337.6 | 336.4–338.5 | 3.0 |
| 1123 | 32 | batched | 10820.4 | 338.1 | 328.7–339.0 | 3.0 |
| 1123 | 32 | sequential | 11037.3 | 344.9 | 342.8–346.6 | 2.9 |

Text-generation baseline (greedy, whole JSON answer, max 12 new tokens):

| batch | per-decision ms | range | new tokens | parse failures | agreement with single-token |
|---:|---:|---|---:|---:|---:|
| 1 | 161.8 | 158.2–165.5 | 2.0 | 16/16 | 0% |
| 8 | 97.3 | 96.2–98.4 | 2.0 | 16/16 | 0% |
