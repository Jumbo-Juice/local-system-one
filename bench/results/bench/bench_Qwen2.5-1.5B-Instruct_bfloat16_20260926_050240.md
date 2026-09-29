# Benchmark: Qwen/Qwen2.5-1.5B-Instruct (bfloat16) on Intel(R) Arc(TM) 140V GPU (8GB)

Host: Intel(R) Core(TM) Ultra 7 256V, 15.6 GB RAM, Windows-11-10.0.26200-SP0. Runtime: torch 2.14.0+xpu, transformers 5.17.0. Quantisation: none (weights in bfloat16). Started 2026-09-26 04:56:35.

Per-decision latency = call time / batch size. Median over repeats; range = min–max.

| prompt tokens (mean) | batch | mode | call median ms | per-decision ms | range | decisions/s |
|---:|---:|---|---:|---:|---|---:|
| 160 | 1 | batched | 66.1 | 66.1 | 65.0–97.9 | 15.1 |
| 160 | 1 | sequential | 64.2 | 64.2 | 63.4–64.7 | 15.6 |
| 170 | 2 | batched | 105.8 | 52.9 | 52.5–53.5 | 18.9 |
| 170 | 2 | sequential | 132.2 | 66.1 | 65.5–67.9 | 15.1 |
| 166 | 4 | batched | 190.3 | 47.6 | 47.3–48.2 | 21.0 |
| 166 | 4 | sequential | 255.7 | 63.9 | 63.8–68.1 | 15.6 |
| 165 | 8 | batched | 318.6 | 39.8 | 39.8–43.2 | 25.1 |
| 165 | 8 | sequential | 519.7 | 65.0 | 64.7–65.4 | 15.4 |
| 164 | 16 | batched | 699.2 | 43.7 | 42.7–45.7 | 22.9 |
| 164 | 16 | sequential | 1028.9 | 64.3 | 64.2–64.5 | 15.6 |
| 165 | 32 | batched | 1162.0 | 36.3 | 36.3–36.7 | 27.5 |
| 165 | 32 | sequential | 2088.0 | 65.2 | 65.0–65.7 | 15.3 |
| 350 | 1 | batched | 99.8 | 99.8 | 99.4–100.1 | 10.0 |
| 350 | 1 | sequential | 99.4 | 99.4 | 99.2–99.9 | 10.1 |
| 352 | 2 | batched | 194.2 | 97.1 | 97.0–97.2 | 10.3 |
| 352 | 2 | sequential | 200.0 | 100.0 | 99.9–100.5 | 10.0 |
| 353 | 4 | batched | 324.4 | 81.1 | 81.0–85.1 | 12.3 |
| 353 | 4 | sequential | 401.1 | 100.3 | 99.9–100.4 | 10.0 |
| 354 | 8 | batched | 706.5 | 88.3 | 88.2–88.7 | 11.3 |
| 354 | 8 | sequential | 811.8 | 101.5 | 101.1–101.6 | 9.9 |
| 356 | 16 | batched | 1244.1 | 77.8 | 77.2–79.9 | 12.9 |
| 356 | 16 | sequential | 1629.0 | 101.8 | 101.5–102.4 | 9.8 |
| 356 | 32 | batched | 2462.4 | 77.0 | 75.5–77.5 | 13.0 |
| 356 | 32 | sequential | 3231.0 | 101.0 | 100.9–101.6 | 9.9 |
| 1125 | 1 | batched | 257.1 | 257.1 | 256.0–257.6 | 3.9 |
| 1125 | 1 | sequential | 256.8 | 256.8 | 255.8–257.0 | 3.9 |
| 1122 | 2 | batched | 533.3 | 266.7 | 266.1–268.5 | 3.8 |
| 1122 | 2 | sequential | 519.9 | 260.0 | 256.7–260.6 | 3.8 |
| 1122 | 4 | batched | 1063.1 | 265.8 | 265.6–266.9 | 3.8 |
| 1122 | 4 | sequential | 1053.5 | 263.4 | 261.4–269.0 | 3.8 |
| 1122 | 8 | batched | 2161.3 | 270.2 | 266.1–274.2 | 3.7 |
| 1122 | 8 | sequential | 2176.6 | 272.1 | 270.5–275.3 | 3.7 |
| 1122 | 16 | batched | 4438.4 | 277.4 | 276.9–281.5 | 3.6 |
| 1122 | 16 | sequential | 4290.3 | 268.1 | 265.4–269.1 | 3.7 |
| 1123 | 32 | batched | 9531.9 | 297.9 | 290.5–302.5 | 3.4 |
| 1123 | 32 | sequential | 8567.7 | 267.7 | 263.3–269.7 | 3.7 |

Text-generation baseline (greedy, no prefill, max 16 new tokens). style label: the model writes the label and stops; style json: it writes {"choice": "X"}.

| style | batch | per-decision ms | range | new tokens | parse failures | agreement with single-token |
|---|---:|---:|---|---:|---:|---:|
| label | 1 | 147.2 | 146.5–147.9 | 2.0 | 0/16 | 50% |
| label | 8 | 85.2 | 84.0–86.3 | 2.0 | 0/16 | 56% |
| json | 1 | 358.0 | 356.6–359.5 | 7.0 | 0/16 | 88% |
| json | 8 | 127.4 | 123.5–131.3 | 7.0 | 0/16 | 88% |
