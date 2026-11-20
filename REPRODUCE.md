# Reproducing the paper

*Can a Small Fine-Tuned LLM Make Fraud Decisions? A Locked, Cost-Based Evaluation*: the paper is in `paper/`, and the
experiments are in `fraud/`.

This page covers:
- what is in the repository and what is not;
- how to get the data;
- the environment the results were produced in;
- how to rebuild the paper, with or without the data.

Every command, with its flags, is in `RUN-GUIDE.md` → Fraud app and → Paper.

## What is here, and what is not

| here | not here |
|---|---|
| All code (`fraud/`, `system_one/`, `paper/`) | **The datasets.** They are not redistributed; download them yourself (below). |
| Every results JSON the paper reads (`fraud/results/`) | **Trained LoRA adapters** (`fraud/models/`, git-ignored). They are not published; retraining recreates them. |
| Pre-registrations: `fraud/prereg/`, each eval script's `PRE-REGISTRATION` docstring, and git history whose commit dates order every registration before its results | **Base model weights.** Download them from Hugging Face at the pinned revisions (below). |
| The revision's contract and log: `REVISION_PROMPT.md`, `REVISION_LOG.md` | **Per-payment run traces** (`fraud/runs/`, git-ignored, except a few `*.pinned.jsonl`) |
| The compiled paper: `paper/exports/transforming-llms-fraud.pdf` | |

The registrations work like this. Each pre-registration is a commit whose date precedes the results files it governs.
- `git log --follow <eval script>` and `git log -- fraud/results/<script>/` show the order.
- `REVISION_LOG.md` gives the commit for every registration and result of the revision.

## Data

### IEEE-CIS (the paper's evaluation)

The data is the Kaggle competition *IEEE-CIS Fraud Detection* (IEEE Computational Intelligence Society and Vesta Corporation, 2019).

1. Sign in to Kaggle, open https://www.kaggle.com/c/ieee-fraud-detection, and accept the competition rules.
2. Download the data, either through the page or with the Kaggle CLI (`pip install kaggle`, with an API token in `~/.kaggle/kaggle.json`):
   ```bash
   kaggle competitions download -c ieee-fraud-detection -p fraud/data/ieee-cis
   ```
3. Unzip it so that `fraud/data/ieee-cis/train_transaction.csv` and `fraud/data/ieee-cis/train_identity.csv` exist.
   Only the two `train_*` files are used: the competition's test files have no labels.
4. Build the caches, a few minutes each:
   ```bash
   python -m fraud.ieee_data
   python -m fraud.history
   ```

`fraud/data/` is git-ignored. Never commit rows from it.

### PaySim (pipeline development only)

Download the Kaggle dataset *Synthetic Financial Datasets For Fraud Detection* (https://www.kaggle.com/datasets/ealaxi/paysim1).
Put its CSV in `fraud/data/raw/`; the original `PS_*.csv` name is accepted. Then build the cache:

```bash
python -m fraud.data
```

### Models

The models come from Hugging Face. `fraud/d1torch.py` checks each file's sha256 against its pinned copy.
- **IEEE-CIS:** `LiquidAI/d1-Omni-2B`, revision `431f1d667de5afcc9b0c58b2d9c170a7b38b358b`, under the LFM Open License v1.0.
  It is read from `fraud/models/d1-omni-hf/`.
- **PaySim:** `Qwen/Qwen2.5-1.5B-Instruct`, under Apache-2.0.

## Environment

The results were produced on one machine:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB |
| NVIDIA driver | 616.92 (CUDA user-mode driver 13.4) |
| PyTorch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| Python | 3.13.15 |
| OS | Windows 11 (10.0.26200) |
| Key libraries | transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, numpy 2.5.4 |
| Paper | Typst 0.15.1, with the vendored `charged-ieee` template |

Install the environment in this order:
1. Install the PyTorch build for your hardware first (see `requirements.txt`).
2. Then install either `pip install -r requirements.txt`, or the exact versions in `requirements-lock.txt`.

Earlier PaySim phases ran on a laptop's integrated Intel GPU (see `fraud/PROGRESS.md`). Every IEEE-CIS number in the paper
comes from the machine above.

Scores from a GPU can differ in the last bits between machines, drivers and library versions. Each eval records:
- its backend;
- the hashes of its adapters and training rows;
- for LM scoring, a check of the batched scores against lone prompts.

## Rebuilding the paper

### Without the data

These read only the committed results, so they need no data and no network:

```bash
python paper/build_data.py
typst compile paper/main.typ paper/exports/transforming-llms-fraud.pdf
python -m pytest tests/test_paper.py
```

### With the data

The steps rerun the analyses that read only saved outputs and dev scores, in this order:
1. Rebuild the IEEE-CIS caches, as above.
2. Retrain the adapters and score dev, following `RUN-GUIDE.md` → Fraud app (Phases 11–25), then the revision's commands.
3. Run the revision's analyses, each with the arguments in `RUN-GUIDE.md`:
   - `fraud.eval_order_bias` (`--score`, then `--report`);
   - `fraud.eval_parity`;
   - `fraud.eval_noc` (`--score`, then the comparison);
   - `fraud.eval_customer_line`;
   - `fraud.eval_readouts`.

**The final test windows are spent. Do not run `fraud.eval_final` or `fraud.eval_gbdt_readout` again.**
- Their "run once" guard only sees traces in your local `fraud/runs/`, so on a fresh clone nothing stops them.
- Running them would score the final windows a second time. That is a second look, not a replication.
- Their committed results are the record. `fraud.eval_readouts` reproduces every final-test analysis from the saved
  traces, which are not published.
