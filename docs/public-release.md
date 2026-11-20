# Public release: scan and open decisions

Revision Stage 5 (`REVISION_PROMPT.md`), 2026-11-20. **Nothing has been pushed or published.**
- Prepared:
  - `REPRODUCE.md` (data download steps, the environment, what is and is not included);
  - `requirements-lock.txt` (the exact package versions of the environment that produced the results);
  - the paper's footnote 1, which now links the repository.
- The repository's only remote is `origin` → `https://github.com/Jumbo-Juice/local-system-one`. Footnote 1 links there.

## What the scan checked (tree at HEAD and all history)

| check | result |
|---|---|
| Raw data: any `.csv`, `.npz`, `.parquet`, `.pkl`, `.safetensors`, `.gguf`, `.bin` or `.pt` ever committed | **None**, in any commit. `fraud/data/`, `fraud/models/` and `fraud/runs/` are git-ignored. The largest blob in history is `docs/media/shooter.gif`, 2.2 MB. |
| Secrets: API keys, tokens, passwords, Hugging Face, OpenAI, GitHub and AWS key patterns | **None.** `LLAMA_API_KEY` is only a variable name; no value is committed. `id-token: write` in the Pages workflow is a permission, not a secret. |
| Absolute local paths | **In the current tree:** two bench results files (`bench/results/shooter_eval/shooter_eval_20260928_123222.json` and `shooter_eval_ammo_20260929_134450.json`, field `traces`) and `slides-handoff.md` line 5. They hold Windows user folder names (`<home>\...`, `<home>\...`). **In history:** 5 occurrences. The other hits are `%LOCALAPPDATA%` or `Path.home()` patterns, not local paths. |
| Personal data in files | **Co-authors' student IDs**, in `slides/index.html` (title slide) and `docs/superpowers/plans/2026-11-12-final-paper-and-deck.md`, and in history, including the pre-revision `paper/main.typ`. The revised paper has none. No email address appears in any file. |
| Personal data in git metadata | **Every one of the 193 commits** names the author `Juji` and gives a personal Gmail address in the author and committer fields. One commit names the author `Juji_Vong`. |

## Decisions that belong to the owner before publishing

1. **Co-authors' student IDs.** They are other people's personal data. Either remove them from the current tree (the title
   slide and the plan doc) and from history, or get the co-authors' consent to publish them.
2. **The commit email.**
   - Either publish as is, or rewrite history to a GitHub no-reply address, for example with `git filter-repo --mailmap`.
   - A rewrite keeps every commit date, so registrations still precede their results, but it changes every commit hash.
     `REVISION_LOG.md` cites hashes, so it would need a hash map. The same applies to removing the student IDs or local
     paths from history.
3. **Local paths:** redact the two bench `traces` fields and the handoff note, or leave them (they show only a Windows
   user folder name).
4. **What becomes public:**
   - Option A: this whole repository (games, story, fraud app, paper), which keeps the registration history intact.
   - Option B: a fraud-and-paper-only repository. It would need a history filter, which also changes hashes.
5. **Adapters stay unpublished,** as the revision decided. The d1-Omni-2B licence (LFM Open License v1.0) would allow
   redistributing derivatives with conditions. The per-payment traces of the final test are git-ignored and stay local. So
   `fraud.eval_readouts` can be re-run only where those traces exist; its committed JSON is the record.
