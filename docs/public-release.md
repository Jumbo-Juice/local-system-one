# Public release: scan and decisions

Revision Stage 5 (`REVISION_PROMPT.md`), 2026-11-20.

**What is prepared:**
- `REPRODUCE.md`: data download steps, the environment, and what is and is not included;
- `requirements-lock.txt`;
- the paper's footnote 1, which links the repository.

**The repository:** the only remote is `origin`, `https://github.com/Jumbo-Juice/local-system-one`. It is **already
public**. The GitHub API, read 2026-11-20, gives `"visibility": "public"`, 0 forks and 0 stars. `origin/main` (169
commits) is published there.

## What the scan found (tree, and history across all refs)

**Raw data.** No `.csv`, `.npz`, `.npy`, `.parquet`, `.pkl`, `.safetensors`, `.gguf`, `.bin`, `.pt`, `.zip`, `.gz` or
`.h5` file was ever committed. `fraud/data/`, `fraud/models/` and `fraud/runs/` are git-ignored. There are two exceptions:
- **One IEEE-CIS row:** `fraud/results/prompt_example.json` holds one fit row, anonymised (masked fields, no identifiers),
  which also appears in the paper's Appendix H. The revision requires "one filled example row". **Kept, by Juji's decision.**
- **One PaySim trace:** `fraud/runs/20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl`. PaySim is synthetic.

**Secrets.** None. The scan searched for API keys, tokens, passwords and private keys, and for `.env` and `kaggle.json`
files. `LLAMA_API_KEY` is only a variable name.

**Absolute local paths.**
- Found: Windows user folders in two bench results files (field `traces`), in `slides-handoff.md`, and in this file's
  first draft. History has the same strings, including the bench files at their old `bench/results/` location.
- Done: redacted in the tree to `<home>` or a relative name (commit after 28fbd03).

**Co-authors' student IDs.**
- Found in `slides/index.html` (title slide), `docs/superpowers/plans/2026-11-12-final-paper-and-deck.md`, and the
  pre-revision `paper/main.typ`, in history and on the public `origin/main`.
- Done: removed from the tree (names kept).

**Commit email.** All 194 commits on `paper-revision`, and 254 of the 257 commits across all refs, give the owner's
personal Gmail address. The refs are `main`, `paper-revision`, `origin/claude/remove-coauthored-by-3y241l` and the tags
`fraud-phase-0..5`.

## Juji's decision (2026-11-20): clean, then publish the whole repository

1. Remove the student IDs and local paths from history, and rewrite the commit email to the GitHub no-reply address. Use
   `git filter-repo` with `--replace-text` and `--mailmap`.
   - Commit dates are kept, so every registration still precedes its results.
   - Every commit hash changes. `docs/commit-map.txt` maps the old hashes to the new ones, because `REVISION_LOG.md` cites
     old hashes.
2. Publish to the same repository, so footnote 1 stays valid. That means force-pushing `main` (fast-forwarded to the
   revision) and the tags, and handling the stale remote branch.
3. Every history rewrite, force-push, branch or tag deletion, or visibility change is repeated to Juji for a final yes
   before it runs.

**Done (2026-11-20), after Juji's final yes to the exact commands:**
- **Backup:** the old history was saved as a mirror clone outside the repository.
- **Rewrite:** `git filter-repo` rewrote every local branch and tag. It was first run as a trial in a scratch copy and
  checked there:
  - no student ID, local path or Gmail address is left in any blob, message or author line;
  - every commit keeps its dates, subject and author name;
  - the tags map to the matching rewritten commits;
  - the newest tree is byte-identical to the one before the rewrite.
  The real run gave the same IDs as the trial. The commit count went from 259 to 258, because GitHub's two "Initial
  commit"s differed only in a signature and now coincide.
- **Map:** `docs/commit-map.txt`, old commit ID to new.
- **Publish:** force-pushed `main` (now the revision) and the tags `fraud-phase-0..5`, and deleted the stale branch
  `claude/remove-coauthored-by-3y241l` on GitHub. It was never merged.

**Checked after the push:**
- A fresh clone from GitHub has 258 commits and no student ID, local path or Gmail address anywhere.
- GitHub still serves old commits by their old IDs (for example `ca1e107`, HTTP 200 from the API).

**After publishing:**
- The old commits stay reachable on GitHub by their old hashes until GitHub garbage-collects them. GitHub Support can be
  asked to purge cached views.
- Anyone who cloned the public repository keeps the old history.
- The co-authors should be told that their IDs were public.
