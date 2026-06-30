# Source Code Analysis — gat-crf-vietmed-ner

## Context
The user asked for a written analysis of the current source code to serve as
shared context for future requests. This document maps the entire codebase so
later tasks (bug fixes, new experiments, refactors) can load context fast. No
code changes are proposed here — this is a reference map.

## What the project is
A master-coursework Vietnamese **medical NER** project. Core contribution: an
**improved PhoBERT + GAT + CRF** model benchmarked against PhoBERT/XLM-R/
ViHealthBERT baselines (Softmax/CRF heads), plus CRF-vs-Softmax ablation,
zero-shot cross-domain eval, and a 4-category error analysis. Designed to run
one experiment per session on **Google Colab T4 (free tier)**.

## Repo layout
- `VietMed_NER.ipynb` — **the entire implementation** (31 cells, §0–§10 + §T).
- `requirements.txt` — pinned deps (torch/PyG installed in-Colab).
- `docs/superpowers/specs/…-design.md` — design spec.
- `docs/superpowers/plans/…-ner.md` — 15-task implementation plan.
- `README.md` — overview + experiment matrix + run instructions.
- No `.py` modules, no `results/` yet (gitignored until produced).

## Notebook structure (cell index → content)
| § | cells | role |
|---|---|---|
| §0 Setup | 2 | pip pins, SEED=42, Drive mount, `DRIVE_DIR/RESULTS_DIR/CKPT_DIR`, `DEVICE` |
| §1 Config | 4 | `EXPERIMENTS` dict + `EXP_ID`/`cfg` selector |
| §2 Data | 6,7 | load `leduckhai/VietMed-NER`, derive `LABEL_LIST/label2id/id2label`; shared cross-domain schema + `project_tags` |
| §3 Preprocess | 9,10,11,12 | `realign_tags`, `segment_to_groups` (py_vncorenlp), `align_labels`, `build_encoded` |
| §4 Verify | 14 | `verify_roundtrip` GATE — must pass before training |
| §5 Model | 16 | `BertCrfForNer`, `build_model` factory |
| §7 Eval | 18 | `evaluate_tags` (seqeval), `decode_predictions` |
| §6 Train | 20 | `train_and_eval` loop + Drive ckpt + CSV |
| §8 Errors | 22 | `spans`, `classify_errors`, `error_counts` |
| §9 Cross-domain | 24,25 | `read_conll`, `crossdomain_eval` on PhoNER-COVID19 |
| §10 GAT | 27,28 | `GatCrfForNer`, `build_gat_crf` (PyG `GATConv` + MHA + CRF) |
| §T Tests | 30 | CPU-only unit tests of the 4 pure functions |

## Key functions & responsibilities (with cell numbers)
- **`build_encoded(split, cfg, tokenizer)`** (cell 12) — pipeline glue:
  `get_words_labels` → optional `segment_to_groups`+`realign_tags` → tokenize
  `is_split_into_words=True` → `align_labels`.
- **`realign_tags`** (cell 9) — collapses syllables into segmented words, fixing
  `I-` → `B-` when a word boundary splits an entity.
- **`align_labels`** (cell 11) — first-subword-keeps-label, rest `-100`.
- **`segment_to_groups`** (cell 10) — py_vncorenlp `wseg`; **fallback = 1
  syllable/word** if reconstructed length mismatches.
- **Models**: `BertCrfForNer` (cell 16) and `GatCrfForNer` (cell 28). Both share
  the `-100→0` "safe labels" CRF trick and return `-crf(...)` loss in training /
  `crf.decode` at inference. GAT path = full token graph `GATConv` + a
  `MultiheadAttention` block before the classifier.
- **`train_and_eval`** (cell 20) — AdamW, linear warmup, AMP `GradScaler`,
  grad-accum, early stopping (patience=3 on val micro-F1), saves `.pt` each epoch
  + `.pt.best`, writes `results/<exp_id>.csv`.
- **`decode_predictions`** (cell 18) — branches on head: softmax indexes preds by
  position `k`, CRF by a compacted counter `j` (CRF output already excludes pads).
- **`crossdomain_eval`** (cell 25) — projects both sides to `SHARED_LABELS`
  (7 types) via `SHARED_MAP_VIETMED`/`SHARED_MAP_PHONER`.

## Important conventions / invariants (reuse these, don't reinvent)
- Labels built dynamically from the dataset; `"O"` forced to index 0.
- `cfg["segment"]` True for PhoBERT/ViHealthBERT (word-segmented encoders),
  False for XLM-R (raw syllables).
- CRF heads cannot pass `-100` to `torchcrf`; the `safe = labels.clone();
  safe[safe==-100]=0` masking pattern is load-bearing and repeated in both models.
- The 4 "bug-prone pure functions" (`realign_tags`, `align_labels`,
  `project_tags`, `spans`/`classify_errors`) are **CPU-verifiable via §T** — any
  change to them should keep §T green.

## Known gaps / things to watch (for future requests)
- **Heavy code duplication** between `BertCrfForNer` and `GatCrfForNer` (forward
  body, CRF loss/decoss) — candidate for a shared base class.
- `torch.cuda.amp.autocast/GradScaler` use the **deprecated API** (newer torch
  wants `torch.amp.*`); fine for the pinned Colab torch, flag if upgrading.
- §9 PhoNER `train`/`dev`/`test` path is hardcoded; comment notes to confirm path
  after clone.
- `__init__` GAT `heads` must divide hidden size (`H//heads`); only checked
  implicitly.
- One experiment per Colab session by design (T4 disconnects) — no orchestration
  loop across `EXPERIMENTS`.
- Results/checkpoints all live on Google Drive paths, not in-repo.

## How to verify any future change (no GPU needed)
- Run **§T** (cell 30) on CPU — exercises the 4 pure functions; expects
  `ALL §T TESTS PASSED`.
- Run **§4 `verify_roundtrip`** (cell 14) per encoder before training — expects
  `verify_roundtrip PASSED for 5/5 samples`.
- Full training/eval requires Colab T4 + Drive (§0 install/mount lines).
