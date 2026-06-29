# Vietnamese Medical NER — Design Spec

**Date:** 2026-06-29
**Scope:** Master coursework / đồ án môn
**Compute:** Google Colab T4 (free tier)
**Goal:** A clean, reproducible NER pipeline that compares several strong baselines, runs a proper CRF-vs-Softmax ablation, adds one improved model (GAT+CRF), measures cross-domain generalization, and ships a serious error analysis. No novelty required — depth and rigor required.

---

## 1. Objectives & non-goals

**Objectives**
- Fine-tune multiple strong encoders on VietMed-NER and compare them entity-level.
- Ablate CRF vs Softmax decoding heads.
- Build one **improved model**: PhoBERT + GAT + CRF, compared head-to-head against the PhoBERT+CRF baseline.
- Evaluate the best model **zero-shot cross-domain** on PhoNER-COVID19.
- Produce an error analysis (boundary / type / missing / spurious) with **real Vietnamese examples**.
- Everything reproducible: fixed seeds, pinned deps, checkpoints + logs to Drive.

**Non-goals**
- No LLM fine-tuning, no beating SOTA. The value is the ablation + analysis, not the leaderboard.
- No production serving, no audio/ASR (text-only; the `audio` column is dropped).

---

## 2. Data

### 2.1 VietMed-NER (primary, in-domain)
- Source: HuggingFace Hub — `load_dataset("leduckhai/VietMed-NER")`.
- Splits: train ≈ 4.62k / validation ≈ 1.15k / test ≈ 3.5k sentences.
- Features: `words` (token list), `labels` (BIO tags), plus `text`/`audio`/`duration` (drop audio/duration).
- **18 medical entity types**, BIO scheme. Token text is **syllable-level** (not word-segmented).
- 17 types confirmed from the Hub card (DISEASESYMPTOM, DRUGCHEMICAL, ORGAN, TREATMENT, DATETIME, FOODDRINK, PERSONALCARE, DIAGNOSTICS, MEDDEVICETECHNIQUE, PREVENTIVEMED, SURGERY, OCCUPATION, AGE, GENDER, UNITCALIBRATOR, LOCATION, ORGANIZATION); the full label list is read programmatically from the dataset at load time — never hard-coded.

### 2.2 PhoNER-COVID19 (cross-domain test only)
- Source: GitHub `VinAIResearch/PhoNER_COVID19`, word-level CoNLL files.
- 10 entity types incl. `SYMPTOM&DISEASE`. Used **only** as a zero-shot test set — never trained on.

### 2.3 Shared cross-domain schema (~7 types)
Both label sets are folded to a common schema for cross-domain eval:

| Shared type | VietMed-NER | PhoNER-COVID19 |
|---|---|---|
| SYMPTOM_DISEASE | DISEASESYMPTOM | SYMPTOM&DISEASE |
| AGE | AGE | AGE |
| GENDER | GENDER | GENDER |
| OCCUPATION | OCCUPATION | OCCUPATION |
| LOCATION | LOCATION | LOCATION |
| ORGANIZATION | ORGANIZATION | ORGANIZATION |
| DATE | DATETIME | DATE |

Any label outside this set maps to `O`. For cross-domain eval, both gold and prediction are projected to the shared schema before scoring.

---

## 3. Code structure

**One self-contained notebook**, no external `.py` files — upload one file to Colab and run. Organized into clearly-labeled section cells so it stays readable for grading.

```
NER/VietMed_NER.ipynb
 §0  Setup        — pip install (pinned), mount Drive, set seed, imports
 §1  Config       — EXPERIMENTS dict; pick EXP_ID to run one row
 §2  Data         — load VietMed-NER (HF) + PhoNER (clone); label maps + shared schema
 §3  Preprocess   — per-encoder word-seg + syllable→word tag re-align; subword + -100
 §4  Verify       — decode-back 5 samples, assert spans match gold  (GATE before training)
 §5  Model        — build_model(): softmax | CRF | GAT+CRF
 §6  Train        — HF Trainer (CRF-aware compute_loss), checkpoint each epoch → Drive
 §7  Eval         — seqeval entity-level: micro / macro / per-entity → results/<EXP_ID>.csv
 §8  Error analysis — boundary/type/missing/spurious + real VN examples → CSV
 §9  Cross-domain — load best ckpt, zero-shot PhoNER through shared schema
 §10 GAT model    — PhoBERT + GAT + CRF (the improved model)
```

Operating rule: run one experiment at a time by setting `EXP_ID`; do **not** run all rows in one Colab session (T4 free disconnects). Each run writes its own checkpoint + CSV to Drive, so a disconnect never loses results.

---

## 4. Preprocessing (per-encoder segmentation)

VietMed-NER is syllable-level; encoders differ in what they expect:

- **PhoBERT / ViHealthBERT** (pretrained on word-segmented text):
  1. Word-segment with `py_vncorenlp` (loads model directly — no Java server, Colab-friendly).
  2. **Re-align syllable BIO tags onto merged words**: a word's tag = its first syllable's tag; collapse `B-X` + `I-X` syllables into a single `B-X` word, demote stray `I-` to match.
- **XLM-R** (pretrained on raw text): use raw syllables, no segmentation.

Then for every encoder: subword-tokenize, keep the label on the **first subword** of each word, set the rest and special tokens to `-100`.

**§4 Verification gate (mandatory before any training):** decode 5 samples back from aligned ids and assert the recovered spans equal the gold spans. Training cells refuse to run if this fails.

---

## 5. Models

`build_model(cfg)` returns one of:

- **Softmax**: `AutoModelForTokenClassification.from_pretrained(encoder, num_labels=N)`.
- **CRF**: encoder + `nn.Linear` emissions + `torchcrf.CRF` (batch_first). Loss = `-crf(emissions, safe_labels, mask)`; `-100` labels mapped to 0 and excluded via mask. Decode via `crf.decode`.
- **GAT+CRF (improved)**: PhoBERT embeddings → `GATConv` over a fully-connected token graph → `MultiheadAttention` → `Linear` → CRF head. Same encoder + same CRF head as the baseline so the **only** delta is the GAT block.

`num_labels` always derived from the loaded dataset's label list.

---

## 6. Experiment matrix

| # | Encoder | Head | max_len | Train | Test | Role |
|---|---|---|---|---|---|---|
| 1 | PhoBERT-base | Softmax | 256 | VietMed-NER | in-domain | baseline |
| 2 | PhoBERT-base | CRF | 256 | VietMed-NER | in-domain | **baseline gốc** |
| 2b | PhoBERT-base | CRF | 128 | VietMed-NER | in-domain | fair-compare mốc cho #5 |
| 3 | XLM-R-base | CRF | 256 | VietMed-NER | in-domain | multilingual baseline |
| 4 | ViHealthBERT | CRF | 256 | VietMed-NER | in-domain | domain baseline |
| 5 | PhoBERT + GAT | CRF | 128 | VietMed-NER | in-domain | **improved model** |
| 6 | best of above | — | — | — | PhoNER-COVID19 | cross-domain zero-shot |

**Headline comparisons:**
- CRF vs Softmax: #1 vs #2.
- Improved vs baseline (fair, matched length): **#2b (PhoBERT+CRF @128) vs #5 (PhoBERT+GAT+CRF @128)**.
- #2 @256 is kept as the baseline's best reported number.

GAT is O(L²) in edges → `max_len=128`, smaller batch + gradient accumulation if OOM.

---

## 7. Training config (T4)

| Hyperparam | Value |
|---|---|
| learning rate | 2e-5, linear warmup + decay |
| warmup ratio | 0.1 |
| batch size | 16 (grad accumulation if OOM) |
| max seq length | 256 (128 for GAT and #2b) |
| epochs | up to 30 + early stopping (patience on val micro-F1) |
| dropout | 0.1 |
| optimizer | AdamW, weight decay 0.01 |
| fp16 | True |

- HF `Trainer` for softmax; `Trainer` subclass with custom `compute_loss` for CRF/GAT+CRF.
- Checkpoint every epoch to Google Drive; resume-safe.
- **Seeds:** 1 fixed seed for all rows; **3 seeds (report mean/std) for the single best config** if time permits.

---

## 8. Evaluation

- `seqeval`, **entity-level strict match** (span + type both correct = TP).
- Report **micro-F1 + macro-F1 + per-entity F1** for every run.
- Each run appends a row to `results/<EXP_ID>.csv`; a final summary table aggregates the matrix.
- Cross-domain (#6): project gold + pred to the shared 7-type schema, then seqeval.

---

## 9. Error analysis (the score-separating section)

`§8` compares gold vs predicted spans and classifies every error:

| Category | Definition |
|---|---|
| Boundary | right type, wrong span extent |
| Type | right span, wrong type |
| Missing | gold entity not predicted (FN) |
| Spurious | predicted entity with no gold (FP) |

Output: a count table + a CSV holding, for each error, the **real Vietnamese sentence and the offending span**. Pull representative examples per category into the report. Compare error profiles across baseline vs improved (#2b vs #5) to explain *why* GAT helps or doesn't on this data.

---

## 10. Reproducibility & ops

- Fixed seed across `random`/`numpy`/`torch` (+ deterministic flags where feasible).
- Pinned installs in §0: `transformers`, `datasets`, `seqeval`, `torchcrf`, `py_vncorenlp`, `torch-geometric` (matched to Colab CUDA), `pandas`.
- All hyperparams live in the §1 `EXPERIMENTS` config.
- Checkpoints + result CSVs persisted to Drive every epoch; runs resume after disconnect.
- `torch-geometric` install for GAT must be pinned to the Colab CUDA version (common failure point).

---

## 11. Milestones (4 weeks)

- **Week 1** — Load VietMed-NER, build preprocessing (per-encoder wseg + `-100` alignment), pass §4 verification gate.
- **Week 2** — PhoBERT softmax baseline end-to-end (#1), seqeval micro/macro/per-entity.
- **Week 3** — Add CRF (#2, #2b); run XLM-R (#3), ViHealthBERT (#4); build + run GAT+CRF (#5); cross-domain (#6).
- **Week 4** — Error analysis (4 categories + real examples) + report + slides.

---

## 12. Report structure

Intro (task + Vietnamese challenges) → Related work → Method (pipeline + CRF + GAT) → Experiments (matrix §6) → Error analysis with real Vietnamese examples → Conclusion.

---

## 13. Risks

- **torch-geometric / CUDA mismatch on Colab** — pin versions; verify import before training GAT.
- **Syllable→word tag re-alignment bugs** — §4 verification gate is the guard; do not skip.
- **T4 disconnects** — per-epoch Drive checkpointing + one-experiment-per-session discipline.
- **ViHealthBERT availability/tokenizer quirks** — confirm the HF model id and tokenizer load early in Week 3; fall back to reporting #1–3 + #5 if it blocks.

---

## 14. References (verify links/venues before citing)

- Le-Duc et al. (2025). *Medical Spoken Named Entity Recognition*. NAACL — VietMed-NER. arXiv:2406.13337.
- Nguyen & Nguyen (2020). *PhoBERT*. EMNLP Findings.
- Minh et al. (2022). *ViHealthBERT*. LREC.
- Truong et al. (2021). *PhoNER-COVID19*. NAACL. arXiv:2104.03879.
- Ba-Quang Nguyen (2025). *TextGraphFuseGAT*. arXiv:2510.11537.
