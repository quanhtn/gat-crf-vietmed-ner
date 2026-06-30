# GAT-CRF VietMed-NER

A **graph-enhanced PhoBERT + GAT + CRF** model for **medical named entity recognition in Vietnamese**, benchmarked against strong baselines (PhoBERT/XLM-R/ViHealthBERT with Softmax/CRF heads). Includes a CRF-vs-Softmax ablation, zero-shot cross-domain evaluation, and a 4-category error analysis. Master coursework project, run on Google Colab (T4, free tier).

> **Core contribution:** the improved **PhoBERT + GAT + CRF** model — a Graph Attention layer over the **syntactic dependency graph** (VnCoreNLP parse, edges from each token to its head word, residually fused with BERT features) before CRF decoding, compared head-to-head against the PhoBERT+CRF baseline at matched sequence length.

## What this does

| | |
|---|---|
| **Primary data** | [VietMed-NER](https://huggingface.co/datasets/leduckhai/VietMed-NER) — 18 medical entity types, BIO, syllable-level. Splits ≈ 4.62k / 1.15k / 3.5k. |
| **Cross-domain test** | [PhoNER-COVID19](https://github.com/VinAIResearch/PhoNER_COVID19) — 10 types, used **test-only** via a shared 7-type schema. |
| **Encoders** | PhoBERT-base, XLM-R-base, ViHealthBERT |
| **Heads** | Softmax · CRF · GAT+CRF |
| **Eval** | `seqeval` entity-level strict (micro / macro / per-entity F1) |

### Experiment matrix

| # | Encoder | Head | max_len | Role |
|---|---|---|---|---|
| 1 | PhoBERT | Softmax | 256 | baseline |
| 2 | PhoBERT | CRF | 256 | baseline gốc |
| 2b | PhoBERT | CRF | 128 | fair-compare mốc cho #5 |
| 3 | XLM-R | CRF | 256 | multilingual baseline |
| 4 | ViHealthBERT | CRF | 256 | domain baseline |
| 5 | **PhoBERT + GAT** | **CRF** | 128 | **improved model** |
| 6 | best of above | — | — | cross-domain zero-shot (PhoNER) |

**Headline ablations:** CRF vs Softmax (#1 vs #2); improved vs baseline at matched length (#2b vs #5).

## Experimental protocol

To keep the baseline-vs-improved comparison fair and reproducible:

- **Controlled comparison.** The headline claim is **#5 (GAT+CRF) vs #2b (CRF)** — *same* encoder (PhoBERT), *same* `max_len=128`, *same* segmentation and `-100` alignment. Only the head differs, so any F1 gap is attributable to the dependency-graph GAT, not to sequence length or tokenization. Do **not** read #5 against #1/#2 (those run at `max_len=256`).
- **Fixed training recipe** (identical across all rows unless noted): AdamW, `lr=2e-5`, weight decay 0.01, linear warmup 10%, AMP, up to 30 epochs, **early stopping** on validation micro-F1 (patience 3). Per-row `batch_size`/`grad_accum` keep the effective batch ≈ 16 within T4 memory.
- **Model selection.** Best checkpoint = highest **validation** micro-F1; test metrics are reported **once** from that checkpoint (no test-set tuning).
- **Metrics.** `seqeval` entity-level **strict** micro / macro / per-entity F1. Cross-domain (#6) projects both sides to the shared 7-type schema before scoring.
- **Seeds.** Single fixed seed (`SEED=42`) for the default runs. For the headline #2b-vs-#5 comparison, report **≥3 seeds** (mean ± std) so the improvement is shown to exceed run-to-run variance.
- **The dependency graph** is built only for the GAT row (other rows carry an identity/self-loop `dep_head` that is never used), so baselines incur no parser cost and stay directly comparable.

## How to run (Colab)

1. Upload `VietMed_NER.ipynb` to Google Colab, set runtime to **GPU (T4)**.
2. Run **§0 Setup** (pins deps, mounts Drive, sets seed).
3. In **§1**, set `EXP_ID` to the row you want (one experiment per session — T4 free disconnects).
4. Run **§4 verify gate** — must print `verify_roundtrip PASSED for 5/5 samples` before training.
5. Run `train_and_eval(cfg, EXP_ID)`. Checkpoints + `results/<EXP_ID>.csv` are written to Drive each epoch.
6. Repeat for each row; then **§9** cross-domain on the best model, **§10** for the GAT model.

### Key design decisions

- **Per-encoder segmentation:** PhoBERT/ViHealthBERT use `py_vncorenlp` word-segmentation + syllable→word tag re-alignment; XLM-R uses raw syllables.
- **Subword `-100` alignment:** only the first subword of each word keeps its label.
- **Reproducibility:** fixed seed, pinned deps, per-epoch Drive checkpoints, CSV logs.

## Local development

The bug-prone pure functions (tag re-alignment, subword alignment, schema projection, error classification) run on CPU with **no GPU/Colab** — see the **§T Tests** cell. They are verified before any training touches them.

## Repo layout

```
VietMed_NER.ipynb                          # the single self-contained notebook
requirements.txt                           # pinned deps (torch/PyG installed in-Colab)
docs/superpowers/specs/...-design.md       # design spec
docs/superpowers/plans/...-ner.md          # 15-task implementation plan
results/                                   # per-run CSVs (gitignored until produced)
```

## References

- Le-Duc et al. (2025). *Medical Spoken Named Entity Recognition.* NAACL — VietMed-NER. arXiv:2406.13337
- Nguyen & Nguyen (2020). *PhoBERT.* EMNLP Findings
- Minh et al. (2022). *ViHealthBERT.* LREC
- Truong et al. (2021). *PhoNER-COVID19.* NAACL. arXiv:2104.03879
- Ba-Quang Nguyen (2025). *TextGraphFuseGAT.* arXiv:2510.11537
