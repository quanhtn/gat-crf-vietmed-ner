# Frozen-Encoder Head Ablation — Design

**Date:** 2026-06-30
**Status:** Approved (design), pending spec review
**Scope:** A new, self-contained Colab notebook that compares three NER decoding heads
(**Softmax · CRF · GAT+CRF**) on top of a **frozen** PhoBERT encoder, on VietMed-NER.

## 1. Motivation

The existing notebook (`VietMed_NER_05_phobert_gat_crf.ipynb`) trains every experiment by
**full fine-tuning** the PhoBERT encoder (`opt = AdamW(model.parameters())`, no
`requires_grad=False` anywhere). For the PhoBERT+GAT+CRF row this proved infeasible on
Colab T4 free tier:

- **Too slow** — backprop flows through all ~135M encoder params for 30 epochs, plus the
  GAT runs a Python `for b in range(B)` loop over a **fully-connected L×L** graph (L=128 →
  ~16k edges/sentence).
- **F1 ≈ 0.0001 after 3 epochs** — a correctness bug, not a resource limit: the GAT's
  `_full_edges` connects **all 128 positions including PAD**, so every real token aggregates
  padding noise → representations collapse → model predicts all-`O` → entity-F1 ≈ 0.

This design sidesteps both by **freezing the encoder and caching its features once**, then
training only lightweight heads. The fair-comparison rule from the project README is kept:
**all heads share the same frozen features; only the head differs.**

## 2. Claim (honest framing)

Because the GAT uses a **fully-connected (PAD-masked)** graph — not the syntactic dependency
graph — the contribution claim is deliberately **modest**:

> On the same frozen PhoBERT representations, a richer graph-attention + CRF head yields
> higher entity-level F1 than simpler heads (CRF, Softmax).

This is a valid **head ablation**, not a claim that syntactic dependency structure helps.
The report must state this explicitly.

## 3. Non-goals

- No syntactic dependency graph / VnCoreNLP `parse` annotator (explicitly deferred — effort).
- No encoder fine-tuning of any kind in this notebook.
- No XLM-R rows, no cross-domain (PhoNER) eval.
- The previously trained **full-fine-tuned PhoBERT+Softmax** result is **not** the baseline
  here; if reported at all it is a separate "full-fine-tuning reference" row, never compared
  head-to-head against the frozen heads.

## 3b. Addendum (2026-06-30) — two no-/low-cost performance levers

After the initial build, two frozen-compatible improvements were added (both keep the encoder
frozen, neither requires fine-tuning):

- **Domain encoder (ViHealthBERT) frozen.** Default `CFG["encoder"]` is now
  `demdecuong/vihealthbert-base-word` (Vietnamese medical pretraining) instead of vanilla
  PhoBERT, for better frozen features on medical text. `vinai/phobert-base` remains a one-line
  swap so the two encoders can be compared. Feature caches are keyed by encoder tag to avoid
  collision. The "single encoder = PhoBERT-base" non-goal above is superseded by this.
- **BIO-constrained CRF.** Illegal tag transitions (`O→I-X`, `B-X→I-Y`, `I-X→I-Y` with X≠Y,
  and starting a sequence with `I-X`) are pinned to `-1e4` in the CRF transition/start matrices
  and frozen (gradient masked to 0), so the CRF and GAT+CRF heads can never emit BIO-invalid
  spans. Applies to the CRF-based heads only; Softmax (independent classification) is left as-is.

Both are inference-/decode-side or domain-swap levers, so they do not change the "frozen
encoder, only the head differs" fairness rule of the comparison.

## 4. Architecture & data flow

```
VietMed-NER raw (syllable BIO)
  → segment (VnCoreNLP wseg) + realign tags + tokenize + (-100) subword align   [REUSED]
  → [ONE TIME] frozen PhoBERT forward → cache: features[B,L,H] (fp16) + mask + labels  → Drive (.pt)
  → [FAST] train each head on cache → decode → seqeval entity-level F1 → comparison table
```

After caching, **PhoBERT is never forwarded again**; each head trains in seconds–minutes on
the cached tensors, with no OOM and no disconnect risk.

## 5. Components (each one clear responsibility)

1. **Data pipeline (reused verbatim).** `get_words_labels`, `segment_to_groups`,
   `realign_tags`, `align_labels`, `build_encoded`, label maps, and the §4 `verify_roundtrip`
   gate and §T CPU pure-function tests are copied unchanged from the existing notebook. Not
   modified, so their existing tests still hold.

2. **Feature cache — `cache_features(split)`.**
   - Load frozen `AutoModel.from_pretrained("vinai/phobert-base")`, `.eval()`,
     all params `requires_grad=False`.
   - Forward under `torch.no_grad()` in batches; store `last_hidden_state` as **fp16**.
   - Persist `{features, attention_mask, labels}` per split as `.pt` on Drive
     (`{CKPT_DIR}/cache_{split}.pt`). Skip recompute if file exists.
   - Estimated size: train ≈ 4.62k×128×768×2B ≈ 0.9 GB; ~1.2–1.5 GB total across splits.
   - Sanity assert: `features.shape[:2] == labels.shape`.

3. **Three heads (operate on `[B,L,H]`, never touch PhoBERT).**
   - `SoftmaxHead`: `Linear(H, C)`, cross-entropy with `ignore_index=-100`.
   - `CrfHead`: `Linear(H, C)` + `CRF(C, batch_first=True)`.
   - `GatCrfHead`: `GATConv(H, H//heads, heads)` over a **PAD-masked** graph (edges only among
     positions with `attention_mask=1`, plus self-loops), batched via PyG `Batch` (no Python
     per-sample loop) → `Linear(H, C)` → `CRF`. fp32 throughout.

4. **Generic trainer — `train_head(head, cache_tr, cache_va)`.**
   - `AdamW`, **lr ≈ 1e-3** (only a small head trains), weight decay 0.01, linear warmup 10%.
   - **fp32, no AMP** (heads are tiny; keeps CRF numerically stable).
   - Early stopping on validation micro-F1, patience 3, up to 30 epochs.
   - Best checkpoint = highest val micro-F1; test metrics reported once from it.

5. **Comparison & report.**
   - Run all three heads; collect `seqeval` entity-level **strict micro / macro / per-entity F1**
     on test.
   - Output a 3-row comparison table + per-head `classification_report`, written to
     `{RESULTS_DIR}/frozen_head_ablation.csv`.

## 6. Carried-over correctness fixes

- **PAD-masked GAT graph** — the fix that resolves F1≈0. Edges only among real tokens.
- **CRF in fp32** — trivial here since AMP is dropped entirely.
- **`-100` handling identical across CRF and GAT+CRF** (`safe = labels.clone(); safe[safe==-100]=0`
  with `mask = attention_mask.bool()`), matching the existing baseline so the heads stay
  directly comparable.

## 7. Robustness / multi-seed

Because head training is cheap, run **3 seeds** per head and report **mean ± std** of test
micro/macro F1. This shows the Softmax < CRF < GAT+CRF ordering exceeds run-to-run variance.
(Reducible to 1 seed if minimal effort is preferred.)

## 8. Success criteria (verifiable)

1. `verify_roundtrip` prints PASS and §T tests print `ALL §T TESTS PASSED` before any training.
2. `cache_features` produces `.pt` files with `features.shape[:2] == labels.shape` for all splits.
3. Each head trains to a **non-degenerate** test micro-F1 (clearly > 0; the F1≈0 bug is gone) —
   this is the explicit regression check that the PAD-mask fix works.
4. Final cell prints a 3-row table (Softmax / CRF / GAT+CRF) of test micro/macro F1 with
   mean ± std over 3 seeds, plus per-entity reports, and writes the CSV.

## 9. Risks & mitigations

- **Frozen vanilla PhoBERT → low absolute F1.** Expected; the comparison is relative ordering,
  not absolute SOTA. Stated openly in the report.
- **GAT+CRF may not beat CRF.** Possible — if so, report it honestly; a null result on a head
  ablation is still a valid finding. Multi-seed makes the conclusion defensible either way.
- **Cache size on Drive.** fp16 keeps it ~1.2–1.5 GB; acceptable on free Drive. Per-split files
  let a session cache one split at a time if needed.
