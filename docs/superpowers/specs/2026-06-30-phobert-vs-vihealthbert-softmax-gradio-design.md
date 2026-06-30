# PhoBERT vs ViHealthBERT (Softmax, full fine-tune) + Gradio demo — Design

**Date:** 2026-06-30
**Status:** Approved (design), pending spec review
**Scope:** A new self-contained Colab notebook (`nb07`) that compares two **full
fine-tuned** Softmax NER models — **PhoBERT** vs **ViHealthBERT** — on VietMed-NER,
then launches a small **Gradio** app where the user types a sentence and picks which
of the two encoders runs.

## 1. Motivation & claim

The earlier head-ablation direction (Softmax/CRF/GAT+CRF on a frozen encoder) is dropped.
New strategy: keep the head fixed (**Softmax**) and **full fine-tune** two encoders, to
isolate the effect of the encoder. The honest claim:

> On VietMed-NER, under identical training (same head, config, pipeline), a domain
> medical encoder (ViHealthBERT) yields higher (or not) entity-level F1 than a general
> encoder (PhoBERT).

Deliverable also includes an interactive Gradio demo (encoder picker) for qualitative
inspection on user-typed Vietnamese medical sentences.

**Why full fine-tune (not frozen):** pretraining (MLM) gives general language
understanding but no task knowledge (label set, entity boundaries, BIO). Frozen + a tiny
random Softmax scored only ~0.546 micro-F1 in our own run; full fine-tune typically
reaches ~0.85+. The ~0.3 F1 gap is the value of fine-tuning, and a demo needs a strong
model.

## 2. Non-goals

- No CRF / GAT / graph heads (Softmax only).
- No frozen-encoder feature caching.
- No XLM-R, no cross-domain (PhoNER) eval.
- No HuggingFace Spaces / local deployment — the app runs **inside Colab** with
  `launch(share=True)`.
- No retraining of PhoBERT — its existing checkpoint is reused.

## 3. Key facts locked from nb05

- PhoBERT Softmax checkpoint already exists on Drive:
  `/content/drive/MyDrive/vietmed_ner/checkpoints/01_phobert_softmax.pt` and `.pt.best`
  (state_dict of `AutoModelForTokenClassification`, RoBERTa head).
- Its training config: `encoder=vinai/phobert-base, head=softmax, segment=True,
  max_len=256, batch_size=16, lr=2e-5, AMP, 30 epochs, early-stop patience 3`.
- `DRIVE_DIR=/content/drive/MyDrive/vietmed_ner`, `CKPT_DIR=.../checkpoints`,
  `RESULTS_DIR=.../results`.
- Tokenizer asymmetry: **PhoBERT loads as `RobertaTokenizerFast`** (has `word_ids()`),
  **ViHealthBERT only has a slow tokenizer** (`AutoTokenizer(use_fast=False)`, no
  `word_ids()`). The pipeline must handle both.

## 4. Architecture & data flow

```
VietMed-NER (syllable BIO)
  → VnCoreNLP wseg + realign tags → encode (tokenizer-agnostic) + (-100) subword align  [REUSE]
  → PhoBERT:      LOAD 01_phobert_softmax.pt.best        ┐
  → ViHealthBERT: TRAIN full FT (same config) → save .pt ┘→ eval test (seqeval) → compare table + CSV
  → Gradio: user sentence → wseg → tokenize (per chosen encoder) → predict → BIO→spans → HighlightedText
```

## 5. Components

1. **Setup / paths.** Mount Drive; reuse `DRIVE_DIR/CKPT_DIR/RESULTS_DIR`. Installs:
   `transformers==4.44.2`, `datasets==2.21.0`, `seqeval`, `py_vncorenlp`, `pandas`,
   `gradio`.
2. **Data pipeline (reused from nb05):** `get_words_labels`, `realign_tags`,
   `segment_to_groups`/`get_segmenter`, label maps, plus the `verify_roundtrip` gate and
   §T CPU tests.
3. **Dual-path tokenization (the crux).** `build_encoded(split, cfg, tokenizer)` branches
   on `tokenizer.is_fast`:
   - **fast** (PhoBERT): the exact nb05 path — `tokenizer(words, is_split_into_words=True,
     truncation, max_length, padding="max_length")` + `align_labels` via `word_ids()`.
     Guarantees the reused PhoBERT checkpoint evaluates to its true test F1.
   - **slow** (ViHealthBERT): manual `encode_words` — tokenize each pre-segmented word
     with `tokenizer.encode(w, add_special_tokens=False)`; first subword keeps the label,
     the rest get `-100`; build CLS/SEP/PAD, attention mask, and a `word_ids` list; wrap in
     an `_Encoded` object exposing `__getitem__` and `word_ids(batch_index)`.
   - Both branches return the same dict interface (`input_ids`, `attention_mask`, `labels`,
     `word_ids`).
4. **Model builder.** `AutoModelForTokenClassification.from_pretrained(encoder,
   num_labels, id2label, label2id)`, then `resize_token_embeddings(len(tokenizer))`.
5. **PhoBERT (reuse).** Build model + tokenizer (fast), `load_state_dict(
   torch.load(CKPT_DIR+'/01_phobert_softmax.pt.best'))`, eval on test. No training.
6. **ViHealthBERT (train).** `train_and_eval`-style loop reused from nb05: lr=2e-5, AMP,
   30 epochs, early-stop patience 3, **same config as PhoBERT** (segment=True,
   max_len=256, batch=16). Save `02_vihealthbert_softmax.pt` and `.pt.best`. Eval on test.
7. **Comparison.** 2-row table (PhoBERT / ViHealthBERT) of test micro/macro F1 +
   `classification_report` per-entity for each, written to
   `{RESULTS_DIR}/compare_phobert_vs_vihealthbert.csv`.
8. **Gradio app.** `gr.Dropdown(["PhoBERT","ViHealthBERT"])` + `gr.Textbox`. On submit:
   lazy-load (and cache) the chosen model+tokenizer; VnCoreNLP word-segment the input;
   tokenize via the same dual path; `argmax` over logits; map subword→word (first subword);
   collapse BIO tags into spans; render with `gr.HighlightedText` (inline colored spans by
   entity type). `launch(share=True)`.

## 6. Eval correctness

- Entity-level **strict** micro/macro F1 via seqeval, reusing nb05 `evaluate_tags` /
  `decode_predictions` (softmax branch).
- PhoBERT must use the fast path identical to its training, so its loaded eval matches
  nb05 (regression check below).

## 7. Success criteria (verifiable)

1. `verify_roundtrip` PASS and §T tests PASS for **both** the fast and slow paths.
2. PhoBERT: load checkpoint, eval test → micro-F1 in a sane range (≈0.8+, clearly not
   ~0) — confirms correct load + matched preprocessing.
3. ViHealthBERT: training converges, checkpoint saved, test metrics produced.
4. Comparison cell prints a 2-row table + per-entity reports and writes the CSV.
5. Gradio `share` link opens; a Vietnamese medical sentence renders highlighted spans;
   switching the encoder changes the output.

## 8. Risks & mitigations

- **Asymmetry** (PhoBERT trained in a prior session, possibly different seed/lib): accept
  for a coursework comparison; report it explicitly as "PhoBERT from prior checkpoint,
  identical config."
- **Tokenizer/checkpoint mismatch → wrong F1:** mitigated by the fast path for PhoBERT
  (identical to training) plus success-criterion 2 as a guard.
- **VnCoreNLP on odd/short user input:** `segment_to_groups` already falls back to
  one-syllable-per-word.
- **Gradio model memory (two fine-tuned encoders):** lazy-load and cache per encoder; T4
  holds one ~135M model at a time comfortably, both if needed.
