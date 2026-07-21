# §9c — Confidence-gated voting (accuracy-oriented post-processing)

**Date:** 2026-07-21
**Notebook:** `VietMed_NER_07_phobert_vihealthbert_gradio.ipynb`
**Replaces:** old §9 (error analysis) and §9b (repair + blind voting) — both removed.

## Goal

Add an **accuracy-oriented** post-processing section. Unlike the removed §9b (which
only *legalized* or blindly flattened argmax output and did not aim to raise F1), §9c
tries to actually improve entity-level F1 — while staying **output-only, no retraining**,
so the reused PhoBERT checkpoint stays valid and the same treatment applies to both
encoders (fair comparison).

Scope decision (agreed): **only confidence-gated voting.** Viterbi decoding was
considered and dropped by the user. Ensembling and per-class threshold tuning are out.

## The insight

A softmax NER head labels each word independently, so `argmax` discards two things:
1. the model's **confidence** (the value of the max probability), and
2. evidence from **other occurrences of the same surface word** across the corpus.

Medical transcripts repeat terms, so a word's true label is usually stable. Treat each
occurrence as an independent noisy measurement of "what label does the model assign this
word?" — the **majority label** denoises random slips. But some words are genuinely
context-dependent (e.g. `đau` in `thuốc giảm đau` is `O`, not `SYMPTOM`); blind voting
destroys the confident-and-correct minority case. **Gating fixes only the tokens the
model was unsure about**, protecting confident predictions.

## Method (output-only)

1. **`test_conf_seqs(cfg, tok, model, split)`** — run the model; return per sentence
   `(gold_tags, pred_tags, words, confidences)` where `confidence` = `max softmax prob`
   per word (subword `-100` positions dropped). Self-contained (replaces the removed
   `test_tag_seqs`).
2. **`build_vote_map(P, W)`** — per **surface word**, tally the predicted entity-type
   across the whole test set; majority type per word. Thousands of independent per-word
   votes, not one global vote. Uses only predictions + inputs — **never gold labels**
   (no leakage).
3. **`gated_vote(P, W, CONF, tau)`** — per token: `conf >= tau` → keep model's own type;
   `conf < tau` → adopt the word's majority type. Re-derive BIO with `_types_to_bio`.
4. **`blind_vote(P, W)`** — reference row: override every token with the majority
   (the removed §9b behaviour), kept only for the gated-vs-blind contrast.
5. **`tune_tau`** — pick `tau` on **validation** over a small grid (0.5–0.95) maximizing
   val micro-F1; report the chosen `tau`. The only fitted parameter.

## Reporting

Per model (PhoBERT, ViHealthBERT), a ladder → CSV
`{RESULTS_DIR}/gated_vote_phobert_vs_vihealthbert.csv`:

`raw → +repair → +blind-vote → +gated-vote(tau*)`

each with entity-level micro/macro-F1 and the boundary/type/missing/spurious breakdown
(`spans` + `classify_errors` brought into the section, since §9 is removed).

## Honest caveats (must appear in the notebook markdown)

- **Transductive / offline-only.** Voting couples test sentences (a sentence's output
  depends on other sentences' outputs). Valid for batch corpus annotation; **NOT usable
  for the single-sentence Gradio demo (§10)**, which sees one sentence at a time.
- **No leakage:** uses model predictions + inputs only, never gold labels.
- **Low ceiling:** gated voting mainly repairs blind-voting's failure mode; it rarely
  produces a large F1 jump on its own. If it barely moves F1, that is itself the finding
  (the models already predict repeated terms consistently). The strong lever (Viterbi /
  a CRF head) would require the dropped structured-decoding work or a retrain.

## Tests (§T style, CPU, pure functions)

- `repair_bio(["O","I-AGE","I-AGE"]) == ["O","B-AGE","I-AGE"]`
- `_types_to_bio(["SYM","SYM","O","SYM"]) == ["B-SYM","I-SYM","O","B-SYM"]`
- gated: a high-confidence `O` token is kept; a low-confidence token flips to the word's
  majority; `blind_vote` overrides all.

## Cells

- **Delete:** old §9 (error analysis), §9b markdown, §9b code.
- **Insert after §8:** one markdown cell (insight + caveats) + one code cell
  (helpers + §T tests + run + report + CSV).

## Out of scope

Viterbi/constrained decoding, cross-model ensembling, per-class threshold tuning,
any retraining or model/loss change.
