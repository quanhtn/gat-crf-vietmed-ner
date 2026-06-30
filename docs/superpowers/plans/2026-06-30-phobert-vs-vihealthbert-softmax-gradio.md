# PhoBERT vs ViHealthBERT (Softmax) + Gradio — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `VietMed_NER_07_phobert_vihealthbert_gradio.ipynb` that reuses the existing PhoBERT Softmax checkpoint, full fine-tunes a ViHealthBERT Softmax model with the identical config, compares both (entity F1 + error analysis), and launches a Gradio demo with an encoder picker.

**Architecture:** A single self-contained Colab notebook, assembled by a builder script `build_nb07.py` (kept in the scratchpad, same pattern as nb06). The notebook reuses nb05's data pipeline, adds a **dual-path tokenizer** (fast for PhoBERT, manual for ViHealthBERT), loads/evaluates PhoBERT, trains/saves/evaluates ViHealthBERT, runs comparison + boundary/type/missing/spurious error analysis, and serves a `gr.HighlightedText` app. Pure helper functions are unit-tested offline; heavy steps are gated on Colab.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers==4.44.2`, `datasets==2.21.0`, `seqeval`, `py_vncorenlp`, `torchcrf` (only for nb05 parity import — not used by softmax), `gradio`, `pandas`.

## Global Constraints

- Notebook output path (committed): `VietMed_NER_07_phobert_vihealthbert_gradio.ipynb` (repo root).
- Builder script path (scratchpad, not committed): `<SCRATCH>/build_nb07.py` where `<SCRATCH>` = `/private/tmp/claude-501/-Users-trttung-Projects-llm-CS2230/<session>/scratchpad`.
- Drive paths (verbatim): `DRIVE_DIR='/content/drive/MyDrive/vietmed_ner'`, `CKPT_DIR='{DRIVE_DIR}/checkpoints'`, `RESULTS_DIR='{DRIVE_DIR}/results'`.
- PhoBERT checkpoint (reuse, do NOT retrain): `{CKPT_DIR}/01_phobert_softmax.pt.best`.
- ViHealthBERT checkpoint (create): `{CKPT_DIR}/02_vihealthbert_softmax.pt` and `.pt.best`.
- Shared train config for BOTH encoders: `segment=True, max_len=256, batch_size=16, lr=2e-5, AMP, epochs=30, patience=3`.
- Encoders: `vinai/phobert-base` (fast tokenizer), `demdecuong/vihealthbert-base-word` (slow tokenizer only — load with `AutoTokenizer(use_fast=False)`).
- Head: Softmax only (`AutoModelForTokenClassification`).
- First subword of each word keeps its label; the rest get `-100`.
- Gradio: `launch(share=True)`, output `gr.HighlightedText` (inline colored spans).
- Offline syntax check: compile every code cell except those whose first non-blank line starts with `!` (Jupyter magics).

---

## File Structure

- **`VietMed_NER_07_phobert_vihealthbert_gradio.ipynb`** — the deliverable, emitted by the builder. Sections: §0 Setup, §1 Config, §2 Data, §3 Preprocess (reuse), §3b Dual-path tokenization, §T Tests, §4 Verify gate, §5 Model builder, §6 PhoBERT load+eval, §7 ViHealthBERT train+eval, §8 Comparison, §9 Error analysis, §10 Gradio app.
- **`<SCRATCH>/build_nb07.py`** — assembles the notebook JSON via `md()`/`code()` helpers and writes the `.ipynb`.
- **`<SCRATCH>/test_helpers.py`** — offline unit tests for the pure functions (`encode_words`, `spans`, `classify_errors`, `tags_to_highlighted`).

Each task adds one cell-group to the builder, regenerates the notebook, runs the offline checks, and commits the regenerated `.ipynb`.

---

## Task 1: Builder scaffold + Setup/Config/Data cells

**Files:**
- Create: `<SCRATCH>/build_nb07.py`
- Create (regenerate): `VietMed_NER_07_phobert_vihealthbert_gradio.ipynb`

**Interfaces:**
- Produces: builder helpers `md(text)` and `code(text)` returning nbformat cell dicts; a `cells` list; final notebook write. Notebook globals defined for later cells: `DRIVE_DIR, CKPT_DIR, RESULTS_DIR, DEVICE, set_seed`, `CFG_PHO`, `CFG_VIH`, `raw, get_words_labels, LABEL_LIST, label2id, id2label`.

- [ ] **Step 1: Create the builder with helpers + §0/§1/§2 cells**

Create `<SCRATCH>/build_nb07.py`:

```python
import json

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": t.splitlines(keepends=True)})

md('''# VietMed-NER — PhoBERT vs ViHealthBERT (Softmax, full fine-tune) + Gradio

Reuse the PhoBERT Softmax checkpoint, full fine-tune ViHealthBERT with the **same config**,
compare entity-level F1 + error analysis, then a Gradio demo with an encoder picker.
Spec: `docs/superpowers/specs/2026-06-30-phobert-vs-vihealthbert-softmax-gradio-design.md`''')

code('''!pip -q install "transformers==4.44.2" "datasets==2.21.0" seqeval pytorch-crf py_vncorenlp pandas gradio
# §0 Setup  (the !pip line must stay first so the syntax-checker skips this cell)
import os, random, numpy as np, torch
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR   = '/content/drive/MyDrive/vietmed_ner'
CKPT_DIR    = f'{DRIVE_DIR}/checkpoints'
RESULTS_DIR = f'{DRIVE_DIR}/results'
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)''')

code('''# §1 Config — identical training config for both encoders (only the encoder differs)
CFG_PHO = dict(encoder="vinai/phobert-base",                head="softmax",
               segment=True, max_len=256, batch_size=16,
               ckpt=f'{CKPT_DIR}/01_phobert_softmax.pt.best')      # REUSE (no retrain)
CFG_VIH = dict(encoder="demdecuong/vihealthbert-base-word", head="softmax",
               segment=True, max_len=256, batch_size=16,
               ckpt=f'{CKPT_DIR}/02_vihealthbert_softmax.pt')      # TRAIN + save here
LR, EPOCHS, PATIENCE = 2e-5, 30, 3''')

code('''# §2 Data
from datasets import load_dataset
raw = load_dataset("leduckhai/VietMed-NER")
for c in ("audio", "duration"):
    if c in raw["train"].column_names:
        raw = raw.remove_columns(c)
TAG_COL  = "labels" if "labels" in raw["train"].column_names else "tags"
WORD_COL = "words"
def get_words_labels(split):
    ds = raw[split]; return list(ds[WORD_COL]), list(ds[TAG_COL])
_all_tags = set(t for split in raw for row in raw[split][TAG_COL] for t in row)
LABEL_LIST = ["O"] + sorted(t for t in _all_tags if t != "O")
label2id = {t:i for i,t in enumerate(LABEL_LIST)}
id2label = {i:t for t,i in label2id.items()}
print(len(LABEL_LIST), "labels; splits:", {k: len(raw[k]) for k in raw})''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
OUT = "/Users/trttung/Projects/llm/CS2230/gat-crf-vietmed-ner/VietMed_NER_07_phobert_vihealthbert_gradio.ipynb"
with open(OUT, "w") as f: json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
```

- [ ] **Step 2: Run the builder and syntax-check**

Run:
```bash
cd /Users/trttung/Projects/llm/CS2230/gat-crf-vietmed-ner && \
python <SCRATCH>/build_nb07.py && python - <<'PY'
import json
nb=json.load(open("VietMed_NER_07_phobert_vihealthbert_gradio.ipynb"))
errs=0
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    s="".join(c["source"])
    if s.lstrip().startswith("!"): continue
    try: compile(s,f"c{i}","exec")
    except SyntaxError as e: errs+=1; print("SYN",i,e)
print(f"{len(nb['cells'])} cells, {errs} syntax errors")
PY
```
Expected: `4 cells, 0 syntax errors` (1 markdown + 3 code; the §0 cell is skipped because its first line is `!pip`).

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): scaffold setup/config/data cells"
```

---

## Task 2: Preprocess reuse + dual-path tokenization + offline tests

**Files:**
- Modify: `<SCRATCH>/build_nb07.py` (append §3, §3b, §T cells before the `nb = {...}` write block)
- Create: `<SCRATCH>/test_helpers.py`
- Modify (regenerate): `VietMed_NER_07_phobert_vihealthbert_gradio.ipynb`

**Interfaces:**
- Consumes: `LABEL_LIST, label2id, get_words_labels` (Task 1).
- Produces: `realign_tags`, `segment_to_groups`, `get_segmenter`, `encode_words(words, label_ids, tokenizer, max_len) -> (ids, mask, labs, wids)`, `_Encoded`, `align_labels`, `build_encoded(split, cfg, tokenizer) -> _Encoded|BatchEncoding` (dual path on `tokenizer.is_fast`).

- [ ] **Step 1: Write the failing offline test for `encode_words`**

Create `<SCRATCH>/test_helpers.py`:

```python
# Mirrors the pure helpers emitted into nb07; asserts their behavior offline.
class _FakeTok:
    is_fast = False
    cls_token_id, sep_token_id, pad_token_id, unk_token_id = 1, 2, 3, 0
    def encode(self, w, add_special_tokens=False): return {"ab":[10,11], "c":[12]}[w]

def encode_words(words, label_ids, tokenizer, max_len):
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad, unk = tokenizer.pad_token_id, tokenizer.unk_token_id
    ids, labs, wids = [cls], [-100], [None]
    for wi, (w, lid) in enumerate(zip(words, label_ids)):
        sub = tokenizer.encode(w, add_special_tokens=False) or [unk]
        for k, s in enumerate(sub):
            ids.append(s); labs.append(lid if k == 0 else -100); wids.append(wi)
    ids, labs, wids = ids[:max_len-1], labs[:max_len-1], wids[:max_len-1]
    ids.append(sep); labs.append(-100); wids.append(None)
    n = len(ids); mask = [1]*n + [0]*(max_len-n)
    ids += [pad]*(max_len-n); labs += [-100]*(max_len-n); wids += [None]*(max_len-n)
    return ids, mask, labs, wids

def test_encode_words():
    ids, mask, labs, wids = encode_words(["ab","c"], [5,7], _FakeTok(), 7)
    assert ids  == [1,10,11,12,2,3,3],             ids
    assert labs == [-100,5,-100,7,-100,-100,-100], labs
    assert wids == [None,0,0,1,None,None,None],    wids
    assert mask == [1,1,1,1,1,0,0],                mask

def test_encode_words_truncates():
    ids, mask, labs, wids = encode_words(["ab","c"], [5,7], _FakeTok(), 4)
    assert ids  == [1,10,11,2],         ids
    assert labs == [-100,5,-100,-100],  labs

if __name__ == "__main__":
    test_encode_words(); test_encode_words_truncates()
    print("encode_words tests PASS")
```

- [ ] **Step 2: Run it to confirm it passes (reference behavior is the spec)**

Run: `python <SCRATCH>/test_helpers.py`
Expected: `encode_words tests PASS`. (This file is the executable spec the notebook cell must match verbatim.)

- [ ] **Step 3: Append §3 / §3b / §T cells to the builder**

Insert these `code(...)` calls in `build_nb07.py` immediately after the §2 cell and before the `nb = {...}` block:

```python
code('''# §3 Preprocess (reused from nb05)
def realign_tags(syllables, tags, word_groups):
    words, new_tags = [], []
    for grp in word_groups:
        words.append("_".join(syllables[i] for i in grp))
        head = tags[grp[0]]
        if head.startswith("I-"): head = "B-" + head[2:]
        new_tags.append(head)
    return words, new_tags

import py_vncorenlp
_SEG = None
def get_segmenter():
    global _SEG
    if _SEG is None:
        os.makedirs(f'{DRIVE_DIR}/vncorenlp', exist_ok=True)
        py_vncorenlp.download_model(save_dir=f'{DRIVE_DIR}/vncorenlp')
        _SEG = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=f'{DRIVE_DIR}/vncorenlp')
    return _SEG

def segment_to_groups(syllables):
    seg = get_segmenter()
    word_units = " ".join(seg.word_segment(" ".join(syllables))).split()
    groups, cur = [], 0
    for wu in word_units:
        n = wu.count("_") + 1
        groups.append(list(range(cur, cur + n))); cur += n
    if cur != len(syllables):
        return [[i] for i in range(len(syllables))]
    return groups''')

code('''# §3b Dual-path tokenization — fast (PhoBERT, word_ids) OR slow (ViHealthBERT, manual)
class _Encoded:
    def __init__(self, ids, mask, labels, wids):
        self._d = {"input_ids": ids, "attention_mask": mask, "labels": labels}; self._wid = wids
    def __getitem__(self, k): return self._d[k]
    def word_ids(self, batch_index=0): return self._wid[batch_index]

def encode_words(words, label_ids, tokenizer, max_len):
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad, unk = tokenizer.pad_token_id, tokenizer.unk_token_id
    ids, labs, wids = [cls], [-100], [None]
    for wi, (w, lid) in enumerate(zip(words, label_ids)):
        sub = tokenizer.encode(w, add_special_tokens=False) or [unk]
        for k, s in enumerate(sub):
            ids.append(s); labs.append(lid if k == 0 else -100); wids.append(wi)
    ids, labs, wids = ids[:max_len-1], labs[:max_len-1], wids[:max_len-1]
    ids.append(sep); labs.append(-100); wids.append(None)
    n = len(ids); mask = [1]*n + [0]*(max_len-n)
    ids += [pad]*(max_len-n); labs += [-100]*(max_len-n); wids += [None]*(max_len-n)
    return ids, mask, labs, wids

def align_labels(tok, word_label_ids):           # fast-path alignment via word_ids()
    aligned = []
    for i, labels in enumerate(word_label_ids):
        prev, ids = None, []
        for wid in tok.word_ids(batch_index=i):
            ids.append(-100 if (wid is None or wid == prev) else labels[wid]); prev = wid
        aligned.append(ids)
    tok["labels"] = aligned
    return tok

def _prep_words(split, cfg):
    words_col, tags_col = get_words_labels(split)
    enc_words, enc_lids = [], []
    for syl, tags in zip(words_col, tags_col):
        w, t = (realign_tags(syl, tags, segment_to_groups(syl)) if cfg["segment"] else (syl, tags))
        enc_words.append(w); enc_lids.append([label2id[x] for x in t])
    return enc_words, enc_lids

def build_encoded(split, cfg, tokenizer):
    enc_words, enc_lids = _prep_words(split, cfg)
    if tokenizer.is_fast:                          # PhoBERT — identical to nb05
        tok = tokenizer(enc_words, is_split_into_words=True, truncation=True,
                        max_length=cfg["max_len"], padding="max_length")
        return align_labels(tok, enc_lids)
    ids, mask, labs, wids = [], [], [], []         # ViHealthBERT — manual
    for w, l in zip(enc_words, enc_lids):
        a, b, c, d = encode_words(w, l, tokenizer, cfg["max_len"])
        ids.append(a); mask.append(b); labs.append(c); wids.append(d)
    return _Encoded(ids, mask, labs, wids)''')

code('''# §T tests — pure functions, CPU-only
def _test_realign():
    w,t = realign_tags(["bệnh","nhân","bị"], ["B-AGE","I-AGE","O"], [[0,1],[2]])
    assert w==["bệnh_nhân","bị"] and t==["B-AGE","O"]
    assert realign_tags(["a","b"], ["O","I-AGE"], [[0],[1]])[1]==["O","B-AGE"]
    print("realign_tags OK")
class _FakeTok:
    is_fast=False; cls_token_id,sep_token_id,pad_token_id,unk_token_id=1,2,3,0
    def encode(self,w,add_special_tokens=False): return {"ab":[10,11],"c":[12]}[w]
def _test_encode():
    ids,mask,labs,wids = encode_words(["ab","c"],[5,7],_FakeTok(),7)
    assert ids==[1,10,11,12,2,3,3] and labs==[-100,5,-100,7,-100,-100,-100]
    assert wids==[None,0,0,1,None,None,None] and mask==[1,1,1,1,1,0,0]
    print("encode_words OK")
_test_realign(); _test_encode()
print("ALL §T (part 1) PASSED")''')
```

- [ ] **Step 4: Regenerate + syntax-check**

Run:
```bash
cd /Users/trttung/Projects/llm/CS2230/gat-crf-vietmed-ner && \
python <SCRATCH>/build_nb07.py && python - <<'PY'
import json
nb=json.load(open("VietMed_NER_07_phobert_vihealthbert_gradio.ipynb")); e=0
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    s="".join(c["source"])
    if s.lstrip().startswith("!"): continue
    try: compile(s,f"c{i}","exec")
    except SyntaxError as ex: e+=1; print("SYN",i,ex)
print(f"{len(nb['cells'])} cells, {e} syntax errors")
PY
```
Expected: `7 cells, 0 syntax errors`.

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): reuse preprocess + dual-path tokenizer + unit tests"
```

---

## Task 3: Verify gate + model/tokenizer loader

**Files:**
- Modify: `<SCRATCH>/build_nb07.py`
- Modify (regenerate): the notebook

**Interfaces:**
- Consumes: `build_encoded, get_words_labels, realign_tags, segment_to_groups, id2label` (Tasks 1–2).
- Produces: `load_tokenizer(cfg) -> tokenizer`, `build_softmax_model(cfg, tokenizer) -> AutoModelForTokenClassification`, `verify_roundtrip(cfg, tokenizer, n=5)`.

- [ ] **Step 1: Append the §4 gate + §5 loader cells to the builder**

```python
code('''# §4 Verify (GATE) — roundtrip both tokenizer paths before any train/eval
from transformers import AutoTokenizer, RobertaTokenizerFast

def load_tokenizer(cfg):
    if cfg["encoder"] == "vinai/phobert-base":
        # MUST match nb05: RobertaTokenizerFast (byte-level, vocab=66119, is_fast=True).
        # AutoTokenizer(use_fast=True) returns the slow PhobertTokenizer (64001), which both
        # mismatches the saved checkpoint embedding and takes the wrong (slow) encode path.
        return RobertaTokenizerFast.from_pretrained(cfg["encoder"], add_prefix_space=True)
    return AutoTokenizer.from_pretrained(cfg["encoder"], use_fast=False)   # ViHealthBERT slow

def verify_roundtrip(cfg, tokenizer, n=5):
    tok = build_encoded("train", cfg, tokenizer)
    words_col, tags_col = get_words_labels("train"); ok = 0
    for i in range(n):
        wids, labs = tok.word_ids(batch_index=i), tok["labels"][i]
        rec, seen = [], set()
        for wid, lab in zip(wids, labs):
            if wid is None or wid in seen: continue
            seen.add(wid); rec.append(id2label[lab])
        syl, tags = words_col[i], tags_col[i]
        exp = realign_tags(syl, tags, segment_to_groups(syl))[1] if cfg["segment"] else tags
        assert rec == exp[:len(rec)], f"mismatch {i}: {rec} vs {exp[:len(rec)]}"
        ok += 1
    print(f"verify_roundtrip PASSED ({cfg['encoder']}) {ok}/{n}")

for _cfg in (CFG_PHO, CFG_VIH):
    verify_roundtrip(_cfg, load_tokenizer(_cfg))''')

code('''# §5 Model builder (Softmax head)
from transformers import AutoModelForTokenClassification

def build_softmax_model(cfg, tokenizer):
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["encoder"], num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id)
    model.resize_token_embeddings(len(tokenizer))   # top-level call: encoder-attr agnostic, same result/shape
    return model''')
```

- [ ] **Step 2: Regenerate + syntax-check**

Run (same authoritative PY check as Task 2 Step 4).
Expected: `9 cells, 0 syntax errors`.

- [ ] **Step 3: Document the Colab gate (no code change)**

Add nothing to code. Note in the commit body: on Colab this cell must print `verify_roundtrip PASSED` for **both** encoders before proceeding. A red gate here means a tokenizer/segmentation mismatch — stop and fix.

- [ ] **Step 4: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): verify gate (both tokenizers) + softmax model builder"
```

---

## Task 4: PhoBERT — load checkpoint + eval

**Files:**
- Modify: `<SCRATCH>/build_nb07.py`
- Modify (regenerate): the notebook

**Interfaces:**
- Consumes: `build_encoded, build_softmax_model, load_tokenizer, id2label, DEVICE` (earlier tasks).
- Produces: `evaluate_tags(true_tags, pred_tags) -> dict`, `decode_predictions(pred, label_ids, head) -> (T, P)`, `eval_softmax(cfg, tokenizer, model, split) -> dict`, and globals `tok_pho, model_pho, m_pho`.

- [ ] **Step 1: Append §7-eval-helpers + §6 PhoBERT cells**

```python
code('''# Eval helpers (reused from nb05)
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

def evaluate_tags(true_tags, pred_tags):
    return {"micro_f1": f1_score(true_tags, pred_tags, average="micro"),
            "macro_f1": f1_score(true_tags, pred_tags, average="macro"),
            "report":   classification_report(true_tags, pred_tags, digits=4)}

def decode_predictions(pred, label_ids, head):
    T, P = [], []
    for i, gold in enumerate(label_ids):
        p_seq = pred[i]; t_row, p_row, j = [], [], 0
        for k, g in enumerate(gold):
            if g == -100: continue
            t_row.append(id2label[g])
            p_row.append(id2label[p_seq[k]] if head == "softmax" else id2label[p_seq[j]]); j += 1
        T.append(t_row); P.append(p_row)
    return T, P

def _loader(cfg, tokenizer, split):
    tok = build_encoded(split, cfg, tokenizer)
    ds = TensorDataset(torch.tensor(tok["input_ids"]), torch.tensor(tok["attention_mask"]),
                       torch.tensor(tok["labels"]))
    return DataLoader(ds, batch_size=cfg["batch_size"]), tok

def eval_softmax(cfg, tokenizer, model, split="test"):
    dl, _ = _loader(cfg, tokenizer, split); model.eval(); T, P = [], []
    with torch.no_grad():
        for ids, mask, labs in dl:
            pred = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits.argmax(-1).cpu().tolist()
            t, p = decode_predictions(pred, labs.tolist(), "softmax"); T += t; P += p
    return evaluate_tags(T, P)''')

code('''# §6 PhoBERT — load existing checkpoint, evaluate on test (NO training)
tok_pho   = load_tokenizer(CFG_PHO)
model_pho = build_softmax_model(CFG_PHO, tok_pho).to(DEVICE)
state = torch.load(CFG_PHO["ckpt"], map_location=DEVICE)
model_pho.load_state_dict(state)
m_pho = eval_softmax(CFG_PHO, tok_pho, model_pho, "test")
print("PhoBERT  test micro_f1 =", round(m_pho["micro_f1"], 4), " macro_f1 =", round(m_pho["macro_f1"], 4))
assert m_pho["micro_f1"] > 0.5, "PhoBERT load/preproc mismatch (expected ~0.8+)"''')
```

- [ ] **Step 2: Regenerate + syntax-check** (authoritative PY check). Expected: `11 cells, 0 syntax errors`.

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): load PhoBERT checkpoint + test eval with sanity gate"
```

---

## Task 5: ViHealthBERT — train, save, eval

**Files:**
- Modify: `<SCRATCH>/build_nb07.py`
- Modify (regenerate): the notebook

**Interfaces:**
- Consumes: `build_encoded, build_softmax_model, load_tokenizer, eval_softmax, DEVICE, LR, EPOCHS, PATIENCE` (earlier).
- Produces: `train_softmax(cfg, tokenizer) -> (model, test_metrics)`, globals `tok_vih, model_vih, m_vih`.

- [ ] **Step 1: Append §7 training cell**

```python
code('''# §7 ViHealthBERT — full fine-tune Softmax (same config), save checkpoint, eval
from transformers import get_linear_schedule_with_warmup

def train_softmax(cfg, tokenizer):
    set_seed(42)
    dl_tr, _ = _loader_shuffle(cfg, tokenizer, "train")
    model = build_softmax_model(cfg, tokenizer).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(dl_tr) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    scaler = torch.cuda.amp.GradScaler()
    best, bad = -1.0, 0
    for ep in range(EPOCHS):
        model.train()
        for ids, mask, labs in dl_tr:
            ids, mask, labs = ids.to(DEVICE), mask.to(DEVICE), labs.to(DEVICE)
            with torch.cuda.amp.autocast():
                loss = model(ids, attention_mask=mask, labels=labs).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        mv = eval_softmax(cfg, tokenizer, model, "validation")
        print(f"ep{ep} val micro_f1={mv['micro_f1']:.4f}")
        torch.save(model.state_dict(), cfg["ckpt"])
        if mv["micro_f1"] > best:
            best, bad = mv["micro_f1"], 0; torch.save(model.state_dict(), cfg["ckpt"]+".best")
        else:
            bad += 1
            if bad >= PATIENCE: print("early stop"); break
    model.load_state_dict(torch.load(cfg["ckpt"]+".best", map_location=DEVICE))
    return model, eval_softmax(cfg, tokenizer, model, "test")

def _loader_shuffle(cfg, tokenizer, split):
    tok = build_encoded(split, cfg, tokenizer)
    ds = TensorDataset(torch.tensor(tok["input_ids"]), torch.tensor(tok["attention_mask"]),
                       torch.tensor(tok["labels"]))
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True), tok

tok_vih = load_tokenizer(CFG_VIH)
model_vih, m_vih = train_softmax(CFG_VIH, tok_vih)
print("ViHealthBERT test micro_f1 =", round(m_vih["micro_f1"], 4), " macro_f1 =", round(m_vih["macro_f1"], 4))
assert m_vih["micro_f1"] > 0.5, "ViHealthBERT training degenerate"''')
```

- [ ] **Step 2: Regenerate + syntax-check** (authoritative PY check). Expected: `12 cells, 0 syntax errors`.

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): train + save + eval ViHealthBERT softmax"
```

---

## Task 6: Comparison + error analysis

**Files:**
- Modify: `<SCRATCH>/build_nb07.py`
- Modify: `<SCRATCH>/test_helpers.py` (add span/error tests)
- Modify (regenerate): the notebook

**Interfaces:**
- Consumes: `m_pho, m_vih, CFG_PHO, CFG_VIH, tok_pho, tok_vih, model_pho, model_vih, eval_softmax, build_encoded, decode_predictions, get_words_labels` (earlier).
- Produces: `spans(tags) -> set[(start,end,type)]`, `classify_errors(true, pred, words) -> DataFrame`, `error_counts(df) -> dict`, `test_tag_seqs(cfg, tokenizer, model) -> (T, P, words)`.

- [ ] **Step 1: Add failing offline tests for `spans` / `classify_errors`**

Append to `<SCRATCH>/test_helpers.py`:

```python
import pandas as pd
def spans(tags):
    out, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-"):
            if start is not None: out.add((start, i, typ))
            start, typ = None, None
        if t.startswith("B-"): start, typ = i, t[2:]
        elif t.startswith("I-") and start is None: start, typ = i, t[2:]
    return out

def classify_errors(true_tags, pred_tags, words):
    rows = []
    for gold, pred, w in zip(true_tags, pred_tags, words):
        G, P = spans(gold), spans(pred)
        gbt = {(s,e):ty for (s,e,ty) in G}; pbt = {(s,e):ty for (s,e,ty) in P}
        sent = " ".join(w)
        for (s,e,ty) in G:
            if (s,e,ty) in P: continue
            if (s,e) in pbt:
                rows.append(dict(category="type", sentence=sent, gold_span=f"{ty}:{w[s:e]}", pred_span=f"{pbt[(s,e)]}:{w[s:e]}"))
            elif any(ps<e and s<pe and pty==ty for (ps,pe,pty) in P):
                rows.append(dict(category="boundary", sentence=sent, gold_span=f"{ty}:{w[s:e]}", pred_span="(overlap)"))
            else:
                rows.append(dict(category="missing", sentence=sent, gold_span=f"{ty}:{w[s:e]}", pred_span="-"))
        for (s,e,ty) in P:
            if (s,e,ty) in G: continue
            if (s,e) in gbt: continue
            if any(gs<e and s<ge and gty==ty for (gs,ge,gty) in G): continue
            rows.append(dict(category="spurious", sentence=sent, gold_span="-", pred_span=f"{ty}:{w[s:e]}"))
    return pd.DataFrame(rows)

def test_spans():
    assert spans(["B-AGE","I-AGE","O","B-LOC"]) == {(0,2,"AGE"),(3,4,"LOC")}

def test_classify_errors():
    cat = lambda T,P,W: set(classify_errors(T,P,W)["category"])
    assert cat([["B-AGE","I-AGE","O"]], [["B-AGE","O","O"]], [["a","b","c"]]) == {"boundary"}
    assert cat([["B-AGE","O"]], [["B-GEN","O"]], [["a","b"]]) == {"type"}
    assert cat([["B-AGE","O"]], [["O","B-AGE"]], [["a","b"]]) == {"missing","spurious"}

if __name__ == "__main__":
    test_encode_words(); test_encode_words_truncates()
    test_spans(); test_classify_errors()
    print("ALL helper tests PASS")
```

- [ ] **Step 2: Run to confirm pass**

Run: `python <SCRATCH>/test_helpers.py`
Expected: `ALL helper tests PASS`.

- [ ] **Step 3: Append §8 comparison + §9 error-analysis cells**

```python
code('''# §8 Comparison — entity-level micro/macro + per-entity report + CSV
import pandas as pd
df_cmp = pd.DataFrame([
    {"model":"PhoBERT",      "micro_f1":m_pho["micro_f1"], "macro_f1":m_pho["macro_f1"]},
    {"model":"ViHealthBERT", "micro_f1":m_vih["micro_f1"], "macro_f1":m_vih["macro_f1"]},
])
print(df_cmp.to_string(index=False))
df_cmp.to_csv(f'{RESULTS_DIR}/compare_phobert_vs_vihealthbert.csv', index=False)
print("\\n===== PhoBERT =====\\n",      m_pho["report"])
print("\\n===== ViHealthBERT =====\\n", m_vih["report"])''')

code('''# §9 Error analysis — boundary/type/missing/spurious per model + CSV + examples
def spans(tags):
    out, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-"):
            if start is not None: out.add((start, i, typ))
            start, typ = None, None
        if t.startswith("B-"): start, typ = i, t[2:]
        elif t.startswith("I-") and start is None: start, typ = i, t[2:]
    return out

def classify_errors(true_tags, pred_tags, words):
    rows = []
    for gold, pred, w in zip(true_tags, pred_tags, words):
        G, P = spans(gold), spans(pred)
        gbt = {(s,e):ty for (s,e,ty) in G}; pbt = {(s,e):ty for (s,e,ty) in P}
        sent = " ".join(w)
        for (s,e,ty) in G:
            if (s,e,ty) in P: continue
            if (s,e) in pbt:
                rows.append(dict(model="", category="type", sentence=sent, gold=f"{ty}:{w[s:e]}", pred=f"{pbt[(s,e)]}:{w[s:e]}"))
            elif any(ps<e and s<pe and pty==ty for (ps,pe,pty) in P):
                rows.append(dict(model="", category="boundary", sentence=sent, gold=f"{ty}:{w[s:e]}", pred="(overlap)"))
            else:
                rows.append(dict(model="", category="missing", sentence=sent, gold=f"{ty}:{w[s:e]}", pred="-"))
        for (s,e,ty) in P:
            if (s,e,ty) in G: continue
            if (s,e) in gbt or any(gs<e and s<ge and gty==ty for (gs,ge,gty) in G): continue
            rows.append(dict(model="", category="spurious", sentence=sent, gold="-", pred=f"{ty}:{w[s:e]}"))
    return pd.DataFrame(rows)

def test_tag_seqs(cfg, tokenizer, model):
    dl, tok = _loader(cfg, tokenizer, "test"); model.eval(); preds = []
    with torch.no_grad():
        for ids, mask, _ in dl:
            preds += model(ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits.argmax(-1).cpu().tolist()
    T, P = decode_predictions(preds, [tok["labels"][i] for i in range(len(preds))], "softmax")
    words = [realign_tags(s, t, segment_to_groups(s))[0] if cfg["segment"] else s
             for s, t in zip(*get_words_labels("test"))]
    return T, P, words

all_err = []
for name, cfg, tk, mdl in [("PhoBERT",CFG_PHO,tok_pho,model_pho), ("ViHealthBERT",CFG_VIH,tok_vih,model_vih)]:
    T, P, W = test_tag_seqs(cfg, tk, mdl)
    e = classify_errors(T, P, W); e["model"] = name
    print(f"{name}: {e['category'].value_counts().to_dict()}")
    for catg in ["boundary","type","missing","spurious"]:
        ex = e[e.category==catg].head(2)
        if len(ex): print(f"  [{catg}] e.g. ", list(ex["gold"])[:2], "->", list(ex["pred"])[:2])
    all_err.append(e)
err_df = pd.concat(all_err, ignore_index=True)
err_df.to_csv(f'{RESULTS_DIR}/error_analysis_phobert_vs_vihealthbert.csv', index=False)
print("wrote error_analysis CSV:", len(err_df), "errors")''')
```

- [ ] **Step 4: Regenerate + syntax-check** (authoritative PY check). Expected: `14 cells, 0 syntax errors`.

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): comparison table + boundary/type/missing/spurious error analysis"
```

---

## Task 7: Gradio app + span-highlight helper

**Files:**
- Modify: `<SCRATCH>/build_nb07.py`
- Modify: `<SCRATCH>/test_helpers.py` (add `tags_to_highlighted` test)
- Modify (regenerate): the notebook

**Interfaces:**
- Consumes: `CFG_PHO, CFG_VIH, load_tokenizer, build_softmax_model, encode_words, segment_to_groups, id2label, DEVICE` (earlier).
- Produces: `tags_to_highlighted(words, tags) -> list[(text, label_or_None)]`, `get_model(name) -> (cfg, tokenizer, model)`, `predict_sentence(text, encoder_name) -> list[tuple]`, a launched `gr.Interface`.

- [ ] **Step 1: Add failing offline test for `tags_to_highlighted`**

Append to `<SCRATCH>/test_helpers.py`:

```python
def tags_to_highlighted(words, tags):
    out, cur, cur_ty = [], [], None
    def flush():
        if cur: out.append((" ".join(cur).replace("_"," "), cur_ty))
    for w, t in zip(words, tags):
        if t == "O":
            flush(); cur, cur_ty = [], None; out.append((w.replace("_"," "), None))
        elif t.startswith("B-") or (t.startswith("I-") and t[2:] != cur_ty):
            flush(); cur, cur_ty = [w], t[2:]
        else:                                   # I- continuing same type
            cur.append(w)
    flush()
    return out

def test_tags_to_highlighted():
    out = tags_to_highlighted(["bệnh_nhân","75","tuổi"], ["O","B-AGE","I-AGE"])
    assert out == [("bệnh nhân", None), ("75 tuổi", "AGE")], out

if __name__ == "__main__":
    test_encode_words(); test_encode_words_truncates()
    test_spans(); test_classify_errors(); test_tags_to_highlighted()
    print("ALL helper tests PASS")
```

- [ ] **Step 2: Run to confirm pass**

Run: `python <SCRATCH>/test_helpers.py`
Expected: `ALL helper tests PASS`.

- [ ] **Step 3: Append §10 Gradio cell**

```python
code('''# §10 Gradio demo — pick encoder, type a sentence, see highlighted entities
import gradio as gr

def tags_to_highlighted(words, tags):
    out, cur, cur_ty = [], [], None
    def flush():
        if cur: out.append((" ".join(cur).replace("_"," "), cur_ty))
    for w, t in zip(words, tags):
        if t == "O":
            flush(); cur[:], cur_ty = [], None; out.append((w.replace("_"," "), None))
        elif t.startswith("B-") or (t.startswith("I-") and t[2:] != cur_ty):
            flush(); cur, cur_ty = [w], t[2:]
        else:
            cur.append(w)
    flush()
    return out

_MODELS = {}
def get_model(name):
    if name not in _MODELS:
        cfg = CFG_PHO if name == "PhoBERT" else CFG_VIH
        tk  = load_tokenizer(cfg)
        mdl = build_softmax_model(cfg, tk).to(DEVICE)
        ckpt = cfg["ckpt"] if name == "PhoBERT" else cfg["ckpt"]+".best"
        mdl.load_state_dict(torch.load(ckpt, map_location=DEVICE)); mdl.eval()
        _MODELS[name] = (cfg, tk, mdl)
    return _MODELS[name]

def predict_sentence(text, encoder_name):
    text = (text or "").strip()
    if not text: return []
    cfg, tk, mdl = get_model(encoder_name)
    syl = text.split()
    groups = segment_to_groups(syl)
    words = ["_".join(syl[i] for i in g) for g in groups]
    ids, mask, _, wids = encode_words(words, [0]*len(words), tk, cfg["max_len"])
    ids_t  = torch.tensor([ids]).to(DEVICE); mask_t = torch.tensor([mask]).to(DEVICE)
    with torch.no_grad():
        pred = mdl(ids_t, attention_mask=mask_t).logits.argmax(-1)[0].cpu().tolist()
    tags, seen = [], set()
    for k, wid in enumerate(wids):
        if wid is None or wid in seen: continue
        seen.add(wid); tags.append(id2label[pred[k]])
    tags = (tags + ["O"]*len(words))[:len(words)]
    return tags_to_highlighted(words, tags)

demo = gr.Interface(
    fn=predict_sentence,
    inputs=[gr.Textbox(label="Câu tiếng Việt (y khoa)", lines=2),
            gr.Dropdown(["PhoBERT","ViHealthBERT"], value="ViHealthBERT", label="Encoder")],
    outputs=gr.HighlightedText(label="Thực thể NER"),
    title="VietMed-NER — PhoBERT vs ViHealthBERT",
    examples=[["bệnh nhân nam 75 tuổi bị viêm phổi", "ViHealthBERT"]])
demo.launch(share=True)''')
```

- [ ] **Step 4: Regenerate + syntax-check** (authoritative PY check). Expected: `15 cells, 0 syntax errors`.

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER_07_phobert_vihealthbert_gradio.ipynb
git commit -m "feat(nb07): Gradio demo with encoder picker + highlighted entities"
```

---

## Self-Review

**Spec coverage:**
- §1 motivation/claim → Tasks 4–6 (compare two full-FT encoders). ✓
- §3 locked facts (paths, config, ckpt, tokenizer asymmetry) → Global Constraints + Tasks 1,3. ✓
- §5.3 dual-path tokenization → Task 2. ✓
- §5.5 PhoBERT reuse → Task 4. ✓
- §5.6 ViHealthBERT train+save → Task 5. ✓
- §5.7 comparison + error analysis (boundary/type/missing/spurious + CSV + examples) → Task 6. ✓
- §5.8 Gradio (dropdown, wseg, HighlightedText, share) → Task 7. ✓
- §6 eval parity (seqeval, PhoBERT fast path) → Tasks 2,4. ✓
- §7 success criteria 1–5 → gates: §T tests (T2/T6/T7), verify_roundtrip (T3), PhoBERT sanity assert (T4), ViHealthBERT assert (T5), comparison+error CSV (T6), Gradio launch (T7). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `build_encoded`/`_Encoded.word_ids`/`encode_words` signatures identical across Tasks 2,3,4,7. `eval_softmax`, `_loader`, `decode_predictions`, `spans`, `classify_errors`, `tags_to_highlighted` names match between builder cells and `test_helpers.py`. `cfg["ckpt"]` semantics: PhoBERT points at `...pt.best` (load directly); ViHealthBERT points at base `...pt` and training writes `+".best"` (Task 5/7 load `.best`). ✓

**Note on harness:** `torchcrf`/`torch_geometric` are NOT needed by nb07 (softmax only); the §0 install keeps `pytorch-crf` only for environment parity with nb05 and can be dropped if desired.
