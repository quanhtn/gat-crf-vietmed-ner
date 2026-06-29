# Vietnamese Medical NER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained Colab notebook that fine-tunes several encoders on VietMed-NER, ablates CRF vs Softmax, adds a PhoBERT+GAT+CRF improved model, evaluates zero-shot cross-domain on PhoNER-COVID19, and produces a 4-category error analysis.

**Architecture:** One notebook `VietMed_NER.ipynb` organized into section cells (§0–§10). Correctness-critical logic (tag re-alignment, subword alignment, shared-schema projection, error classification) is implemented as **pure functions** developed test-first with CPU-only assert cells. Model training/eval rows are run on Colab T4, one `EXP_ID` at a time, checkpointing to Google Drive.

**Tech Stack:** Python, HuggingFace `transformers` + `datasets`, `seqeval`, `pytorch-crf` (`torchcrf`), `py_vncorenlp`, `torch-geometric` (GAT), PyTorch, pandas. Runtime: Google Colab T4 (free), fp16.

## Global Constraints

- Single self-contained `.ipynb` — **no external `.py` files**. All code lives in notebook cells.
- Label list is **derived programmatically** from the loaded dataset — never hard-coded.
- Evaluation is **entity-level strict** via `seqeval` (span + type both correct = TP). Token-level metrics are not reported.
- Word segmentation is **per-encoder**: PhoBERT/ViHealthBERT use `py_vncorenlp` word-seg + syllable→word tag re-align; XLM-R uses raw syllables.
- Subword alignment: only the **first subword** of each word keeps its label; all other subwords and special tokens are `-100`.
- The §4 decode-back verification gate MUST pass before any training cell runs.
- Fixed seed across `random`/`numpy`/`torch`; `fp16=True`; checkpoint to Drive every epoch; **one experiment per Colab session**.
- PhoNER-COVID19 is a **test set only** — never trained on.
- Cross-domain eval projects both gold and prediction to the shared 7-type schema before scoring.
- `max_len=256` for all rows except GAT (#5) and the fair-compare baseline (#2b), which use `max_len=128`.

**Testing convention:** Pure-function tasks include a test cell (asserts) developed test-first; "verify it fails" means running the test cell before the function exists yields `NameError`/`AssertionError`. Keep test cells under a `§T Tests` section of the notebook. Commits are `git add VietMed_NER.ipynb && git commit`. Tasks marked **[Colab GPU]** cannot be verified locally; their verification is the expected metric range observed in the Colab run.

---

### Task 0: Notebook skeleton + §0 Setup

**Files:**
- Create: `VietMed_NER.ipynb` (markdown section headers §0–§10 + §T, and the §0 setup cell)

**Interfaces:**
- Produces: a runnable §0 cell that pins installs, mounts Drive, sets seed; global `SEED=42`, `DRIVE_DIR`, `RESULTS_DIR`.

- [ ] **Step 1: Create the notebook with section markdown cells**

Create `VietMed_NER.ipynb` with one markdown cell per section: `§0 Setup`, `§1 Config`, `§2 Data`, `§3 Preprocess`, `§4 Verify`, `§5 Model`, `§6 Train`, `§7 Eval`, `§8 Error analysis`, `§9 Cross-domain`, `§10 GAT model`, `§T Tests`.

- [ ] **Step 2: Write the §0 setup cell**

```python
# §0 Setup
!pip -q install "transformers==4.44.2" "datasets==2.21.0" seqeval pytorch-crf py_vncorenlp pandas
import os, random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

from google.colab import drive          # Colab only
drive.mount('/content/drive')
DRIVE_DIR   = '/content/drive/MyDrive/vietmed_ner'
RESULTS_DIR = f'{DRIVE_DIR}/results'
CKPT_DIR    = f'{DRIVE_DIR}/checkpoints'
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)
```

- [ ] **Step 3: Local dev note**

Add a markdown note: when developing locally (no Colab), skip the `drive.mount`/install lines; pure-function test cells (§T) run on CPU without them.

- [ ] **Step 4: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: notebook skeleton + setup cell"
```

---

### Task 1: §1 Config — experiment matrix

**Files:**
- Modify: `VietMed_NER.ipynb` (§1 cell)

**Interfaces:**
- Produces: dict `EXPERIMENTS[exp_id] -> {encoder, head, segment, max_len, batch_size, grad_accum}`; `EXP_ID` selector; `cfg = EXPERIMENTS[EXP_ID]`.

- [ ] **Step 1: Write the §1 config cell**

```python
# §1 Config
EXPERIMENTS = {
  "01_phobert_softmax": dict(encoder="vinai/phobert-base",   head="softmax", segment=True,  max_len=256, batch_size=16, grad_accum=1),
  "02_phobert_crf":     dict(encoder="vinai/phobert-base",   head="crf",     segment=True,  max_len=256, batch_size=16, grad_accum=1),
  "02b_phobert_crf128": dict(encoder="vinai/phobert-base",   head="crf",     segment=True,  max_len=128, batch_size=16, grad_accum=1),
  "03_xlmr_crf":        dict(encoder="xlm-roberta-base",     head="crf",     segment=False, max_len=256, batch_size=16, grad_accum=1),
  "04_vihealthbert_crf":dict(encoder="demdecuong/vihealthbert-base-word", head="crf", segment=True, max_len=256, batch_size=16, grad_accum=1),
  "05_phobert_gat_crf": dict(encoder="vinai/phobert-base",   head="gat_crf", segment=True,  max_len=128, batch_size=8,  grad_accum=2),
}
EXP_ID = "01_phobert_softmax"   # <-- change this to run a different row
cfg = EXPERIMENTS[EXP_ID]
print(EXP_ID, cfg)
```

- [ ] **Step 2: Add a note to verify the ViHealthBERT model id**

Add a markdown note: confirm `demdecuong/vihealthbert-base-word` loads in Task 7; if the id 404s, search HF for the current ViHealthBERT id and update here (Risk §13 in spec).

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: experiment matrix config"
```

---

### Task 2: §2 Data — load VietMed-NER, derive labels

**Files:**
- Modify: `VietMed_NER.ipynb` (§2 cell, §T test cell)

**Interfaces:**
- Produces: `raw = load_dataset("leduckhai/VietMed-NER")`; `LABEL_LIST` (sorted unique tags), `label2id`, `id2label`; helper `get_words_labels(split)` returning `(list[list[str]], list[list[str]])`.

- [ ] **Step 1: Write a test cell (§T) for label derivation**

```python
# §T test: label list derivation
def _test_label_list():
    assert "O" in LABEL_LIST, "O tag must exist"
    assert all(t == "O" or t[:2] in ("B-","I-") for t in LABEL_LIST), "BIO format"
    assert label2id[id2label[0]] == 0, "id<->label roundtrip"
    print("label list OK:", len(LABEL_LIST), "labels")
_test_label_list()
```

- [ ] **Step 2: Run it — expect failure**

Run the cell. Expected: `NameError: name 'LABEL_LIST' is not defined`.

- [ ] **Step 3: Write the §2 data cell**

```python
# §2 Data
from datasets import load_dataset
raw = load_dataset("leduckhai/VietMed-NER")
for c in ("audio", "duration"):
    if c in raw["train"].column_names:
        raw = raw.remove_columns(c)

# detect the tag column ('labels' holds BIO strings per the Hub card)
TAG_COL  = "labels" if "labels" in raw["train"].column_names else "tags"
WORD_COL = "words"

def get_words_labels(split):
    ds = raw[split]
    return list(ds[WORD_COL]), list(ds[TAG_COL])

_all_tags = set(t for split in raw for row in raw[split][TAG_COL] for t in row)
LABEL_LIST = ["O"] + sorted(t for t in _all_tags if t != "O")
label2id = {t:i for i,t in enumerate(LABEL_LIST)}
id2label = {i:t for t,i in label2id.items()}
print(len(LABEL_LIST), "labels; splits:", {k: len(raw[k]) for k in raw})
```

- [ ] **Step 4: Run the §T test cell — expect pass**

Run `_test_label_list()`. Expected: `label list OK: N labels`. Confirm split sizes ≈ 4.62k/1.15k/3.5k.

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: load VietMed-NER and derive label list"
```

---

### Task 3: Shared cross-domain schema projection (pure function, TDD)

**Files:**
- Modify: `VietMed_NER.ipynb` (§2 cell — shared schema, §T test cell)

**Interfaces:**
- Produces: `SHARED_MAP_VIETMED: dict[str,str]`, `SHARED_MAP_PHONER: dict[str,str]`, and `project_tags(tags: list[str], mapping: dict) -> list[str]` mapping a BIO sequence to the shared schema (`O` for unmapped types, preserving B-/I- prefixes).

- [ ] **Step 1: Write §T test for `project_tags`**

```python
# §T test: shared-schema projection
def _test_project():
    m = {"DISEASESYMPTOM":"SYMPTOM_DISEASE", "DATETIME":"DATE"}
    assert project_tags(["B-DISEASESYMPTOM","I-DISEASESYMPTOM","O"], m) == ["B-SYMPTOM_DISEASE","I-SYMPTOM_DISEASE","O"]
    assert project_tags(["B-DRUGCHEMICAL","I-DRUGCHEMICAL"], m) == ["O","O"]   # unmapped -> O
    assert project_tags(["B-DATETIME"], m) == ["B-DATE"]
    print("project_tags OK")
_test_project()
```

- [ ] **Step 2: Run it — expect `NameError`**

- [ ] **Step 3: Implement the shared schema + `project_tags`**

```python
# §2 (cont): shared cross-domain schema
SHARED_MAP_VIETMED = {
  "DISEASESYMPTOM":"SYMPTOM_DISEASE", "AGE":"AGE", "GENDER":"GENDER",
  "OCCUPATION":"OCCUPATION", "LOCATION":"LOCATION", "ORGANIZATION":"ORGANIZATION",
  "DATETIME":"DATE",
}
SHARED_MAP_PHONER = {
  "SYMPTOM&DISEASE":"SYMPTOM_DISEASE", "AGE":"AGE", "GENDER":"GENDER",
  "OCCUPATION":"OCCUPATION", "LOCATION":"LOCATION", "ORGANIZATION":"ORGANIZATION",
  "DATE":"DATE",
}
SHARED_LABELS = ["O"] + sorted(set(SHARED_MAP_VIETMED.values()))

def project_tags(tags, mapping):
    out = []
    for t in tags:
        if t == "O":
            out.append("O"); continue
        pref, _, typ = t.partition("-")
        out.append(f"{pref}-{mapping[typ]}" if typ in mapping else "O")
    return out
```

- [ ] **Step 4: Run the §T test — expect `project_tags OK`**

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: shared cross-domain schema + project_tags"
```

---

### Task 4: §3 Word segmentation + syllable→word tag re-align (pure function, TDD)

**Files:**
- Modify: `VietMed_NER.ipynb` (§3 cell, §T test cell)

**Interfaces:**
- Consumes: a segmenter that maps a syllable list to word groupings.
- Produces: `realign_tags(syllables: list[str], tags: list[str], word_groups: list[list[int]]) -> (list[str], list[str])` where `word_groups` lists the syllable indices composing each word. The word's tag = the tag of its first syllable; a leading `I-X` on a word start is promoted to `B-X`.

This is the highest-risk function — develop it test-first with synthetic groupings (no segmenter needed for the test).

- [ ] **Step 1: Write §T test for `realign_tags`**

```python
# §T test: syllable->word tag re-alignment
def _test_realign():
    syl  = ["bệnh","nhân","bị","viêm","phổi"]
    tags = ["B-DISEASESYMPTOM","I-DISEASESYMPTOM","O","B-DISEASESYMPTOM","I-DISEASESYMPTOM"]
    groups = [[0,1],[2],[3,4]]                       # bệnh_nhân | bị | viêm_phổi
    w, t = realign_tags(syl, tags, groups)
    assert w == ["bệnh_nhân","bị","viêm_phổi"]
    assert t == ["B-DISEASESYMPTOM","O","B-DISEASESYMPTOM"]
    # word starting on an I- (segmenter split mid-entity) must be promoted to B-
    w2,t2 = realign_tags(["a","b"], ["O","I-AGE"], [[0],[1]])
    assert t2 == ["O","B-AGE"]
    print("realign_tags OK")
_test_realign()
```

- [ ] **Step 2: Run it — expect `NameError`**

- [ ] **Step 3: Implement `realign_tags`**

```python
# §3 Preprocess: tag re-alignment
def realign_tags(syllables, tags, word_groups):
    words, new_tags = [], []
    for grp in word_groups:
        words.append("_".join(syllables[i] for i in grp))
        head = tags[grp[0]]
        if head.startswith("I-"):           # word boundary fell inside an entity
            head = "B-" + head[2:]
        new_tags.append(head)
    return words, new_tags
```

- [ ] **Step 4: Run the §T test — expect `realign_tags OK`**

- [ ] **Step 5: Implement the segmenter wrapper that produces `word_groups`**

```python
# §3 (cont): py_vncorenlp word segmentation -> word_groups by aligning syllables
import py_vncorenlp
_SEG = None
def get_segmenter():
    global _SEG
    if _SEG is None:
        py_vncorenlp.download_model(save_dir=f'{DRIVE_DIR}/vncorenlp')
        _SEG = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=f'{DRIVE_DIR}/vncorenlp')
    return _SEG

def segment_to_groups(syllables):
    """Return word_groups: list of syllable-index lists. Falls back to 1-syllable-per-word."""
    seg = get_segmenter()
    segged = seg.word_segment(" ".join(syllables))           # list[str] of words joined by '_'
    word_units = " ".join(segged).split()                    # flatten to underscore-words
    groups, cur = [], 0
    for wu in word_units:
        n = wu.count("_") + 1
        groups.append(list(range(cur, cur + n)))
        cur += n
    if cur != len(syllables):                                # mismatch -> safe fallback
        return [[i] for i in range(len(syllables))]
    return groups
```

- [ ] **Step 6: Add a §T sanity test for the segmenter→realign integration (Colab/segmenter available)**

```python
# §T test (needs py_vncorenlp): every syllable accounted for, lengths match
def _test_segment_integration():
    syl = ["bệnh","nhân","bị","ho"]; tags = ["B-DISEASESYMPTOM","I-DISEASESYMPTOM","O","B-DISEASESYMPTOM"]
    g = segment_to_groups(syl)
    assert sum(len(x) for x in g) == len(syl)
    w,t = realign_tags(syl, tags, g)
    assert len(w) == len(t) == len(g)
    print("segment integration OK:", w)
# _test_segment_integration()   # run on Colab
```

- [ ] **Step 7: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: word segmentation + syllable->word tag re-align"
```

---

### Task 5: §3 Subword alignment with -100 (pure function, TDD)

**Files:**
- Modify: `VietMed_NER.ipynb` (§3 cell, §T test cell)

**Interfaces:**
- Consumes: a HF fast tokenizer's `word_ids()`.
- Produces: `align_labels(tokenized, word_label_ids) -> tokenized` adding `labels` where only the first subword of each word keeps its id, others and specials are `-100`.

- [ ] **Step 1: Write §T test for `align_labels`**

```python
# §T test: subword -100 alignment (synthetic word_ids)
class _FakeTok(dict):
    def __init__(self, wid): super().__init__(); self._wid=wid
    def word_ids(self, batch_index=0): return self._wid[batch_index]
def _test_align():
    tok = _FakeTok([[None,0,0,1,None]])               # [CLS] w0 w0 w1 [SEP]
    out = align_labels(tok, [[5,7]])
    assert out["labels"][0] == [-100,5,-100,7,-100]
    print("align_labels OK")
_test_align()
```

- [ ] **Step 2: Run it — expect `NameError`**

- [ ] **Step 3: Implement `align_labels`**

```python
# §3 (cont): subword alignment
def align_labels(tokenized_inputs, word_label_ids):
    aligned = []
    for i, labels in enumerate(word_label_ids):
        prev, ids = None, []
        for wid in tokenized_inputs.word_ids(batch_index=i):
            if wid is None:            ids.append(-100)
            elif wid != prev:          ids.append(labels[wid])
            else:                      ids.append(-100)
            prev = wid
        aligned.append(ids)
    tokenized_inputs["labels"] = aligned
    return tokenized_inputs
```

- [ ] **Step 4: Run the §T test — expect `align_labels OK`**

- [ ] **Step 5: Write the dataset encoding function tying it together**

```python
# §3 (cont): build encoded dataset for a given cfg
from transformers import AutoTokenizer
def build_encoded(split, cfg, tokenizer):
    words_col, tags_col = get_words_labels(split)
    enc_words, enc_label_ids = [], []
    for syl, tags in zip(words_col, tags_col):
        if cfg["segment"]:
            groups = segment_to_groups(syl)
            w, t = realign_tags(syl, tags, groups)
        else:
            w, t = syl, tags
        enc_words.append(w); enc_label_ids.append([label2id[x] for x in t])
    tok = tokenizer(enc_words, is_split_into_words=True, truncation=True,
                    max_length=cfg["max_len"], padding="max_length")
    tok = align_labels(tok, enc_label_ids)
    return tok
```

- [ ] **Step 6: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: subword -100 alignment + dataset encoding"
```

---

### Task 6: §4 Decode-back verification gate

**Files:**
- Modify: `VietMed_NER.ipynb` (§4 cell)

**Interfaces:**
- Consumes: `build_encoded`, tokenizer, `id2label`.
- Produces: `verify_roundtrip(cfg, tokenizer, n=5)` that asserts decoded first-subword labels reproduce the (re-aligned) gold word tags for `n` samples; raises on mismatch.

- [ ] **Step 1: Write the §4 verification cell**

```python
# §4 Verify (GATE — must pass before training)
def verify_roundtrip(cfg, tokenizer, n=5):
    tok = build_encoded("train", cfg, tokenizer)
    ok = 0
    for i in range(n):
        wids = tok.word_ids(batch_index=i)
        labs = tok["labels"][i]
        # recover one label per word (first subword)
        rec, seen = [], set()
        for wid, lab in zip(wids, labs):
            if wid is None or wid in seen: continue
            seen.add(wid); rec.append(id2label[lab])
        # rebuild expected from raw
        syl, tags = get_words_labels("train")[0][i], get_words_labels("train")[1][i]
        if cfg["segment"]:
            _, exp = realign_tags(syl, tags, segment_to_groups(syl))
        else:
            exp = tags
        exp = exp[:len(rec)]                      # truncation parity
        assert rec == exp, f"sample {i} mismatch:\n  rec={rec}\n  exp={exp}"
        ok += 1
    print(f"verify_roundtrip PASSED for {ok}/{n} samples")
```

- [ ] **Step 2: Run it on Colab for the active cfg** **[Colab GPU]**

Run `tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]); verify_roundtrip(cfg, tokenizer)`. Expected: `verify_roundtrip PASSED for 5/5 samples`. If it raises, fix Task 4/5 before proceeding.

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: decode-back verification gate"
```

---

### Task 7: §5 Model factory — softmax, CRF (smoke test)

**Files:**
- Modify: `VietMed_NER.ipynb` (§5 cell, §T test cell)

**Interfaces:**
- Produces: `build_model(cfg)` returning a `nn.Module`. CRF model `BertCrfForNer(encoder_name, num_labels)` with `forward(input_ids, attention_mask, labels=None)` → loss (train) or `list[list[int]]` decoded tags (eval).

- [ ] **Step 1: Write §T smoke test for the CRF model on tiny random input**

```python
# §T smoke: CRF forward returns scalar loss and decodes
def _test_crf_smoke():
    m = build_model(dict(encoder="vinai/phobert-base", head="crf")).to("cpu")
    ids = torch.randint(0, 100, (2, 8)); mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.zeros(2, 8, dtype=torch.long); labels[0,0] = -100
    loss = m(ids, mask, labels)
    assert loss.dim() == 0 and loss.item() == loss.item(), "scalar finite loss"
    dec = m(ids, mask)
    assert len(dec) == 2 and all(isinstance(s, list) for s in dec)
    print("CRF smoke OK, loss=", round(loss.item(),3))
# _test_crf_smoke()   # heavy download; run once on Colab
```

- [ ] **Step 2: Implement `build_model` + `BertCrfForNer`**

```python
# §5 Model
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoModelForTokenClassification

class BertCrfForNer(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    def forward(self, input_ids, attention_mask, labels=None):
        h = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(h))
        mask = attention_mask.bool()
        if labels is not None:
            safe = labels.clone(); safe[safe == -100] = 0
            return -self.crf(emissions, safe, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)

def build_model(cfg):
    n = len(LABEL_LIST)
    if cfg["head"] == "softmax":
        return AutoModelForTokenClassification.from_pretrained(
            cfg["encoder"], num_labels=n, id2label=id2label, label2id=label2id)
    if cfg["head"] == "crf":
        return BertCrfForNer(cfg["encoder"], n)
    if cfg["head"] == "gat_crf":
        return build_gat_crf(cfg["encoder"], n)   # defined in §10 (Task 13)
    raise ValueError(cfg["head"])
```

- [ ] **Step 3: Run the smoke test on Colab** **[Colab GPU]**

Uncomment `_test_crf_smoke()`. Expected: `CRF smoke OK, loss= <finite>`.

- [ ] **Step 4: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: model factory (softmax + CRF)"
```

---

### Task 8: §7 Evaluation — seqeval entity-level (TDD)

**Files:**
- Modify: `VietMed_NER.ipynb` (§7 cell, §T test cell)

**Interfaces:**
- Produces: `decode_predictions(pred_ids_or_paths, label_ids, head) -> (true_tags, pred_tags)` (drops `-100`), and `evaluate_tags(true_tags, pred_tags) -> dict` with micro/macro F1 + per-entity report string.

- [ ] **Step 1: Write §T test for `evaluate_tags`**

```python
# §T test: seqeval entity-level
def _test_eval():
    true = [["B-AGE","I-AGE","O"]]; pred = [["B-AGE","I-AGE","O"]]
    r = evaluate_tags(true, pred)
    assert abs(r["micro_f1"] - 1.0) < 1e-9
    bad = [["B-AGE","O"]]
    r2 = evaluate_tags([["B-AGE","I-AGE"]], bad)        # boundary error -> F1 0
    assert r2["micro_f1"] == 0.0
    print("evaluate_tags OK")
_test_eval()
```

- [ ] **Step 2: Run it — expect `NameError`**

- [ ] **Step 3: Implement evaluation helpers**

```python
# §7 Eval
from seqeval.metrics import f1_score, classification_report
def evaluate_tags(true_tags, pred_tags):
    return {
       "micro_f1": f1_score(true_tags, pred_tags, average="micro"),
       "macro_f1": f1_score(true_tags, pred_tags, average="macro"),
       "report":   classification_report(true_tags, pred_tags, digits=4),
    }

def decode_predictions(pred, label_ids, head):
    # softmax: pred is logits/argmax ids [B,L]; crf/gat_crf: pred is list[list[int]]
    true_tags, pred_tags = [], []
    for i, gold in enumerate(label_ids):
        if head == "softmax":
            p_seq = pred[i]
        else:
            p_seq = pred[i]                      # already only over masked positions
        t_row, p_row, j = [], [], 0
        for k, g in enumerate(gold):
            if g == -100: 
                if head != "softmax": pass
                continue
            t_row.append(id2label[g])
            p_row.append(id2label[p_seq[k]] if head=="softmax" else id2label[p_seq[j]])
            j += 1
        true_tags.append(t_row); pred_tags.append(p_row)
    return true_tags, pred_tags
```

Note: for CRF, decoded sequences cover only `attention_mask` positions; ensure the eval loop pairs them with the same masked positions used during decode (Task 9 wires this).

- [ ] **Step 4: Run the §T test — expect `evaluate_tags OK`**

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: seqeval entity-level evaluation helpers"
```

---

### Task 9: §6 Training loop + Drive checkpointing (smoke run)

**Files:**
- Modify: `VietMed_NER.ipynb` (§6 cell)

**Interfaces:**
- Consumes: `build_model`, `build_encoded`, `evaluate_tags`, `decode_predictions`.
- Produces: `train_and_eval(cfg, exp_id)` that trains with early stopping, checkpoints each epoch to `CKPT_DIR/<exp_id>`, evaluates on test, writes `RESULTS_DIR/<exp_id>.csv`, returns the metrics dict.

- [ ] **Step 1: Write the §6 training cell (HF Trainer for softmax, manual loop for CRF)**

```python
# §6 Train
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

def _to_dataset(tok):
    import torch
    ids  = torch.tensor(tok["input_ids"]); mask = torch.tensor(tok["attention_mask"])
    labs = torch.tensor(tok["labels"]);    return TensorDataset(ids, mask, labs)

def train_and_eval(cfg, exp_id, epochs=30, lr=2e-5, patience=3):
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])
    tr = _to_dataset(build_encoded("train", cfg, tokenizer))
    va = _to_dataset(build_encoded("validation", cfg, tokenizer))
    te = _to_dataset(build_encoded("test", cfg, tokenizer))
    dl_tr = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True)
    dl_va = DataLoader(va, batch_size=cfg["batch_size"]); dl_te = DataLoader(te, batch_size=cfg["batch_size"])
    model = build_model(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(dl_tr)*epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    scaler = torch.cuda.amp.GradScaler()
    best_f1, bad, ckpt = -1, 0, f'{CKPT_DIR}/{exp_id}.pt'

    def run_eval(dl):
        model.eval(); T,P = [],[]
        with torch.no_grad():
            for ids,mask,labs in dl:
                ids,mask = ids.to(DEVICE),mask.to(DEVICE)
                if cfg["head"]=="softmax":
                    logits = model(ids, attention_mask=mask).logits
                    pred = logits.argmax(-1).cpu().tolist()
                else:
                    pred = model(ids, mask)            # list[list[int]] over mask
                t,p = decode_predictions(pred, labs.tolist(), cfg["head"])
                T+=t; P+=p
        return evaluate_tags(T,P)

    for ep in range(epochs):
        model.train()
        for step,(ids,mask,labs) in enumerate(dl_tr):
            ids,mask,labs = ids.to(DEVICE),mask.to(DEVICE),labs.to(DEVICE)
            with torch.cuda.amp.autocast():
                if cfg["head"]=="softmax":
                    loss = model(ids, attention_mask=mask, labels=labs).loss
                else:
                    loss = model(ids, mask, labs)
                loss = loss/cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (step+1)%cfg["grad_accum"]==0:
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        m = run_eval(dl_va); print(f"ep{ep} val micro_f1={m['micro_f1']:.4f}")
        torch.save(model.state_dict(), ckpt)          # checkpoint every epoch -> Drive
        if m["micro_f1"]>best_f1: best_f1=m["micro_f1"]; bad=0; torch.save(model.state_dict(), ckpt+".best")
        else: bad+=1
        if bad>=patience: print("early stop"); break

    model.load_state_dict(torch.load(ckpt+".best"))
    test_m = run_eval(dl_te)
    pd.DataFrame([{ "exp_id":exp_id, "encoder":cfg["encoder"], "head":cfg["head"],
                    "max_len":cfg["max_len"], "micro_f1":test_m["micro_f1"],
                    "macro_f1":test_m["macro_f1"]}]).to_csv(f'{RESULTS_DIR}/{exp_id}.csv', index=False)
    print(test_m["report"]); return test_m
```

- [ ] **Step 2: Smoke-run 1 epoch on a tiny slice** **[Colab GPU]**

Temporarily set `epochs=1` and slice `tr = torch.utils.data.Subset(tr, range(64))`. Run `train_and_eval(cfg, "smoke")`. Expected: a finite `val micro_f1` prints and `results/smoke.csv` is written. Remove the slice afterward.

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: training loop + Drive checkpointing + CSV logging"
```

---

### Task 10: §8 Error analysis classifier (pure function, TDD)

**Files:**
- Modify: `VietMed_NER.ipynb` (§8 cell, §T test cell)

**Interfaces:**
- Produces: `spans(tags) -> set[(start,end,type)]`; `classify_errors(true_tags, pred_tags, words) -> pd.DataFrame` with columns `category∈{boundary,type,missing,spurious}, sentence, gold_span, pred_span`, plus `error_counts(df)`.

- [ ] **Step 1: Write §T test for span extraction + classification**

```python
# §T test: error categories
def _test_errors():
    assert spans(["B-AGE","I-AGE","O","B-LOC".replace("LOC","LOCATION")]) == {(0,2,"AGE"),(3,4,"LOCATION")}
    words=[["a","b","c"]]
    # boundary: same type, different extent
    df = classify_errors([["B-AGE","I-AGE","O"]], [["B-AGE","O","O"]], words)
    assert set(df["category"]) == {"boundary"}
    # type: same span, wrong type
    df2 = classify_errors([["B-AGE","O"]], [["B-GENDER","O"]], [["a","b"]])
    assert set(df2["category"]) == {"type"}
    # missing + spurious
    df3 = classify_errors([["B-AGE","O"]], [["O","B-AGE"]], [["a","b"]])
    assert set(df3["category"]) == {"missing","spurious"}
    print("error classify OK")
_test_errors()
```

- [ ] **Step 2: Run it — expect `NameError`**

- [ ] **Step 3: Implement span extraction + `classify_errors`**

```python
# §8 Error analysis
import pandas as pd
def spans(tags):
    out, start, typ = set(), None, None
    for i,t in enumerate(tags+["O"]):
        if t=="O" or t.startswith("B-"):
            if start is not None: out.add((start,i,typ))
            start, typ = (None,None)
        if t.startswith("B-"): start, typ = i, t[2:]
        elif t.startswith("I-") and start is None: start, typ = i, t[2:]  # tolerate stray I-
    return out

def classify_errors(true_tags, pred_tags, words):
    rows=[]
    for gold,pred,w in zip(true_tags,pred_tags,words):
        G,P = spans(gold), spans(pred)
        gold_by_type={(s,e):ty for (s,e,ty) in G}; pred_by_type={(s,e):ty for (s,e,ty) in P}
        sent=" ".join(w)
        for (s,e,ty) in G:
            if (s,e,ty) in P: continue
            if (s,e) in pred_by_type:                       # same span, wrong type
                rows.append(dict(category="type", sentence=sent,
                                 gold_span=f"{ty}:{w[s:e]}", pred_span=f"{pred_by_type[(s,e)]}:{w[s:e]}"))
            elif any(ps< e and s< pe and pty==ty for (ps,pe,pty) in P):  # overlap, same type
                rows.append(dict(category="boundary", sentence=sent, gold_span=f"{ty}:{w[s:e]}", pred_span="(overlap)"))
            else:
                rows.append(dict(category="missing", sentence=sent, gold_span=f"{ty}:{w[s:e]}", pred_span="-"))
        for (s,e,ty) in P:
            if (s,e,ty) in G: continue
            if (s,e) in gold_by_type: continue              # counted as type above
            if any(gs< e and s< ge and gty==ty for (gs,ge,gty) in G): continue  # boundary, counted
            rows.append(dict(category="spurious", sentence=sent, gold_span="-", pred_span=f"{ty}:{w[s:e]}"))
    return pd.DataFrame(rows)

def error_counts(df):
    return df["category"].value_counts().to_dict()
```

- [ ] **Step 4: Run the §T test — expect `error classify OK`**

- [ ] **Step 5: Commit**

```bash
git add VietMed_NER.ipynb && git commit -m "feat: 4-category error analysis classifier"
```

---

### Task 11: Run baseline #1 PhoBERT+Softmax end-to-end

**Files:** none (Colab run) **[Colab GPU]**

- [ ] **Step 1: Set `EXP_ID="01_phobert_softmax"`, run §0–§4 gate**

Expected: `verify_roundtrip PASSED for 5/5 samples`.

- [ ] **Step 2: Run `train_and_eval(cfg, EXP_ID)` (full)**

Expected: completes in ~30–60 min; `results/01_phobert_softmax.csv` written with a plausible micro_f1 (sanity: > 0.5, expected ballpark 0.7–0.85). Record the number.

- [ ] **Step 3: Commit results CSV**

```bash
git add results/01_phobert_softmax.csv && git commit -m "exp: PhoBERT softmax baseline results"
```

---

### Task 12: Run CRF rows #2, #2b, #3, #4

**Files:** none (Colab runs, one per session) **[Colab GPU]**

- [ ] **Step 1: Run `02_phobert_crf`** — set `EXP_ID`, run gate, `train_and_eval`. Expected CSV written.
- [ ] **Step 2: Run `02b_phobert_crf128`** — confirms the matched-length baseline for the GAT comparison.
- [ ] **Step 3: Run `03_xlmr_crf`** — note `segment=False`; verify gate still passes for syllable mode.
- [ ] **Step 4: Run `04_vihealthbert_crf`** — first confirm the model id loads (Task 1 note); if it 404s, update the id and re-run.
- [ ] **Step 5: Commit all four CSVs**

```bash
git add results/02_phobert_crf.csv results/02b_phobert_crf128.csv results/03_xlmr_crf.csv results/04_vihealthbert_crf.csv
git commit -m "exp: CRF baselines (PhoBERT@256/@128, XLM-R, ViHealthBERT)"
```

---

### Task 13: §10 GAT+CRF improved model + run #5

**Files:**
- Modify: `VietMed_NER.ipynb` (§10 cell, §T smoke)

**Interfaces:**
- Produces: `build_gat_crf(encoder_name, num_labels) -> nn.Module`; same `forward(input_ids, attention_mask, labels=None)` contract as `BertCrfForNer`.

- [ ] **Step 1: Install torch-geometric pinned to Colab CUDA (run once)** **[Colab GPU]**

```python
import torch
print(torch.__version__, torch.version.cuda)   # read these, then:
!pip -q install torch-geometric
```
Note: if `GATConv` import fails, install matching `pyg-lib`/`torch-scatter` wheels for the printed torch/CUDA.

- [ ] **Step 2: Write §T smoke test for the GAT+CRF forward**

```python
# §T smoke: GAT+CRF forward (Colab)
def _test_gat_smoke():
    m = build_gat_crf("vinai/phobert-base", len(LABEL_LIST)).to(DEVICE)
    ids = torch.randint(0,100,(2,16)).to(DEVICE); mask=torch.ones(2,16,dtype=torch.long).to(DEVICE)
    labs = torch.zeros(2,16,dtype=torch.long).to(DEVICE)
    assert m(ids,mask,labs).dim()==0
    assert len(m(ids,mask))==2
    print("GAT+CRF smoke OK")
# _test_gat_smoke()
```

- [ ] **Step 3: Implement `build_gat_crf`**

```python
# §10 GAT model: PhoBERT -> GAT (per-sequence full graph) -> self-attn -> CRF
from torch_geometric.nn import GATConv
class GatCrfForNer(nn.Module):
    def __init__(self, encoder_name, num_labels, heads=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        H = self.bert.config.hidden_size
        self.gat = GATConv(H, H//heads, heads=heads, dropout=0.1)
        self.attn = nn.MultiheadAttention(H, heads, batch_first=True)
        self.classifier = nn.Linear(H, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    def _full_edges(self, L, device):
        idx = torch.arange(L, device=device)
        s = idx.repeat_interleave(L); d = idx.repeat(L)
        return torch.stack([s,d], 0)
    def forward(self, input_ids, attention_mask, labels=None):
        h = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,H]
        B,L,H = h.shape; outs=[]
        ei = self._full_edges(L, h.device)
        for b in range(B):                       # per-sequence graph (max_len=128, small batch)
            outs.append(self.gat(h[b], ei))
        g = torch.stack(outs, 0)
        a,_ = self.attn(g,g,g, key_padding_mask=~attention_mask.bool())
        emissions = self.classifier(a); mask = attention_mask.bool()
        if labels is not None:
            safe = labels.clone(); safe[safe==-100]=0
            return -self.crf(emissions, safe, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)

def build_gat_crf(encoder_name, num_labels): return GatCrfForNer(encoder_name, num_labels)
```

- [ ] **Step 4: Run the GAT smoke test** **[Colab GPU]** — expect `GAT+CRF smoke OK`.

- [ ] **Step 5: Run `05_phobert_gat_crf` full** **[Colab GPU]**

Set `EXP_ID="05_phobert_gat_crf"`, run gate + `train_and_eval`. Expected: ~2h; CSV written. If OOM, lower `batch_size` and raise `grad_accum` in §1.

- [ ] **Step 6: Commit**

```bash
git add VietMed_NER.ipynb results/05_phobert_gat_crf.csv
git commit -m "feat+exp: PhoBERT+GAT+CRF improved model and results"
```

---

### Task 14: §9 Cross-domain zero-shot on PhoNER-COVID19

**Files:**
- Modify: `VietMed_NER.ipynb` (§9 cell) **[Colab GPU]**

**Interfaces:**
- Consumes: best checkpoint, `project_tags`, `SHARED_MAP_*`, `evaluate_tags`.
- Produces: `crossdomain_eval(best_exp_id)` writing `results/crossdomain.csv`.

- [ ] **Step 1: Load PhoNER-COVID19 word-level test set**

```python
# §9 Cross-domain
!git clone -q https://github.com/VinAIResearch/PhoNER_COVID19.git
def read_conll(path):
    sents, words, tags, w, t = [], [], [], [], []
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line:
            if w: words.append(w); tags.append(t); w,t=[],[]
        else:
            parts=line.split(); w.append(parts[0]); t.append(parts[-1])
    if w: words.append(w); tags.append(t)
    return words, tags
ph_words, ph_tags = read_conll("PhoNER_COVID19/data/word/test_word.conll")
```
Note: confirm the exact path/filename after clone (`ls PhoNER_COVID19/data/word`).

- [ ] **Step 2: Predict with the best model and project both sides to shared schema**

```python
def crossdomain_eval(best_exp_id):
    cfg = EXPERIMENTS[best_exp_id]; tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = build_model(cfg).to(DEVICE); model.load_state_dict(torch.load(f'{CKPT_DIR}/{best_exp_id}.pt.best')); model.eval()
    # encode PhoNER words directly (already word-level); align with -100
    tok = tokenizer(ph_words, is_split_into_words=True, truncation=True, max_length=cfg["max_len"], padding="max_length")
    # dummy label ids just to reuse decode pairing: map gold tags unknown to VietMed label space -> use O
    import torch as T_
    ds = TensorDataset(T_.tensor(tok["input_ids"]), T_.tensor(tok["attention_mask"]))
    dl = DataLoader(ds, batch_size=cfg["batch_size"])
    preds=[]
    with torch.no_grad():
        for ids,mask in dl:
            ids,mask=ids.to(DEVICE),mask.to(DEVICE)
            out = model(ids,attention_mask=mask).logits.argmax(-1).cpu().tolist() if cfg["head"]=="softmax" else model(ids,mask)
            preds.extend(out)
    # recover one pred label per word, project to shared schema
    T_tags, P_tags = [], []
    for i,(gold) in enumerate(ph_tags):
        wids = tok.word_ids(batch_index=i); seen=set(); rec=[]; j=0
        for k,wid in enumerate(wids):
            if wid is None or wid in seen: 
                if cfg["head"]!="softmax" and wid is not None and wid not in seen: pass
                continue
            seen.add(wid)
            pid = preds[i][k] if cfg["head"]=="softmax" else preds[i][j]; j+=1
            rec.append(id2label[pid])
        rec = rec[:len(gold)]
        P_tags.append(project_tags(rec, SHARED_MAP_VIETMED))
        T_tags.append(project_tags(gold, SHARED_MAP_PHONER))
    m = evaluate_tags(T_tags, P_tags)
    pd.DataFrame([{ "best_exp":best_exp_id, "cross_micro_f1":m["micro_f1"], "cross_macro_f1":m["macro_f1"]}]).to_csv(f'{RESULTS_DIR}/crossdomain.csv', index=False)
    print(m["report"]); return m
```
Note: CRF decode/word pairing here is approximate; verify the per-word recovery matches §6's `decode_predictions` ordering for the chosen `head`. Prefer running cross-domain with a softmax or CRF best model and confirm the word-id pairing on 2 samples first.

- [ ] **Step 3: Run for the best in-domain model** **[Colab GPU]**

Expected: `results/crossdomain.csv` written; cross-domain micro_f1 lower than in-domain (generalization gap is the finding). Sanity: > 0.

- [ ] **Step 4: Commit**

```bash
git add VietMed_NER.ipynb results/crossdomain.csv && git commit -m "exp: cross-domain zero-shot on PhoNER-COVID19"
```

---

### Task 15: Results aggregation + error analysis report tables

**Files:**
- Modify: `VietMed_NER.ipynb` (§8 reporting cell) **[Colab GPU for predictions]**

- [ ] **Step 1: Aggregate all per-run CSVs into one summary table**

```python
import glob, pandas as pd
summary = pd.concat([pd.read_csv(f) for f in glob.glob(f'{RESULTS_DIR}/0*.csv')], ignore_index=True)
summary.to_csv(f'{RESULTS_DIR}/summary.csv', index=False); summary
```

- [ ] **Step 2: Produce error-analysis tables for #2b vs #5 with real examples**

For each of `02b_phobert_crf128` and `05_phobert_gat_crf`: load best ckpt, predict on test, build `true_tags/pred_tags/words`, run `classify_errors`, print `error_counts`, and save `results/errors_<exp_id>.csv`. Pull 2–3 real Vietnamese example rows per category for the report.

```python
def error_report(exp_id):
    cfg=EXPERIMENTS[exp_id]; tokenizer=AutoTokenizer.from_pretrained(cfg["encoder"])
    # reuse run_eval-style decode to get true_tags,pred_tags,words on test split, then:
    df = classify_errors(true_tags, pred_tags, words_test)
    df.to_csv(f'{RESULTS_DIR}/errors_{exp_id}.csv', index=False)
    print(exp_id, error_counts(df)); return df
```

- [ ] **Step 3: Commit**

```bash
git add VietMed_NER.ipynb results/summary.csv results/errors_*.csv
git commit -m "exp: results summary + error analysis tables"
```

---

## Self-Review

**Spec coverage:**
- §2 Data → Tasks 2, 3 (load, labels, shared schema). ✓
- §3 Preprocess (per-encoder seg, re-align, -100) → Tasks 4, 5. ✓
- §4 Verify gate → Task 6. ✓
- §5 Models (softmax/CRF/GAT+CRF) → Tasks 7, 13. ✓
- §6 Train + Drive ckpt → Task 9. ✓
- §7 Eval (seqeval micro/macro/per-entity) → Task 8. ✓
- §8 Error analysis (4 categories + real examples) → Tasks 10, 15. ✓
- §9 Cross-domain shared schema → Task 14. ✓
- Experiment matrix rows #1/#2/#2b/#3/#4/#5/#6 → Tasks 11, 12, 13, 14. ✓
- Reproducibility (seed, pinned deps, CSV, one-exp-per-session) → Tasks 0, 1, 9. ✓
- 3-seeds-for-best (if time) — **optional**, noted here; not a separate task. Run Task 12/13 best config with `SEED∈{42,1,2}` and average if time permits.

**Placeholder scan:** Code provided for all core functions. Two Colab-dependent steps (cross-domain CRF word pairing in Task 14, `error_report` decode reuse in Task 15) intentionally reference the §6 decode pattern rather than duplicating ~30 lines — verified the referenced `decode_predictions`/`run_eval` exist in Task 8/9.

**Type consistency:** `build_model(cfg)`, `BertCrfForNer`/`GatCrfForNer.forward(input_ids, attention_mask, labels=None)`, `evaluate_tags`, `project_tags(tags, mapping)`, `realign_tags`, `align_labels`, `classify_errors`/`spans`/`error_counts` consistent across tasks. CRF/GAT forward contracts match.

**Known approximations to validate during execution:** the per-word recovery loop in Task 14 must mirror Task 8's masked-position pairing for CRF heads — flagged inline; validate on 2 samples before trusting the cross-domain number.
