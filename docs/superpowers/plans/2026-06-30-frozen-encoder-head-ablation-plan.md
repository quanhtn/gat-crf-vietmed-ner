# Frozen-Encoder Head Ablation — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-06-30-frozen-encoder-head-ablation-design.md`
**Deliverable:** new notebook `VietMed_NER_06_frozen_head_ablation.ipynb`
**Date:** 2026-06-30

Each step lists what to build and a **verify** checkpoint. Steps are ordered; do not start a
step until the previous verify passes. Code blocks for NEW components are concrete; REUSED
components are copied verbatim from `VietMed_NER_05_phobert_gat_crf.ipynb` and not retyped here.

---

## Step 0 — §0 Setup
**Build:** install + imports + seed + Drive.
```python
!pip -q install "transformers==4.44.2" "datasets==2.21.0" seqeval pytorch-crf py_vncorenlp pandas
!pip -q install torch_geometric            # NEW vs notebook 05's setup cell
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
from google.colab import drive; drive.mount('/content/drive')
DRIVE_DIR='/content/drive/MyDrive/vietmed_ner'
RESULTS_DIR=f'{DRIVE_DIR}/results'; CKPT_DIR=f'{DRIVE_DIR}/checkpoints'; CACHE_DIR=f'{DRIVE_DIR}/feat_cache'
for d in (RESULTS_DIR, CKPT_DIR, CACHE_DIR): os.makedirs(d, exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; print('device:', DEVICE)
```
**Verify:** prints `device: cuda`; `import torch_geometric` succeeds.

## Step 1 — §1 Config
**Build:**
```python
CFG = {"encoder":"vinai/phobert-base", "max_len":128, "segment":True, "batch":32}
SEEDS = [42, 1, 2]
```
plus the **reused** label list / `label2id` / `id2label` / `LABEL_LIST` cell from notebook 05.
**Verify:** `len(LABEL_LIST)` equals the VietMed-NER tag count (37 BIO tags for 18 types + O).

## Step 2 — §2–§3 Data pipeline (REUSED verbatim)
**Build:** copy unchanged from notebook 05:
`get_words_labels`, `get_segmenter`, `segment_to_groups`, `realign_tags`, `align_labels`,
`build_encoded`. No edits.
**Verify:** `build_encoded("validation", CFG, tokenizer)` returns a dict with `input_ids`,
`attention_mask`, `labels` of equal outer length.

## Step 3 — §T tests + §4 verify gate (REUSED verbatim)
**Build:** copy `_test_project/_test_realign/_test_align/_test_errors` and `verify_roundtrip`.
**Verify (GATE):** `ALL §T TESTS PASSED` prints AND
`verify_roundtrip PASSED for 5/5 samples` prints. Do not proceed otherwise.

## Step 4 — §5 Feature cache (NEW)
**Build:** freeze encoder, forward once per split, store fp16 features.
```python
from transformers import AutoModel, RobertaTokenizerFast
def load_frozen_encoder():
    enc = AutoModel.from_pretrained(CFG["encoder"]).to(DEVICE).eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc

def cache_features(split, tokenizer, encoder, batch=32):
    path = f'{CACHE_DIR}/cache_{split}.pt'
    if os.path.exists(path): return torch.load(path)
    tok  = build_encoded(split, CFG, tokenizer)
    ids  = torch.tensor(tok["input_ids"]); mask = torch.tensor(tok["attention_mask"])
    labs = torch.tensor(tok["labels"])
    feats = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            h = encoder(ids[i:i+batch].to(DEVICE),
                        attention_mask=mask[i:i+batch].to(DEVICE)).last_hidden_state
            feats.append(h.half().cpu())
    feats = torch.cat(feats, 0)
    assert feats.shape[:2] == labs.shape, (feats.shape, labs.shape)
    obj = {"features": feats, "mask": mask, "labels": labs}
    torch.save(obj, path); print(f'{split}: {tuple(feats.shape)} cached -> {path}')
    return obj

tokenizer = RobertaTokenizerFast.from_pretrained(CFG["encoder"], add_prefix_space=True)
encoder   = load_frozen_encoder()
CACHE = {s: cache_features(s, tokenizer, encoder, CFG["batch"]) for s in ["train","validation","test"]}
```
**Verify:** three `.pt` files exist; printed shapes are `[N,128,768]`; assert passes; free VRAM
after (encoder can stay loaded but is unused downstream).

## Step 5 — §6 Heads (NEW)
**Build:** three heads on `[B,L,H]` features. CRF in fp32; GAT masks PAD and batches with PyG.
```python
from torchcrf import CRF
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
C = len(LABEL_LIST)

class SoftmaxHead(nn.Module):
    def __init__(self, H, C): super().__init__(); self.fc=nn.Linear(H,C)
    def forward(self, feats, mask, labels=None):
        logits = self.fc(feats)
        if labels is not None:
            return F.cross_entropy(logits.reshape(-1,C), labels.reshape(-1), ignore_index=-100)
        return logits.argmax(-1)                 # [B,L]; decode with head='softmax'
    head_kind = "softmax"

class CrfHead(nn.Module):
    def __init__(self, H, C): super().__init__(); self.fc=nn.Linear(H,C); self.crf=CRF(C,batch_first=True)
    def forward(self, feats, mask, labels=None):
        em = self.fc(feats); m = mask.bool()
        if labels is not None:
            safe = labels.clone(); safe[safe==-100]=0
            return -self.crf(em, safe, mask=m, reduction="mean")
        return self.crf.decode(em, mask=m)        # list[list]; decode with head='crf'
    head_kind = "crf"

class GatCrfHead(nn.Module):
    def __init__(self, H, C, heads=4):
        super().__init__()
        self.gat=GATConv(H, H//heads, heads=heads, dropout=0.1)
        self.fc=nn.Linear(H,C); self.crf=CRF(C,batch_first=True)
    def _emissions(self, feats, mask):
        B,L,H = feats.shape
        datas=[]
        for b in range(B):
            n=int(mask[b].sum()); x=feats[b,:n]
            idx=torch.arange(n, device=feats.device)
            ei=torch.stack([idx.repeat_interleave(n), idx.repeat(n)],0)   # FC over REAL tokens only
            datas.append(Data(x=x, edge_index=ei))
        g=Batch.from_data_list(datas)
        out=self.gat(g.x, g.edge_index)           # [sum_n, H]
        h=feats.clone(); ptr=0
        for b in range(B):
            n=int(mask[b].sum()); h[b,:n]=out[ptr:ptr+n]; ptr+=n
        return self.fc(h)
    def forward(self, feats, mask, labels=None):
        em=self._emissions(feats, mask); m=mask.bool()
        if labels is not None:
            safe=labels.clone(); safe[safe==-100]=0
            return -self.crf(em, safe, mask=m, reduction="mean")
        return self.crf.decode(em, mask=m)
    head_kind = "crf"
```
**Verify:** smoke test on one cached batch — each head's training call returns a finite scalar
loss (`torch.isfinite(loss)`), and inference returns predictions of the right shape/length.
Confirms the PAD-mask GAT does not NaN.

## Step 6 — §7 Trainer + eval (NEW trainer, REUSED decode/eval)
**Build:** reuse `evaluate_tags` and `decode_predictions` from notebook 05 (head flag = `head_kind`).
```python
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

def _loader(cache, bs, shuffle):
    ds=TensorDataset(cache["features"], cache["mask"], cache["labels"])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def _eval(head, cache, bs=64):
    head.eval(); T,P=[],[]
    dl=_loader(cache, bs, False)
    with torch.no_grad():
        for feats,mask,labs in dl:
            feats=feats.float().to(DEVICE); mask=mask.to(DEVICE)
            pred=head(feats, mask)
            pred=pred.cpu().tolist() if head.head_kind=="softmax" else pred
            t,p=decode_predictions(pred, labs.tolist(), head.head_kind); T+=t; P+=p
    return evaluate_tags(T,P)

def train_head(make_head, seed, lr=1e-3, epochs=30, patience=3, bs=64):
    set_seed(seed); head=make_head().to(DEVICE)
    opt=torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    dl=_loader(CACHE["train"], bs, True)
    total=len(dl)*epochs; sched=get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    best,bad,best_state=-1,0,None
    for ep in range(epochs):
        head.train()
        for feats,mask,labs in dl:
            feats=feats.float().to(DEVICE); mask=mask.to(DEVICE); labs=labs.to(DEVICE)
            loss=head(feats, mask, labs); opt.zero_grad(); loss.backward()
            opt.step(); sched.step()
        f1=_eval(head, CACHE["validation"])["micro_f1"]
        if f1>best: best,bad,best_state=f1,0,{k:v.cpu() for k,v in head.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    head.load_state_dict(best_state)
    return _eval(head, CACHE["test"])
```
**Verify:** one `train_head(lambda: CrfHead(768,C), seed=42, epochs=2)` run completes and returns
a test report with **micro_f1 clearly > 0** (regression check: PAD-mask bug is gone).

## Step 7 — §8 Run matrix (NEW)
**Build:** all heads × all seeds.
```python
import pandas as pd
H=768
HEADS={"Softmax":lambda:SoftmaxHead(H,C),"CRF":lambda:CrfHead(H,C),"GAT+CRF":lambda:GatCrfHead(H,C)}
rows=[]
for name,mk in HEADS.items():
    for s in SEEDS:
        m=train_head(mk, seed=s)
        rows.append({"head":name,"seed":s,"micro_f1":m["micro_f1"],"macro_f1":m["macro_f1"]})
        print(f'{name} seed{s}: micro={m["micro_f1"]:.4f} macro={m["macro_f1"]:.4f}')
df=pd.DataFrame(rows)
```
**Verify:** 9 rows (3 heads × 3 seeds); every `micro_f1 > 0`.

## Step 8 — §9 Comparison report (NEW)
**Build:** aggregate mean±std + final per-head report on best seed; write CSV.
```python
agg=df.groupby("head").agg(micro_mean=("micro_f1","mean"),micro_std=("micro_f1","std"),
                           macro_mean=("macro_f1","mean"),macro_std=("macro_f1","std")).reset_index()
print(agg.to_string(index=False))
agg.to_csv(f'{RESULTS_DIR}/frozen_head_ablation.csv', index=False)
# per-entity report for each head at its best seed
for name,mk in HEADS.items():
    best_seed=df[df.head==name].sort_values("micro_f1").iloc[-1]["seed"]
    print(f'\n===== {name} (seed {int(best_seed)}) =====')
    print(train_head(mk, seed=int(best_seed))["report"])
```
**Verify:** table prints 3 rows with mean±std; CSV written; ordering observed and recorded
(expected Softmax ≤ CRF ≤ GAT+CRF, but report whatever is found — a null result is valid).

---

## Build order summary
1. Setup → 2. Config/labels → 3. Data pipeline (reuse) → 4. Tests+verify gate (GATE) →
5. Feature cache → 6. Heads → 7. Trainer → 8. Run matrix → 9. Report.

## Definition of done
- Notebook runs top-to-bottom on Colab T4 free without OOM/disconnect after the one-time cache.
- §T + verify gate pass.
- All 9 runs produce non-degenerate F1 (> 0).
- `frozen_head_ablation.csv` written with a 3-row mean±std comparison.
- Report text states the modest **head-ablation** framing (no dependency-graph claim).

## Notes / deviations from notebook 05
- AMP/`GradScaler` dropped (heads tiny, CRF wants fp32).
- `lr=1e-3` (head-only) vs `2e-5` (full fine-tune in nb05).
- GAT graph: PAD-masked FC over real tokens, PyG-batched (vs nb05's full-L Python loop).
- Encoder frozen via `requires_grad=False`, forwarded once and cached.
