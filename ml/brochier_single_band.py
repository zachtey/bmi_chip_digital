#!/usr/bin/env python3
"""
SINGLE-BAND SBP BASELINE — for comparison against dual-band version
=====================================================================
Identical pipeline to bmi_pipeline.py except:
  - SBP computed over full 250-sample window (not split in half)
  - 8 features instead of 16
  - MLP: 8->8->4 (108 bytes) instead of 16->16->4 (340 bytes)

Run both and compare accuracy to see whether splitting the window helps.

USAGE: python bmi_single_band.py
"""

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_FS   = 5000
N_CH        = 8
SAMPLES     = 250
N_CLASSES   = 4
HIDDEN      = 8        # 8->8->4 to match original hardware spec
BP_LOW      = 300
BP_HIGH     = 3000
OFFSET_MS   = 100
CLASS_NAMES = ['PG-LF', 'PG-HF', 'SG-LF', 'SG-HF']
GO_ON_MAP   = {'65381': 0, '65382': 1, '65385': 2, '65386': 3}

print("="*65)
print("SINGLE-BAND SBP BASELINE (8 features, 8->8->4 MLP)")
print("="*65)

ns5 = glob.glob("*.ns5") + glob.glob("**/*.ns5", recursive=True)
nev = glob.glob("*-02.nev") + glob.glob("**/*-02.nev", recursive=True)
if not ns5 or not nev:
    print("ERROR: data files not found."); sys.exit(1)

try:
    import neo
except ImportError:
    print("pip install neo quantities"); sys.exit(1)

# Load
print("\nLoading...")
reader  = neo.io.BlackrockIO(filename=os.path.abspath(ns5[0]), nsx_to_load=5)
block   = reader.read_block(lazy=False, load_waveforms=False)
raw_sig = next((a for a in block.segments[0].analogsignals
                if float(a.sampling_rate.rescale('Hz').magnitude) >= 25000), None)
raw    = raw_sig.magnitude.astype(np.float32)
T_raw, n_raw = raw.shape
fs_raw = float(raw_sig.sampling_rate.rescale('Hz').magnitude)
DS     = int(round(fs_raw / TARGET_FS))

nev_reader = neo.io.BlackrockIO(filename=os.path.abspath(nev[0]), nsx_to_load=None)
nev_seg    = nev_reader.read_block(lazy=False, load_waveforms=False).segments[0]
events = {}
for evt in nev_seg.events:
    times  = evt.times.rescale('s').magnitude
    labels = np.array(evt.labels, dtype=str)
    for lbl, t in zip(labels, times):
        events.setdefault(str(lbl).strip(), []).append(float(t))

trials = sorted(
    [(t, label) for code, label in GO_ON_MAP.items()
     for t in events.get(code, [])],
    key=lambda x: x[0]
)
print(f"Trials: {len(trials)}")

# Channel selection
OFFSET_RAW = int(OFFSET_MS / 1000.0 * fs_raw)
WIN_RAW    = SAMPLES * DS
sbp_raw, valid_trials = [], []
for go_t, label in trials:
    s, e = int(go_t*fs_raw)+OFFSET_RAW, int(go_t*fs_raw)+OFFSET_RAW+WIN_RAW
    if s < 0 or e > T_raw: continue
    win = raw[s:e:DS, :]
    sbp_raw.append(np.abs(win - win.mean(0)).mean(0))
    valid_trials.append((go_t, label))

sbp_raw = np.array(sbp_raw)
y_raw   = np.array([t[1] for t in valid_trials])
cls_m   = np.array([sbp_raw[y_raw==c].mean(0) for c in range(4) if (y_raw==c).sum()>0])
grand   = sbp_raw.mean(0)
between = np.mean((cls_m - grand)**2, axis=0)
within  = np.mean([sbp_raw[y_raw==c].var(0) for c in range(4) if (y_raw==c).sum()>0], axis=0)
TOP8    = np.argsort(between / (within + 1e-10))[-N_CH:]
print(f"Selected channels: {sorted(TOP8)}")

# Filter + ADC
nyq  = fs_raw / 2.0
b, a = sp.butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype='band')
filt8 = np.zeros((T_raw, N_CH), dtype=np.float32)
for ci, ch in enumerate(TOP8):
    filt8[:, ci] = sp.filtfilt(b, a, raw[:, ch].astype(np.float64))
del raw
v_min = np.percentile(filt8, 1)
v_max = np.percentile(filt8, 99)
filt_ds  = filt8[::DS].astype(np.float32)
float_ds = np.clip((filt_ds - v_min)/(v_max - v_min), 0, 1) * 255.0
adc_ds   = np.round(float_ds).astype(np.int32)
del filt8, filt_ds
T_ds = adc_ds.shape[0]

OFFSET_DS = int(OFFSET_MS / 1000.0 * TARGET_FS)
X_all, y_all = [], []
for go_t, label in valid_trials:
    s, e = int(go_t*TARGET_FS)+OFFSET_DS, int(go_t*TARGET_FS)+OFFSET_DS+SAMPLES
    if s < 0 or e > T_ds: continue
    X_all.append(adc_ds[s:e, :].T)
    y_all.append(label)

X_all = np.array(X_all, dtype=np.int32)
y_all = np.array(y_all)
print(f"Windows: {len(y_all)} | Classes: {np.bincount(y_all, minlength=4)}")

# ── SINGLE-BAND SBP (full window, 8 features) ──
sbp_full = np.mean(np.abs(X_all.astype(float)-128), axis=2)   # (N, 8)
N_FEAT   = sbp_full.shape[1]
print(f"Features: {N_FEAT} (single-band, full window)")
print(f"SBP range: [{sbp_full.min():.2f}, {sbp_full.max():.2f}]")

# Train/test
perm    = np.random.permutation(len(y_all))
n_train = int(0.8 * len(y_all))
n_test  = len(y_all) - n_train
sbp_mean = sbp_full[perm[:n_train]].mean(0)
sbp_std  = sbp_full[perm[:n_train]].std(0) + 1e-8
sbp_norm = (sbp_full - sbp_mean) / sbp_std

X_tr = torch.tensor(sbp_norm[perm[:n_train]], dtype=torch.float32)
y_tr = torch.tensor(y_all[perm[:n_train]], dtype=torch.long)
X_te = torch.tensor(sbp_norm[perm[n_train:]], dtype=torch.float32)
y_te = torch.tensor(y_all[perm[n_train:]], dtype=torch.long)

train_cnt     = np.bincount(y_all[perm[:n_train]], minlength=4).astype(float)
class_weights = torch.tensor(1.0/(train_cnt+1e-6), dtype=torch.float32)
class_weights /= class_weights.sum()

loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

class SbpMLP(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.h1   = nn.Linear(N_FEAT, HIDDEN)
        self.drop = nn.Dropout(p)
        self.o    = nn.Linear(HIDDEN, N_CLASSES)
    def forward(self, x):
        return self.o(self.drop(F.relu(self.h1(x))))

model     = SbpMLP()
opt       = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600, eta_min=1e-5)

print(f"\nTraining {N_FEAT}->{HIDDEN}->{N_CLASSES} (params={sum(p.numel() for p in model.parameters())})...")
best_acc, best_state = 0, None
for ep in range(600):
    model.train()
    for bx, by in loader:
        l = loss_fn(model(bx), by)
        opt.zero_grad(); l.backward(); opt.step()
    scheduler.step()
    if (ep+1) % 150 == 0:
        model.eval()
        with torch.no_grad():
            acc = 100*(model(X_te).argmax(1)==y_te).sum().item()/len(y_te)
        print(f"  Epoch {ep+1:3d} | {acc:.1f}%")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds    = model(X_te).argmax(1)
    acc_test = 100*(preds==y_te).sum().item()/len(y_te)

print(f"\n>>> Single-band SBP+MLP: {acc_test:.1f}%  (chance=25%)")
print("Per-class:")
for c in range(4):
    mask = (y_te == c)
    if mask.sum() > 0:
        ca = 100*(preds[mask]==c).sum().item()/mask.sum().item()
        print(f"  {CLASS_NAMES[c]:>6s}: {ca:.1f}%  ({mask.sum().item()} samples)")

print(f"\n{'='*65}")
print("COMPARISON TABLE (run bmi_pipeline.py for dual-band number)")
print(f"{'='*65}")
print(f"  Single-band SBP (8 features,  8->8->4,  108 bytes): {acc_test:.1f}%")
print(f"  Dual-band   SBP (16 features, 16->16->4, 340 bytes): see bmi_pipeline.py")
print(f"  The difference shows whether splitting the window adds value.")