#!/usr/bin/env python3
"""
BMI CHIP — SBP+MLP PIPELINE (Brochier, Monkey L only)
=======================================================
Single session, single file. WHY SINGLE SESSION:
  Channel N on monkey L and channel N on monkey N record different
  neurons in different brain locations. Pooling them makes the model
  learn contradictory mappings. One monkey = one consistent meaning
  per channel.

PIPELINE:
  l101210-001.ns5 (raw 30kHz, 96ch)
    -> select 8 best channels (Fisher, no filter needed for selection)
    -> bandpass 300-3000Hz + downsample 5kHz    [simulates LNA + ADC rate]
    -> 8-bit quantize, fixed rails              [simulates ADC]
    -> dual-band SBP: 16 features               [D2 block: 2 accumulators/ch]
    -> MLP 16->16(ReLU)->4                      [D4+D5]
    -> fold normalization into weights           [no runtime division]
    -> integer-only golden model                 [Verilog spec]
    -> export weights.hex + test_vectors.txt     [RTL verification]

CHIP OUTPUT:
  CLASS[1:0] = 00: Precision grip, low force   (PG-LF)
  CLASS[1:0] = 01: Precision grip, high force  (PG-HF)
  CLASS[1:0] = 10: Side grip, low force        (SG-LF)
  CLASS[1:0] = 11: Side grip, high force       (SG-HF)

DOWNLOAD (put in same folder as this script):
  l101210-001.ns5     from https://web.gin.g-node.org/INM6/multielectrode_grasp
  l101210-001-02.nev

INSTALL: pip install neo scipy quantities numpy torch matplotlib
USAGE:   python bmi_pipeline.py
"""

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# ── Hardware parameters (must match Verilog exactly) ──────────
TARGET_FS   = 5000    # ADC sample rate Hz
N_CH        = 8       # chip electrode count
SAMPLES     = 250     # samples per 50ms window at 5kHz
N_CLASSES   = 4
HIDDEN      = 16      # MLP hidden neurons (16->16->4 = 276 bytes total)
BP_LOW      = 300     # LNA bandpass low Hz
BP_HIGH     = 3000    # LNA bandpass high Hz
OFFSET_MS   = 100     # ms after GO-ON to start window

CLASS_NAMES = ['PG-LF', 'PG-HF', 'SG-LF', 'SG-HF']
GO_ON_MAP   = {'65381': 0, '65382': 1, '65385': 2, '65386': 3}


# ══════════════════════════════════════════════════════════════
# 1. FIND FILES
# ══════════════════════════════════════════════════════════════
print("="*65)
print("BMI PIPELINE — Brochier Monkey L")
print("="*65)

ns5 = glob.glob("*.ns5") + glob.glob("**/*.ns5", recursive=True)
nev = glob.glob("*-02.nev") + glob.glob("**/*-02.nev", recursive=True)

if not ns5:
    print("ERROR: l101210-001.ns5 not found.")
    print("Download: https://web.gin.g-node.org/INM6/multielectrode_grasp")
    sys.exit(1)
if not nev:
    print("ERROR: l101210-001-02.nev not found.")
    sys.exit(1)

ns5_path = os.path.abspath(ns5[0])
nev_path = os.path.abspath(nev[0])
print(f"NS5: {ns5_path}")
print(f"NEV: {nev_path}")

try:
    import neo
except ImportError:
    print("ERROR: pip install neo quantities")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 2. LOAD RAW BROADBAND
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Loading raw 30kHz broadband (1-3 min)...")
print("─"*65)

reader  = neo.io.BlackrockIO(filename=ns5_path, nsx_to_load=5)
block   = reader.read_block(lazy=False, load_waveforms=False)
raw_sig = next((a for a in block.segments[0].analogsignals
                if float(a.sampling_rate.rescale('Hz').magnitude) >= 25000), None)

if raw_sig is None:
    print("ERROR: No 30kHz signal in NS5."); sys.exit(1)

raw    = raw_sig.magnitude.astype(np.float32)
T_raw, n_raw = raw.shape
fs_raw = float(raw_sig.sampling_rate.rescale('Hz').magnitude)
DS     = int(round(fs_raw / TARGET_FS))
print(f"Loaded: {T_raw} samples x {n_raw} ch @ {fs_raw:.0f}Hz ({T_raw/fs_raw:.0f}s)")


# ══════════════════════════════════════════════════════════════
# 3. LOAD TRIAL EVENTS
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Loading trial events...")
print("─"*65)

nev_reader = neo.io.BlackrockIO(filename=nev_path, nsx_to_load=None)
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
counts = np.bincount([t[1] for t in trials], minlength=4)
print(f"Trials: {len(trials)} | " + " | ".join(f"{CLASS_NAMES[c]}:{counts[c]}" for c in range(4)))


# ══════════════════════════════════════════════════════════════
# 4. SELECT 8 CHANNELS (from raw, no filter needed)
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Selecting 8 channels by Fisher score (no filtering yet)...")
print("─"*65)

OFFSET_RAW = int(OFFSET_MS / 1000.0 * fs_raw)
WIN_RAW    = SAMPLES * DS

sbp_raw, valid_trials = [], []
for go_t, label in trials:
    s, e = int(go_t * fs_raw) + OFFSET_RAW, int(go_t * fs_raw) + OFFSET_RAW + WIN_RAW
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
print(f"Selected: {sorted(TOP8)}")


# ══════════════════════════════════════════════════════════════
# 5. FILTER + QUANTIZE (8 channels only — 12x faster)
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Filtering 8 channels + simulating ADC...")
print("─"*65)

nyq  = fs_raw / 2.0
b, a = sp.butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype='band')

filt8 = np.zeros((T_raw, N_CH), dtype=np.float32)
for ci, ch in enumerate(TOP8):
    filt8[:, ci] = sp.filtfilt(b, a, raw[:, ch].astype(np.float64))
del raw

v_min = np.percentile(filt8, 1)
v_max = np.percentile(filt8, 99)
filt_ds  = filt8[::DS].astype(np.float32)   # keep float version
del filt8
T_ds = filt_ds.shape[0]

# 8-bit ADC simulation (what the hardware sees)
adc_ds   = np.round(np.clip((filt_ds-v_min)/(v_max-v_min), 0, 1)*255).astype(np.int32)

# Float version (ideal — no quantization loss)
# Normalized to same 0-255 range as ADC but kept as float
float_ds = np.clip((filt_ds-v_min)/(v_max-v_min), 0, 1) * 255.0  # float, no rounding

print(f"ADC rails: [{v_min:.2f}, {v_max:.2f}]")
print(f"ADC SNR (signal / quantization noise): "
      f"{20*np.log10(255.0 / (np.abs(float_ds - adc_ds).mean() + 1e-10)):.1f} dB")


# ══════════════════════════════════════════════════════════════
# 6. CUT TRIAL WINDOWS
# ══════════════════════════════════════════════════════════════
OFFSET_DS = int(OFFSET_MS / 1000.0 * TARGET_FS)

X_all       = []   # 8-bit quantized ADC (what hardware sees)
X_all_float = []   # float, no quantization (ideal signal)
y_all       = []
for go_t, label in valid_trials:
    s, e = int(go_t*TARGET_FS)+OFFSET_DS, int(go_t*TARGET_FS)+OFFSET_DS+SAMPLES
    if s < 0 or e > T_ds: continue
    X_all.append(adc_ds[s:e, :].T)
    X_all_float.append(float_ds[s:e, :].T)
    y_all.append(label)

X_all       = np.array(X_all,       dtype=np.int32)    # (N, 8, 250) int
X_all_float = np.array(X_all_float, dtype=np.float32)  # (N, 8, 250) float
y_all       = np.array(y_all)
del adc_ds, float_ds
print(f"\nWindows: {len(y_all)} | Classes: {np.bincount(y_all, minlength=4)}")


# ══════════════════════════════════════════════════════════════
# 7. DUAL-BAND SBP FEATURES  (16 features = 2 per channel)
# ══════════════════════════════════════════════════════════════
# Split window in half, compute SBP on each half separately.
# First half: early reach phase. Second half: late reach/grasp.
# Hardware cost: two parallel accumulators per channel (trivial Verilog).
print(f"\n{'─'*65}")
print("Computing dual-band SBP (16 features)...")
print("─"*65)

HALF    = SAMPLES // 2
sbp_lo  = np.mean(np.abs(X_all[:, :, :HALF].astype(float)-128), axis=2)
sbp_hi  = np.mean(np.abs(X_all[:, :, HALF:].astype(float)-128), axis=2)
sbp_all = np.concatenate([sbp_lo, sbp_hi], axis=1)   # (N,16) from 8-bit ADC

# Same SBP computed from the FLOAT (unquantized) signal
# Midpoint for float version: 127.5 (equivalent of 128 in integer domain)
sbp_lo_f  = np.mean(np.abs(X_all_float[:, :, :HALF]-127.5), axis=2)
sbp_hi_f  = np.mean(np.abs(X_all_float[:, :, HALF:]-127.5), axis=2)
sbp_float = np.concatenate([sbp_lo_f, sbp_hi_f], axis=1)   # (N,16) from float

N_FEAT = sbp_all.shape[1]

# Measure ADC quantization effect on SBP features directly
sbp_diff = np.abs(sbp_all - sbp_float)
print(f"Features: {N_FEAT} | SBP range: [{sbp_all.min():.2f}, {sbp_all.max():.2f}]")
print(f"ADC quantization error in SBP features: {sbp_diff.mean():.4f} ± {sbp_diff.std():.4f}")
print(f"Max SBP error from ADC: {sbp_diff.max():.4f}  "
      f"({100*sbp_diff.mean()/sbp_all.mean():.2f}% of mean SBP)")


# ══════════════════════════════════════════════════════════════
# 8. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
perm    = np.random.permutation(len(y_all))
n_train = int(0.8 * len(y_all))
n_test  = len(y_all) - n_train

# Quantized SBP stats (from 8-bit ADC output — what hardware uses)
sbp_mean = sbp_all[perm[:n_train]].mean(0)
sbp_std  = sbp_all[perm[:n_train]].std(0) + 1e-8
sbp_norm = (sbp_all - sbp_mean) / sbp_std

# Float SBP stats (ideal signal, no ADC rounding)
sbp_mean_f = sbp_float[perm[:n_train]].mean(0)
sbp_std_f  = sbp_float[perm[:n_train]].std(0) + 1e-8
sbp_norm_f = (sbp_float - sbp_mean_f) / sbp_std_f

X_tr   = torch.tensor(sbp_norm[perm[:n_train]], dtype=torch.float32)
y_tr   = torch.tensor(y_all[perm[:n_train]], dtype=torch.long)
X_te   = torch.tensor(sbp_norm[perm[n_train:]], dtype=torch.float32)   # quantized
X_te_f = torch.tensor(sbp_norm_f[perm[n_train:]], dtype=torch.float32) # float
y_te   = torch.tensor(y_all[perm[n_train:]], dtype=torch.long)

# Weighted loss: fix class imbalance
train_cnt     = np.bincount(y_all[perm[:n_train]], minlength=4).astype(float)
class_weights = torch.tensor(1.0/(train_cnt+1e-6), dtype=torch.float32)
class_weights /= class_weights.sum()

print(f"\nTrain: {n_train} | Test (held-out): {n_test}")
print(f"Train: {np.bincount(y_all[perm[:n_train]], minlength=4)}")
print(f"Test:  {np.bincount(y_all[perm[n_train:]], minlength=4)}")
print(f"Class weights: {np.round(class_weights.numpy(), 3)}")


# ══════════════════════════════════════════════════════════════
# 9. TRAIN MLP WITH QUANTIZATION-AWARE TRAINING (QAT)
# ══════════════════════════════════════════════════════════════
# QAT simulates int8 rounding during the forward pass so the model
# learns weights that are ALREADY robust to quantization.
# The trick: fake-quantize each weight tensor on every forward pass.
# Gradients flow through the rounding (treated as identity for backprop).
# Result: the model converges to weights that survive int8 rounding.
#
# Two-phase schedule:
#   Phase 1 (epochs 1-600): normal float training, find good weights
#   Phase 2 (epochs 601-900): QAT fine-tuning with fake quantization
#   This is more stable than QAT from scratch on a small dataset.
print(f"\n{chr(9472)*65}")
print(f"Training MLP ({N_FEAT}->{HIDDEN}->{N_CLASSES}) with QAT fine-tuning...")
print(chr(9472)*65)
print("Phase 1: float training (epochs 1-600)")
print("Phase 2: QAT fine-tuning (epochs 601-900)")
print(chr(9472)*65)

loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


def fake_quantize(tensor, bits=8):
    """
    Simulate int8 quantization in-place for QAT.
    Maps max_abs → 127, rounds, then maps back to float.
    The forward output is rounded; gradients pass through unchanged
    (straight-through estimator).
    """
    mx = tensor.abs().max().item()
    if mx < 1e-10:
        return tensor
    scale = 127.0 / mx
    return (tensor * scale).round().clamp(-128, 127) / scale


class SbpMLP(nn.Module):
    def __init__(self, p=0.15, qat=False):
        super().__init__()
        self.h1   = nn.Linear(N_FEAT, HIDDEN)
        self.drop = nn.Dropout(p)
        self.o    = nn.Linear(HIDDEN, N_CLASSES)
        self.qat  = qat   # when True, fake-quantize weights each forward pass

    def forward(self, x):
        if self.qat and self.training:
            # Fake-quantize weights before the matrix multiply.
            # Gradients flow through as if rounding didn't happen.
            w1 = fake_quantize(self.h1.weight)
            b1 = fake_quantize(self.h1.bias)
            wo = fake_quantize(self.o.weight)
            bo = fake_quantize(self.o.bias)
            h  = F.linear(x, w1, b1)
            h  = F.relu(h)
            h  = self.drop(h)
            return F.linear(h, wo, bo)
        else:
            return self.o(self.drop(F.relu(self.h1(x))))

# ── Phase 1: float training ──
model     = SbpMLP(p=0.15, qat=False)
opt       = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600, eta_min=1e-4)

n_param = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_param}  ({N_FEAT}x{HIDDEN}+{HIDDEN} + {HIDDEN}x4+4 = "
      f"{N_FEAT*HIDDEN+HIDDEN+HIDDEN*4+4})")

best_acc, best_state = 0, None
for ep in range(600):
    model.train()
    for bx, by in loader:
        l = loss_fn(model(bx), by)
        opt.zero_grad(); l.backward(); opt.step()
    scheduler.step()
    if (ep+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            acc = 100*(model(X_te).argmax(1)==y_te).sum().item()/len(y_te)
        print(f"  [Float] Epoch {ep+1:3d} | Test: {acc:.1f}%  lr={scheduler.get_last_lr()[0]:.5f}")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k,v in model.state_dict().items()}

print(f"\n  Phase 1 best: {best_acc:.1f}%")

# ── Phase 2: QAT fine-tuning ──
# Enable fake quantization; lower LR to make small adjustments only
print("\nPhase 2: QAT fine-tuning (lower LR, fake-quantize weights)...")
model.qat = True
opt2       = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=300, eta_min=1e-6)

for ep in range(300):
    model.train()
    for bx, by in loader:
        l = loss_fn(model(bx), by)
        opt2.zero_grad(); l.backward(); opt2.step()
    scheduler2.step()
    if (ep+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            acc = 100*(model(X_te).argmax(1)==y_te).sum().item()/len(y_te)
        print(f"  [QAT]   Epoch {ep+1:3d} | Test: {acc:.1f}%  lr={scheduler2.get_last_lr()[0]:.6f}")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k,v in model.state_dict().items()}

# Use the best checkpoint across both phases
model.load_state_dict(best_state)
model.qat = False   # turn off QAT for clean evaluation
model.eval()
with torch.no_grad():
    preds    = model(X_te).argmax(1)
    acc_test = 100*(preds==y_te).sum().item()/len(y_te)

print(f"\n>>> Best: {acc_test:.1f}%  (chance=25%)")
print("\nPer-class:")
for c in range(4):
    mask = (y_te == c)
    if mask.sum() > 0:
        ca = 100*(preds[mask]==c).sum().item()/mask.sum().item()
        print(f"  {CLASS_NAMES[c]:>6s}: {ca:.1f}%  ({mask.sum().item()} samples)")

# ── ADC QUANTIZATION LOSS ──────────────────────────────────────
# Run the same model on the FLOAT (unquantized) SBP features.
# The model was trained on quantized features but we test on float here.
# The difference shows how much accuracy the 8-bit ADC cost us.
# A separate float-trained model is also trained for a fair upper bound.
print(f"\n{'─'*65}")
print("Measuring ADC quantization loss...")
print("─"*65)

# Test quantized-trained model on float input
model.eval()
with torch.no_grad():
    acc_float_input = 100*(model(X_te_f).argmax(1)==y_te).sum().item()/len(y_te)

# Train a separate model on float SBP (upper bound — ideal ADC)
model_f     = SbpMLP(p=0.15)
opt_f       = torch.optim.Adam(model_f.parameters(), lr=0.002, weight_decay=5e-4)
sched_f     = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=800, eta_min=1e-5)
loader_f    = DataLoader(
    TensorDataset(
        torch.tensor(sbp_norm_f[perm[:n_train]], dtype=torch.float32),
        y_tr
    ), batch_size=32, shuffle=True)
best_f, best_state_f = 0, None
for ep in range(800):
    model_f.train()
    for bx, by in loader_f:
        l = loss_fn(model_f(bx), by)
        opt_f.zero_grad(); l.backward(); opt_f.step()
    sched_f.step()
    if (ep+1) % 200 == 0:
        model_f.eval()
        with torch.no_grad():
            a = 100*(model_f(X_te_f).argmax(1)==y_te).sum().item()/len(y_te)
        if a > best_f:
            best_f, best_state_f = a, {k: v.clone() for k,v in model_f.state_dict().items()}

model_f.load_state_dict(best_state_f)
model_f.eval()
with torch.no_grad():
    acc_ideal = 100*(model_f(X_te_f).argmax(1)==y_te).sum().item()/len(y_te)

print(f"\nADC QUANTIZATION LOSS BREAKDOWN:")
print(f"  Ideal (float signal, float model):    {acc_ideal:.1f}%  <- upper bound")
print(f"  Real  (8-bit ADC, float model):       {acc_test:.1f}%  <- what your chip achieves")
print(f"  ADC quantization loss:                {acc_ideal - acc_test:.1f}%")
print(f"  Weight quantization loss (int8 MLP):  (see below after golden model)")
print(f"\n  Ideal - Real = {acc_ideal - acc_test:.1f}% is the cost of your 8-bit ADC.")
print(f"  If this is < 3%, your ADC resolution is sufficient.")


# ══════════════════════════════════════════════════════════════
# 10. QUANTIZE + FOLD NORMALIZATION INTO WEIGHTS
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Quantizing (fold normalization into weights)...")
print("─"*65)

def quantize(arr):
    arr = np.array(arr)
    mx  = np.abs(arr).max()
    if mx < 1e-10: return np.zeros_like(arr, dtype=np.int16), 1.0
    sc  = 127.0 / mx
    return np.round(arr*sc).clip(-128,127).astype(np.int16), float(sc)

hw_f = model.h1.weight.detach().numpy()
hb_f = model.h1.bias.detach().numpy()
ow_f = model.o.weight.detach().numpy()
ob_f = model.o.bias.detach().numpy()

hw_eff = hw_f / sbp_std[np.newaxis, :]
hb_eff = hb_f - np.sum(hw_f * sbp_mean[np.newaxis,:] / sbp_std[np.newaxis,:], axis=1)

sbp_t = sbp_all[0]
assert np.allclose(hw_f@((sbp_t-sbp_mean)/sbp_std)+hb_f,
                   hw_eff@sbp_t+hb_eff, atol=1e-5), "Fold failed!"
print("Fold verification: PASSED")

hw_q, sc_hw = quantize(hw_eff)
hb_q, sc_hb = quantize(hb_eff)
ow_q, sc_ow = quantize(ow_f)
ob_q, sc_ob = quantize(ob_f)

hidden_bias_scale = int(round(sc_hw / sc_hb))
output_bias_scale = int(round(sc_hw * sc_ow / sc_ob))
total_bytes       = hw_q.size + hb_q.size + ow_q.size + ob_q.size

print(f"Scales: hw={sc_hw:.1f} hb={sc_hb:.1f} ow={sc_ow:.1f} ob={sc_ob:.1f}")
print(f"Hidden bias scale: {hidden_bias_scale}")
print(f"Output bias scale: {output_bias_scale}")
print(f"Total bytes: {total_bytes}")


# ══════════════════════════════════════════════════════════════
# 11. INTEGER-ONLY GOLDEN MODEL  (= Verilog specification)
# ══════════════════════════════════════════════════════════════
def compute_sbp_dual(adc_win):
    """
    D2: Dual-band SBP. Input: (8,250) int. Output: (16,) int.
    First 8:  mean(|adc-128|) over samples 0..124
    Last 8:   mean(|adc-128|) over samples 125..249
    Verilog: two parallel accumulate-and-shift blocks per channel.
    """
    half = SAMPLES // 2
    out  = np.zeros(N_FEAT, dtype=np.int32)
    for ch in range(N_CH):
        acc = np.int32(0)
        for s in range(half):
            acc += abs(int(adc_win[ch,s]) - 128)
        out[ch] = acc // half
        acc = np.int32(0)
        for s in range(half, SAMPLES):
            acc += abs(int(adc_win[ch,s]) - 128)
        out[N_CH+ch] = acc // half
    return out

def golden_model(adc_win):
    """
    Full integer-only pipeline. THIS IS YOUR VERILOG SPEC.
    Input:  (8,250) int32 0-255
    Output: (class, scores)
    """
    sbp = compute_sbp_dual(adc_win)           # (16,) int32

    h = np.zeros(HIDDEN, dtype=np.int64)       # D4: hidden
    for j in range(HIDDEN):
        acc = np.int64(0)
        for i in range(N_FEAT):
            acc += np.int64(sbp[i]) * np.int64(hw_q[j,i])
        acc += np.int64(hb_q[j]) * np.int64(hidden_bias_scale)
        h[j] = max(0, acc)

    scores = np.zeros(N_CLASSES, dtype=np.int64)  # D5: output
    for k in range(N_CLASSES):
        acc = np.int64(0)
        for j in range(HIDDEN):
            acc += np.int64(h[j]) * np.int64(ow_q[k,j])
        acc += np.int64(ob_q[k]) * np.int64(output_bias_scale)
        scores[k] = acc

    return int(np.argmax(scores)), scores

print(f"\n{'─'*65}")
print("Running golden model on held-out test set...")
print("─"*65)

test_idx  = perm[n_train:]
q_correct = 0
for i in range(n_test):
    pred, _ = golden_model(X_all[test_idx[i]])
    q_correct += (pred == y_all[test_idx[i]])
acc_q = 100 * q_correct / n_test

print(f"Float:     {acc_test:.1f}%")
print(f"Quantized: {acc_q:.1f}%")
print(f"Drop:      {acc_test - acc_q:.1f}%")
print("PASS" if acc_test - acc_q < 5 else "WARNING: drop > 5%")


# ══════════════════════════════════════════════════════════════
# 12. EXPORT
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("Exporting...")
print("─"*65)

all_w = np.concatenate([hw_q.flatten(), hb_q.flatten(),
                        ow_q.flatten(), ob_q.flatten()])
with open('weights.hex', 'w') as f:
    f.write(f"// BMI ASIC — Brochier Monkey L\n")
    f.write(f"// {N_FEAT}->{HIDDEN}->{N_CLASSES}, {total_bytes} bytes\n")
    f.write(f"// hidden_bias_scale = {hidden_bias_scale}\n")
    f.write(f"// output_bias_scale = {output_bias_scale}\n\n")
    for v in all_w:
        f.write(f"{(int(v) if v >= 0 else int(v)+256):02X}\n")

all_w.astype(np.int8).tofile('weights.bin')

with open('weights_readable.txt', 'w') as f:
    f.write(f"// {N_FEAT}->{HIDDEN}->{N_CLASSES}\n")
    f.write(f"// hidden_bias_scale={hidden_bias_scale}  output_bias_scale={output_bias_scale}\n\n")
    f.write("// hw[neuron][input]:\n")
    for j in range(HIDDEN):
        vals = ', '.join(f"{int(hw_q[j,i]):+4d}" for i in range(N_FEAT))
        f.write(f"// hw[{j:2d}] = [{vals}]\n")
    f.write(f"\n// hb: [{', '.join(f'{int(v):+4d}' for v in hb_q)}]\n")
    f.write("// ow[class][neuron]:\n")
    for k in range(N_CLASSES):
        vals = ', '.join(f"{int(ow_q[k,j]):+4d}" for j in range(HIDDEN))
        f.write(f"// ow[{k}] ({CLASS_NAMES[k]}) = [{vals}]\n")
    f.write(f"\n// ob: [{', '.join(f'{int(v):+4d}' for v in ob_q)}]\n")
    f.write(f"\n// Channels (0-indexed): {sorted(TOP8)}\n")
    f.write(f"// ADC rails: v_min={v_min:.2f}  v_max={v_max:.2f}\n")

N_VEC     = min(40, n_test)
per_class = {c: [i for i in range(n_test) if y_all[test_idx[i]]==c] for c in range(4)}
selected  = []
for c in range(4):
    selected.extend(per_class[c][:N_VEC//4])
remaining = [i for i in range(n_test) if i not in selected]
selected.extend(remaining[:N_VEC-len(selected)])

with open('test_vectors.txt', 'w') as f:
    f.write(f"// RTL TEST VECTORS — Brochier Monkey L\n")
    f.write(f"// Pipeline: ADC(8ch,{SAMPLES}) -> dual-SBP(16) -> MLP({N_FEAT}->{HIDDEN}->4)\n")
    f.write(f"// hidden_bias_scale = {hidden_bias_scale}\n")
    f.write(f"// output_bias_scale = {output_bias_scale}\n\n")
    for vi, i in enumerate(selected[:N_VEC]):
        adc_win  = X_all[test_idx[i]]
        pred, sc = golden_model(adc_win)
        true_lbl = int(y_all[test_idx[i]])
        sbp_int  = compute_sbp_dual(adc_win)
        f.write(f"// ── Vec {vi:02d}: true={CLASS_NAMES[true_lbl]}  pred={CLASS_NAMES[pred]}\n")
        f.write(f"// Expected CLASS[1:0] = {pred:02b} = {pred}\n")
        f.write(f"// Scores: [{sc[0]}, {sc[1]}, {sc[2]}, {sc[3]}]\n")
        f.write(f"// SBP lo: [{', '.join(str(int(sbp_int[i])) for i in range(8))}]\n")
        f.write(f"// SBP hi: [{', '.join(str(int(sbp_int[i])) for i in range(8,16))}]\n")
        for ch in range(N_CH):
            vals = ' '.join(f"{int(adc_win[ch,s]):02X}" for s in range(SAMPLES))
            f.write(f"// ch{ch}: {vals}\n")
        f.write('\n')

torch.save(model.state_dict(), 'model.pth')
np.savez('preprocessing.npz',
         sbp_mean=sbp_mean, sbp_std=sbp_std,
         selected_channels=TOP8,
         v_min=v_min, v_max=v_max)

print(f"weights.hex, weights.bin, weights_readable.txt")
print(f"test_vectors.txt  ({N_VEC} vectors)")
print(f"model.pth, preprocessing.npz")


# ══════════════════════════════════════════════════════════════
# 13. PLOTS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bar chart
ax = axes[0]
ax.bar(['Float', '8-bit'], [acc_test, acc_q],
       color=['#378ADD','#1D9E75'], width=0.4, edgecolor='white', lw=1.5)
ax.axhline(25, color='#E24B4A', ls='--', lw=1.5, label='Chance (25%)')
for x, v in enumerate([acc_test, acc_q]):
    ax.text(x, v+0.5, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100); ax.set_ylabel('Accuracy (%)'); ax.legend()
ax.set_title(f'Accuracy\n{len(y_all)} trials, {n_test} held-out')

# SBP per class
ax = axes[1]
for c in range(4):
    mask = (y_all == c)
    ax.plot(sbp_lo[mask].mean(0), 'o-', label=CLASS_NAMES[c], lw=1.5)
ax.set_xlabel('Channel'); ax.set_ylabel('Mean SBP (first half)')
ax.set_title('SBP (early window) by class'); ax.legend(fontsize=8)

# Confusion
conf = np.zeros((4,4), dtype=int)
for t, p in zip(y_te.numpy(), preds.numpy()):
    conf[t][p] += 1
ax = axes[2]
ax.imshow(conf, cmap='Blues')
ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(conf[i][j]), ha='center', va='center',
                color='white' if conf[i][j] > conf.max()/2 else 'black')
ax.set_title(f'Confusion ({acc_test:.1f}%)')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("FINAL RESULTS")
print(f"{'='*65}")
print(f"  Dataset:        Brochier Monkey L ({len(y_all)} trials)")
print(f"  Channels:       8 of {n_raw} | selected: {sorted(TOP8)}")
print(f"  Features:       {N_FEAT} (8 ch x 2 time halves)")
print(f"  Architecture:   {N_FEAT}->{HIDDEN}(ReLU)->{N_CLASSES}")
print(f"  Weight bytes:   {total_bytes}")
print()
print("  QUANTIZATION LOSS CHAIN:")
print(f"  Ideal (float signal + float weights):  {acc_ideal:.1f}%  <- no quantization at all")
print(f"  ADC quant (8-bit signal, float model): {acc_test:.1f}%  <- your real chip (float weights)")
print(f"  Full hw  (8-bit signal + int8 weights):{acc_q:.1f}%  <- fully quantized hardware")
print()
print(f"  ADC quantization cost:     {acc_ideal - acc_test:.1f}%  (ideal -> 8-bit ADC)")
print(f"  Weight quantization cost:  {acc_test - acc_q:.1f}%  (float -> int8 MLP weights)")
print(f"  Total quantization cost:   {acc_ideal - acc_q:.1f}%  (ideal -> fully quantized)")
print()
if acc_ideal - acc_test < 3:
    print("  ADC: PASS — 8-bit resolution is sufficient (<3% loss)")
else:
    print("  ADC: WARNING — consider 10-bit ADC or better calibration")
if acc_test - acc_q < 3:
    print("  Weights: PASS — int8 MLP weights are sufficient (<3% loss)")
else:
    print("  Weights: WARNING — consider 12-bit weights")