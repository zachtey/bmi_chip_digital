#!/usr/bin/env python3
"""
BMI PIPELINE — Final clean version

Raw 30kHz → filter → downsample → 8-bit ADC → SBP(8) → MLP(8->8->4) → CLASS[1:0]

DOWNLOAD:
  l101210-001.ns5     https://web.gin.g-node.org/INM6/multielectrode_grasp
  l101210-001-02.nev

INSTALL: pip install neo scipy quantities numpy torch matplotlib
USAGE:   python3 bmi_pipeline_final.py
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

TARGET_FS   = 5000
N_CH        = 8
SAMPLES     = 250     # 50ms at 5kHz
N_CLASSES   = 4
HIDDEN      = 8       # 8->8->4 = 108 bytes total
BP_LOW, BP_HIGH = 300, 3000
OFFSET_MS   = 100
CLASS_NAMES = ['PG-LF', 'PG-HF', 'SG-LF', 'SG-HF']
GO_ON_MAP   = {'65381': 0, '65382': 1, '65385': 2, '65386': 3}

# ══════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════
print("="*60)
print("BMI PIPELINE — Brochier Monkey L")
print("="*60)

ns5 = glob.glob("*.ns5") + glob.glob("**/*.ns5", recursive=True)
nev = glob.glob("*-02.nev") + glob.glob("**/*-02.nev", recursive=True)
if not ns5 or not nev:
    print("ERROR: put l101210-001.ns5 and l101210-001-02.nev here")
    sys.exit(1)

try:
    import neo
except ImportError:
    print("pip install neo quantities"); sys.exit(1)

print("Loading raw broadband (1-3 min)...")
reader  = neo.io.BlackrockIO(filename=os.path.abspath(ns5[0]), nsx_to_load=5)
block   = reader.read_block(lazy=False, load_waveforms=False)
raw_sig = next((a for a in block.segments[0].analogsignals
                if float(a.sampling_rate.rescale('Hz').magnitude) >= 25000), None)
if raw_sig is None:
    print("ERROR: no 30kHz signal"); sys.exit(1)

raw    = raw_sig.magnitude.astype(np.float32)
T_raw, n_raw = raw.shape
fs_raw = float(raw_sig.sampling_rate.rescale('Hz').magnitude)
DS     = int(round(fs_raw / TARGET_FS))
print(f"  {T_raw} samples x {n_raw} ch @ {fs_raw:.0f}Hz")

nev_reader = neo.io.BlackrockIO(filename=os.path.abspath(nev[0]), nsx_to_load=None)
nev_seg    = nev_reader.read_block(lazy=False, load_waveforms=False).segments[0]
events = {}
for evt in nev_seg.events:
    for lbl, t in zip(np.array(evt.labels, dtype=str),
                      evt.times.rescale('s').magnitude):
        events.setdefault(str(lbl).strip(), []).append(float(t))

trials = sorted([(t, lbl) for code, lbl in GO_ON_MAP.items()
                 for t in events.get(code, [])], key=lambda x: x[0])
counts = np.bincount([t[1] for t in trials], minlength=4)
print(f"  Trials: {len(trials)} — " +
      " | ".join(f"{CLASS_NAMES[c]}:{counts[c]}" for c in range(4)))

# ══════════════════════════════════════════════════════════════
# 2. SELECT 8 CHANNELS (Fisher, no filter needed for selection)
# ══════════════════════════════════════════════════════════════
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
cls_m   = np.array([sbp_raw[y_raw==c].mean(0) for c in range(4)])
grand   = sbp_raw.mean(0)
between = np.mean((cls_m - grand)**2, axis=0)
within  = np.mean([sbp_raw[y_raw==c].var(0) for c in range(4)], axis=0)
TOP8    = np.argsort(between / (within + 1e-10))[-N_CH:]
print(f"  Selected channels: {sorted(int(x) for x in TOP8)}")

# ══════════════════════════════════════════════════════════════
# 3. FILTER 8 CHANNELS + SIMULATE ADC
# ══════════════════════════════════════════════════════════════
print("Filtering 8 channels + ADC simulation...")
nyq  = fs_raw / 2.0
b, a = sp.butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype='band')

filt8 = np.zeros((T_raw, N_CH), dtype=np.float32)
for ci, ch in enumerate(TOP8):
    filt8[:, ci] = sp.filtfilt(b, a, raw[:, ch].astype(np.float64))
del raw

v_min = np.percentile(filt8, 1)
v_max = np.percentile(filt8, 99)
filt_ds = filt8[::DS]
del filt8
T_ds   = filt_ds.shape[0]
adc_ds = np.round(np.clip((filt_ds-v_min)/(v_max-v_min), 0, 1)*255).astype(np.int32)
del filt_ds
print(f"  ADC rails: [{v_min:.2f}, {v_max:.2f}]")

# ══════════════════════════════════════════════════════════════
# 4. CUT TRIAL WINDOWS
# ══════════════════════════════════════════════════════════════
OFFSET_DS = int(OFFSET_MS / 1000.0 * TARGET_FS)
X_all, y_all = [], []
for go_t, label in valid_trials:
    s, e = int(go_t*TARGET_FS)+OFFSET_DS, int(go_t*TARGET_FS)+OFFSET_DS+SAMPLES
    if s < 0 or e > T_ds: continue
    X_all.append(adc_ds[s:e, :].T)
    y_all.append(label)

X_all = np.array(X_all, dtype=np.int32)  # (N, 8, 250)
y_all = np.array(y_all)
del adc_ds
print(f"  Windows: {len(y_all)} | {np.bincount(y_all, minlength=4)}")

# ══════════════════════════════════════════════════════════════
# 5. COMPUTE SBP FEATURES  (D2 block)
# ══════════════════════════════════════════════════════════════
# SBP = mean(|ADC - 128|) per channel
# 128 = ADC midpoint (0V). Absolute deviation = spike energy.
sbp = np.mean(np.abs(X_all.astype(float) - 128), axis=2)  # (N, 8)
print(f"  SBP range: [{sbp.min():.1f}, {sbp.max():.1f}]")

# ══════════════════════════════════════════════════════════════
# 6. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
perm    = np.random.permutation(len(y_all))
n_train = int(0.8 * len(y_all))
n_test  = len(y_all) - n_train

sbp_mean = sbp[perm[:n_train]].mean(0)
sbp_std  = sbp[perm[:n_train]].std(0) + 1e-8
sbp_norm = (sbp - sbp_mean) / sbp_std

X_tr = torch.tensor(sbp_norm[perm[:n_train]], dtype=torch.float32)
y_tr = torch.tensor(y_all[perm[:n_train]], dtype=torch.long)
X_te = torch.tensor(sbp_norm[perm[n_train:]], dtype=torch.float32)
y_te = torch.tensor(y_all[perm[n_train:]], dtype=torch.long)

# Weighted loss: handle class imbalance
cnt   = np.bincount(y_all[perm[:n_train]], minlength=4).astype(float)
wt    = torch.tensor(1.0/(cnt+1e-6), dtype=torch.float32)
wt   /= wt.sum()

print(f"\nTrain: {n_train} | Test: {n_test}")
print(f"Train: {np.bincount(y_all[perm[:n_train]], minlength=4)}")
print(f"Test:  {np.bincount(y_all[perm[n_train:]], minlength=4)}")

# ══════════════════════════════════════════════════════════════
# 7. TRAIN MLP  (8->8->4, 108 bytes)
# ══════════════════════════════════════════════════════════════
print(f"\nTraining 8->8->4 MLP...")
loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
loss_fn = nn.CrossEntropyLoss(weight=wt)

class SbpMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Linear(N_CH, HIDDEN)
        self.o = nn.Linear(HIDDEN, N_CLASSES)
    def forward(self, x):
        return self.o(F.relu(self.h(x)))

model     = SbpMLP()
opt       = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}  (should be 108)")

best_acc, best_state = 0, None
for ep in range(500):
    model.train()
    for bx, by in loader:
        l = loss_fn(model(bx), by)
        opt.zero_grad(); l.backward(); opt.step()
    scheduler.step()
    if (ep+1) % 100 == 0:
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

print(f"\n>>> Accuracy: {acc_test:.1f}%  (chance=25%)")
for c in range(4):
    mask = (y_te == c)
    if mask.sum() > 0:
        ca = 100*(preds[mask]==c).sum().item()/mask.sum().item()
        print(f"  {CLASS_NAMES[c]:>6s}: {ca:.1f}%  ({mask.sum().item()} samples)")

# ══════════════════════════════════════════════════════════════
# 8. QUANTIZE — fold normalization into weights
# ══════════════════════════════════════════════════════════════
print(f"\nQuantizing to 8-bit...")

def quantize(arr):
    arr = np.array(arr)
    mx  = np.abs(arr).max()
    if mx < 1e-10: return np.zeros_like(arr, dtype=np.int16), 1.0
    sc  = 127.0 / mx
    return np.round(arr*sc).clip(-128,127).astype(np.int16), float(sc)

hw_f = model.h.weight.detach().numpy()   # (8, 8)
hb_f = model.h.bias.detach().numpy()     # (8,)
ow_f = model.o.weight.detach().numpy()   # (4, 8)
ob_f = model.o.bias.detach().numpy()     # (4,)

# Fold normalization: W_eff = W/std,  b_eff = b - W*(mean/std)
hw_eff = hw_f / sbp_std[np.newaxis, :]
hb_eff = hb_f - np.sum(hw_f * sbp_mean[np.newaxis,:] / sbp_std[np.newaxis,:], axis=1)

# Verify
assert np.allclose(hw_f@((sbp[0]-sbp_mean)/sbp_std)+hb_f,
                   hw_eff@sbp[0]+hb_eff, atol=1e-5), "Fold failed!"
print("  Fold verification: PASSED")

hw_q, sc_hw = quantize(hw_eff)
hb_q, sc_hb = quantize(hb_eff)
ow_q, sc_ow = quantize(ow_f)
ob_q, sc_ob = quantize(ob_f)

# Bias scales: input to golden model is raw sbp_int (range 0-128)
# weighted_sum = hw_q * sbp_int ≈ sc_hw * hw_float * sbp_int
# bias must match: hb_q * scale = hb_float * sc_hw → scale = sc_hw / sc_hb
hidden_bias_scale = int(round(sc_hw / sc_hb))
output_bias_scale = int(round(sc_hw * sc_ow / sc_ob))
total_bytes       = hw_q.size + hb_q.size + ow_q.size + ob_q.size

print(f"  hidden_bias_scale = {hidden_bias_scale}")
print(f"  output_bias_scale = {output_bias_scale}")
print(f"  Total bytes: {total_bytes}  (should be 108)")

# ══════════════════════════════════════════════════════════════
# 9. INTEGER-ONLY GOLDEN MODEL  (= Verilog spec)
# ══════════════════════════════════════════════════════════════
def compute_sbp(adc_win):
    """D2: SBP = sum(|adc-128|)/SAMPLES per channel. (8,250)int -> (8,)int."""
    out = np.zeros(N_CH, dtype=np.int32)
    for ch in range(N_CH):
        acc = np.int32(0)
        for s in range(SAMPLES):
            acc += abs(int(adc_win[ch,s]) - 128)
        out[ch] = acc // SAMPLES
    return out

def golden_model(adc_win):
    """
    Integer-only forward pass — YOUR VERILOG SPECIFICATION.
    Input:  (8, 250) int32, values 0-255
    Output: (class 0-3, scores int64[4])
    """
    sbp_int = compute_sbp(adc_win)                 # (8,) 0-128

    h = np.zeros(HIDDEN, dtype=np.int64)            # D4: hidden
    for j in range(HIDDEN):
        acc = np.int64(0)
        for i in range(N_CH):
            acc += np.int64(sbp_int[i]) * np.int64(hw_q[j,i])
        acc += np.int64(hb_q[j]) * np.int64(hidden_bias_scale)
        h[j] = max(0, acc)                          # ReLU

    scores = np.zeros(N_CLASSES, dtype=np.int64)    # D5: output
    for k in range(N_CLASSES):
        acc = np.int64(0)
        for j in range(HIDDEN):
            acc += np.int64(h[j]) * np.int64(ow_q[k,j])
        acc += np.int64(ob_q[k]) * np.int64(output_bias_scale)
        scores[k] = acc

    return int(np.argmax(scores)), scores            # D6: argmax

print("\nRunning golden model on test set...")
test_idx  = perm[n_train:]
q_correct = 0
for i in range(n_test):
    pred, _ = golden_model(X_all[test_idx[i]])
    q_correct += (pred == y_all[test_idx[i]])
acc_q = 100 * q_correct / n_test

print(f"  Float:     {acc_test:.1f}%")
print(f"  Quantized: {acc_q:.1f}%")
print(f"  Drop:      {acc_test - acc_q:.1f}%")
if acc_test - acc_q < 5:
    print("  PASS")
else:
    print("  WARNING: drop > 5%")

# ══════════════════════════════════════════════════════════════
# 10. EXPORT FOR VERILOG
# ══════════════════════════════════════════════════════════════
print("\nExporting...")
all_w = np.concatenate([hw_q.flatten(), hb_q.flatten(),
                        ow_q.flatten(), ob_q.flatten()])
with open('weights.hex', 'w') as f:
    f.write(f"// SBP+MLP  8->8->4  {total_bytes} bytes\n")
    f.write(f"// hidden_bias_scale = {hidden_bias_scale}\n")
    f.write(f"// output_bias_scale = {output_bias_scale}\n\n")
    for v in all_w:
        f.write(f"{(int(v) if v >= 0 else int(v)+256):02X}\n")

all_w.astype(np.int8).tofile('weights.bin')

with open('weights_readable.txt', 'w') as f:
    f.write(f"// 8->8->4  hidden_bias_scale={hidden_bias_scale}"
            f"  output_bias_scale={output_bias_scale}\n\n")
    f.write("// hw[neuron][input]:\n")
    for j in range(HIDDEN):
        f.write(f"// hw[{j}] = [{', '.join(f'{int(hw_q[j,i]):+4d}' for i in range(N_CH))}]\n")
    f.write(f"\n// hb: [{', '.join(f'{int(v):+4d}' for v in hb_q)}]\n")
    f.write("\n// ow[class][neuron]:\n")
    for k in range(N_CLASSES):
        f.write(f"// ow[{k}] ({CLASS_NAMES[k]}) = [{', '.join(f'{int(ow_q[k,j]):+4d}' for j in range(HIDDEN))}]\n")
    f.write(f"\n// ob: [{', '.join(f'{int(v):+4d}' for v in ob_q)}]\n")
    f.write(f"\n// Channels: {sorted(int(x) for x in TOP8)}\n")

N_VEC = min(40, n_test)
per_class = {c: [i for i in range(n_test) if y_all[test_idx[i]]==c] for c in range(4)}
selected  = []
for c in range(4): selected.extend(per_class[c][:N_VEC//4])
remaining = [i for i in range(n_test) if i not in selected]
selected.extend(remaining[:N_VEC-len(selected)])

with open('test_vectors.txt', 'w') as f:
    f.write(f"// RTL TEST VECTORS — SBP+MLP 8->8->4\n")
    f.write(f"// Pipeline: ADC(8ch,{SAMPLES}) -> SBP=sum(|x-128|)/{SAMPLES} -> MLP\n")
    f.write(f"// hidden_bias_scale = {hidden_bias_scale}\n")
    f.write(f"// output_bias_scale = {output_bias_scale}\n\n")
    for vi, i in enumerate(selected[:N_VEC]):
        adc_win  = X_all[test_idx[i]]
        pred, sc = golden_model(adc_win)
        true_lbl = int(y_all[test_idx[i]])
        sbp_int  = compute_sbp(adc_win)
        f.write(f"// Vec {vi:02d}: true={CLASS_NAMES[true_lbl]}  pred={CLASS_NAMES[pred]}\n")
        f.write(f"// Expected CLASS[1:0] = {pred:02b} = {pred}\n")
        f.write(f"// Scores: [{sc[0]}, {sc[1]}, {sc[2]}, {sc[3]}]\n")
        f.write(f"// SBP: [{', '.join(str(int(x)) for x in sbp_int)}]\n")
        for ch in range(N_CH):
            vals = ' '.join(f"{int(adc_win[ch,s]):02X}" for s in range(SAMPLES))
            f.write(f"// ch{ch}: {vals}\n")
        f.write('\n')

torch.save(model.state_dict(), 'model.pth')
np.savez('preprocessing.npz',
         sbp_mean=sbp_mean, sbp_std=sbp_std,
         selected_channels=TOP8, v_min=v_min, v_max=v_max)

print("  weights.hex, weights.bin, weights_readable.txt")
print(f"  test_vectors.txt  ({N_VEC} vectors)")
print("  model.pth, preprocessing.npz")

# ══════════════════════════════════════════════════════════════
# 11. SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Trials:       {len(y_all)} ({n_train} train, {n_test} test)")
print(f"  Channels:     {N_CH} of {n_raw} → {sorted(int(x) for x in TOP8)}")
print(f"  Architecture: 8->8->4  ({total_bytes} bytes)")
print(f"  Float acc:    {acc_test:.1f}%  (chance=25%)")
print(f"  Quant acc:    {acc_q:.1f}%")
print(f"  Quant drop:   {acc_test-acc_q:.1f}%")
print()
print("  RTL VERIFICATION:")
print("    1. Scan weights.hex into chip via scan chain")
print("    2. Feed ch0..ch7 from test_vectors.txt into ADC bus")
print("    3. Run SBP -> MLP pipeline")
print("    4. Compare CLASS[1:0] to expected value in vector comments")
print("    5. All vectors must match exactly for RTL sign-off")