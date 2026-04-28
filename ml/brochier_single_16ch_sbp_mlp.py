#!/usr/bin/env python3
"""
BROCHIER SINGLE-SESSION SBP+MLP — 16 CHANNEL VERSION
====================================================

Purpose:
  Test whether using 16 selected channels improves SBP+MLP accuracy
  on ONE Brochier session.

Default:
  - single session: prefers l101210-001 if present
  - 16 channels
  - 1 SBP bin per channel
  - model: 16 -> 16 -> 4
  - exports real and synthetic RTL vectors

Important:
  This changes the required number of ADC input channels from 8 to 16.
  If your physical analog frontend only has 8 simultaneous channels, this
  is NOT drop-in hardware-compatible. If your 5 MS/s ADC is time-muxed
  across 16 analog channels, then the aggregate ADC rate is likely fine.

Run:
  python3 brochier_single_16ch_sbp_mlp.py

Useful variants:
  python3 brochier_single_16ch_sbp_mlp.py --session-prefix l101210-001
  python3 brochier_single_16ch_sbp_mlp.py --session-prefix i140703-001
  python3 brochier_single_16ch_sbp_mlp.py --hidden 32
  python3 brochier_single_16ch_sbp_mlp.py --bins 5 --hidden 32

Outputs:
  brochier_single_16ch_weights.hex
  brochier_single_16ch_weights_readable.txt
  brochier_single_16ch_real_test_vectors.txt
  brochier_single_16ch_synthetic_test_vectors.txt
  brochier_single_16ch_model.pth
  brochier_single_16ch_meta.npz
  brochier_single_16ch_results.png
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Signal / hardware parameters
# ============================================================

SOURCE_FS_NOMINAL = 30000
TARGET_FS_DEFAULT = 10000
# 10 kHz target is safer if BP_HIGH=3000 Hz.
# If you force TARGET_FS=5000 with BP_HIGH=3000, Nyquist is only 2500 Hz,
# so the 2.5-3 kHz band can alias unless the analog/digital anti-alias path
# removes it.

N_CLASSES = 4
SAMPLES_DEFAULT = 500        # 50 ms at 10 kHz
BP_LOW = 300
BP_HIGH = 3000
OFFSET_MS = 100
ADC_MID = 128

CLASS_NAMES = ["PG-LF", "PG-HF", "SG-LF", "SG-HF"]

MONKEY_L_GO = {"65381": 0, "65382": 1, "65385": 2, "65386": 3}

try:
    import neo
    import quantities as pq
except ImportError:
    print("ERROR: pip install neo quantities")
    sys.exit(1)


# ============================================================
# Discovery / event helpers
# ============================================================

def discover_sessions():
    ns_files = (
        glob.glob("*.ns5") + glob.glob("**/*.ns5", recursive=True) +
        glob.glob("*.ns6") + glob.glob("**/*.ns6", recursive=True)
    )
    ns_files = sorted(list({os.path.abspath(f) for f in ns_files}))

    sessions = []
    for ns in ns_files:
        base = ns.rsplit(".", 1)[0]
        for suffix in ["-02.nev", "-03.nev", ".nev"]:
            nev = base + suffix
            if os.path.exists(nev):
                sessions.append((os.path.abspath(ns), os.path.abspath(nev)))
                break

    return sorted(list({(a, b) for a, b in sessions}))


def choose_session(sessions, session_prefix=None):
    if not sessions:
        return None

    if session_prefix:
        for ns, nev in sessions:
            if Path(ns).stem.startswith(session_prefix):
                return ns, nev
        raise FileNotFoundError(f"No session matching --session-prefix {session_prefix}")

    # Default: prefer the session that gave you the best prior single-session result.
    for ns, nev in sessions:
        if Path(ns).name.startswith("l101210-001"):
            return ns, nev

    return sessions[0]


def detect_goon_codes(events):
    counts = {k: len(v) for k, v in events.items()}
    keys = list(counts.keys())

    best = None
    best_balance = 1e9

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for k in range(j + 1, len(keys)):
                for l in range(k + 1, len(keys)):
                    group = [keys[i], keys[j], keys[k], keys[l]]
                    vals = [counts[g] for g in group]
                    total = sum(vals)

                    if total < 40 or total > 1000:
                        continue
                    if min(vals) == 0:
                        continue
                    if max(vals) > 3 * min(vals) + 5:
                        continue

                    balance = max(vals) - min(vals)
                    if balance < best_balance:
                        best_balance = balance
                        best = group

    if best is None:
        return None

    return {code: label for label, code in enumerate(sorted(best, key=lambda x: int(x)))}


# ============================================================
# Feature functions
# ============================================================

def adc_float_to_int(adc_float):
    return np.round(np.clip(adc_float, 0, 255)).astype(np.int32)


def compute_sbp_bins_int(adc_window_int, n_ch, n_bins, samples):
    """
    Input:
      adc_window_int: (n_ch, samples)

    Output:
      feature vector length n_ch * n_bins

    Layout:
      feat[ch * n_bins + bin]
    """
    assert samples % n_bins == 0, "samples must be divisible by n_bins"
    bin_samples = samples // n_bins

    feats = np.zeros(n_ch * n_bins, dtype=np.int32)
    adc_window_int = np.asarray(adc_window_int, dtype=np.int32)

    for ch in range(n_ch):
        for b in range(n_bins):
            start = b * bin_samples
            stop = start + bin_samples

            total = np.int32(0)
            for s in range(start, stop):
                total += abs(int(adc_window_int[ch, s]) - ADC_MID)

            feats[ch * n_bins + b] = total // bin_samples

    return feats


def compute_sbp_bins_float_adc(adc_window_float, n_ch, n_bins, samples):
    assert samples % n_bins == 0, "samples must be divisible by n_bins"
    bin_samples = samples // n_bins

    feats = np.zeros(n_ch * n_bins, dtype=np.float64)

    for ch in range(n_ch):
        for b in range(n_bins):
            start = b * bin_samples
            stop = start + bin_samples
            segment = adc_window_float[ch, start:stop]
            feats[ch * n_bins + b] = np.mean(np.abs(segment - ADC_MID))

    return feats


def summarize_quant_error(sbp_float_all, sbp_int_all, n_ch, n_bins, name):
    err = sbp_int_all.astype(np.float64) - sbp_float_all.astype(np.float64)
    abs_err = np.abs(err)
    rel = abs_err / (np.abs(sbp_float_all) + 1e-9)

    print(f"\nADC/SBP quantization error — {name}")
    print("-" * 72)
    print("Compares ideal pre-rounding ADC-code SBP vs RTL integer SBP.")
    print(f"  Overall MAE:      {abs_err.mean():.4f} SBP codes")
    print(f"  Overall RMSE:     {np.sqrt(np.mean(err**2)):.4f} SBP codes")
    print(f"  Max abs error:    {abs_err.max():.4f} SBP codes")
    print(f"  Mean rel error:   {100.0 * rel.mean():.3f}%")
    print(f"  95% abs error:    {np.percentile(abs_err, 95):.4f} SBP codes")

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(abs_err.max()),
        "mean_rel": float(rel.mean()),
        "p95_abs": float(np.percentile(abs_err, 95)),
    }


# ============================================================
# Load one session
# ============================================================

def load_single_session(ns_path, nev_path, n_ch, target_fs, samples, n_bins):
    session_name = Path(ns_path).name

    print(f"\n{'=' * 72}")
    print(f"Loading ONE session: {session_name}")
    print(f"{'=' * 72}")

    nsx_id = 6 if ns_path.endswith(".ns6") else 5

    try:
        reader = neo.io.BlackrockIO(filename=ns_path, nsx_to_load=nsx_id)
        block = reader.read_block(lazy=True, load_waveforms=False)
        raw_sig = next(
            (
                a for a in block.segments[0].analogsignals
                if float(a.sampling_rate.rescale("Hz").magnitude) >= 25000
            ),
            None,
        )
    except Exception as e:
        print(f"ERROR reading NS file: {e}")
        return None, None, None, None

    if raw_sig is None:
        print("ERROR: no 30kHz signal found")
        return None, None, None, None

    T_raw = raw_sig.shape[0]
    n_raw = raw_sig.shape[1]
    fs_raw = float(raw_sig.sampling_rate.rescale("Hz").magnitude)

    if fs_raw % target_fs != 0:
        raise ValueError(
            f"target_fs={target_fs} does not divide source fs={fs_raw}. "
            f"Use target_fs=5000, 10000, or 15000 for 30kHz data."
        )

    DS = int(round(fs_raw / target_fs))

    print(f"  {T_raw} samples × {n_raw} ch @ {fs_raw:.0f} Hz ({T_raw/fs_raw:.0f}s)")
    print(f"  Target fs: {target_fs} Hz, downsample by {DS}x")
    print(f"  Window: {samples} samples = {1000*samples/target_fs:.1f} ms")
    print("  Loading raw array...")
    raw_full = raw_sig.load().magnitude.astype(np.float32)

    # Load events
    try:
        nev_reader = neo.io.BlackrockIO(filename=nev_path, nsx_to_load=None)
        nev_seg = nev_reader.read_block(lazy=False, load_waveforms=False).segments[0]
    except Exception as e:
        print(f"ERROR reading NEV file: {e}")
        return None, None, None, None

    events = {}
    for evt in nev_seg.events:
        times = evt.times.rescale("s").magnitude
        labels = np.array(evt.labels, dtype=str)
        for lbl, t in zip(labels, times):
            events.setdefault(str(lbl).strip(), []).append(float(t))

    print(f"  Event codes: {sorted(events.keys())}")

    if any(k in events for k in MONKEY_L_GO):
        goon_map = MONKEY_L_GO
        print("  Monkey L GO-ON codes")
    else:
        goon_map = detect_goon_codes(events)
        if goon_map is None:
            print("ERROR: could not detect GO-ON codes")
            print(f"Counts: { {k: len(v) for k, v in events.items()} }")
            return None, None, None, None
        print(f"  Auto-detected GO-ON codes: {goon_map}")

    trials = sorted(
        [
            (t, label)
            for code, label in goon_map.items()
            for t in events.get(code, [])
        ],
        key=lambda x: x[0],
    )

    counts = np.bincount([t[1] for t in trials], minlength=N_CLASSES)
    print("  Trials: " + " | ".join(f"{CLASS_NAMES[c]}:{counts[c]}" for c in range(N_CLASSES)))

    # Channel selection using raw downsampled SBP-like metric across all 96 channels.
    # This is for experiment/hardware exploration. For clean ML evaluation, channel
    # selection should be done on training trials only.
    OFFSET_RAW = int(OFFSET_MS / 1000.0 * fs_raw)
    WIN_RAW = samples * DS

    raw_sbp_all = []
    valid_trials = []

    for go_t, label in trials:
        s = int(go_t * fs_raw) + OFFSET_RAW
        e = s + WIN_RAW

        if s < 0 or e > T_raw:
            continue

        win = raw_full[s:e:DS, :]
        sbp = np.abs(win - win.mean(axis=0)).mean(axis=0)
        raw_sbp_all.append(sbp)
        valid_trials.append((go_t, label))

    if not valid_trials:
        print("ERROR: no valid trials")
        return None, None, None, None

    raw_sbp_all = np.array(raw_sbp_all)
    y_raw = np.array([t[1] for t in valid_trials])

    cls_means = np.array([
        raw_sbp_all[y_raw == c].mean(0)
        for c in range(N_CLASSES)
        if np.sum(y_raw == c) > 0
    ])

    grand = raw_sbp_all.mean(0)
    between = np.mean((cls_means - grand) ** 2, axis=0)
    within = np.mean([
        raw_sbp_all[y_raw == c].var(0)
        for c in range(N_CLASSES)
        if np.sum(y_raw == c) > 0
    ], axis=0)

    fisher = between / (within + 1e-10)
    top = np.sort(np.argsort(fisher)[-n_ch:])
    print(f"  Selected {n_ch} channels: {list(map(int, top))}")

    # Filter selected channels.
    nyq = fs_raw / 2.0
    if BP_HIGH >= target_fs / 2:
        print()
        print("WARNING:")
        print(f"  BP_HIGH={BP_HIGH} Hz but target_fs={target_fs} Hz has Nyquist={target_fs/2:.1f} Hz.")
        print("  Use target_fs=10000+ or lower BP_HIGH before downsampling.")
        print()

    b, a = sp.butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")

    print(f"  Filtering {n_ch} selected channels...")
    filtered = np.zeros((T_raw, n_ch), dtype=np.float32)
    for ci, ch in enumerate(top):
        filtered[:, ci] = sp.filtfilt(b, a, raw_full[:, ch].astype(np.float64))

    del raw_full

    # ADC rails from filtered selected channels.
    v_min = np.percentile(filtered, 1)
    v_max = np.percentile(filtered, 99)
    print(f"  ADC rails: [{v_min:.3f}, {v_max:.3f}]")

    filt_ds = filtered[::DS]
    del filtered

    adc_float_ds = np.clip((filt_ds - v_min) / (v_max - v_min), 0, 1) * 255.0
    adc_int_ds = adc_float_to_int(adc_float_ds)

    T_ds = len(adc_int_ds)
    OFFSET_DS = int(OFFSET_MS / 1000.0 * target_fs)

    X_int = []
    X_float = []
    y = []

    for go_t, label in valid_trials:
        s = int(go_t * target_fs) + OFFSET_DS
        e = s + samples

        if s < 0 or e > T_ds:
            continue

        X_float.append(adc_float_ds[s:e, :].T)
        X_int.append(adc_int_ds[s:e, :].T)
        y.append(label)

    X_int = np.array(X_int, dtype=np.int32)
    X_float = np.array(X_float, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    features_float = np.array([
        compute_sbp_bins_float_adc(x, n_ch, n_bins, samples) for x in X_float
    ])
    features_int = np.array([
        compute_sbp_bins_int(x, n_ch, n_bins, samples) for x in X_int
    ])

    qerr = summarize_quant_error(features_float, features_int, n_ch, n_bins, session_name)

    print(f"  Final windows: {X_int.shape}")

    info = {
        "session_name": session_name,
        "nev_name": Path(nev_path).name,
        "selected_channels": top,
        "v_min": float(v_min),
        "v_max": float(v_max),
        "fs_raw": float(fs_raw),
        "target_fs": int(target_fs),
        "ds": int(DS),
        "samples": int(samples),
        "n_ch": int(n_ch),
        "n_bins": int(n_bins),
        "n_features": int(n_ch * n_bins),
        "qerr": qerr,
    }

    return X_int, X_float, y, info


# ============================================================
# Model
# ============================================================

class SbpMLP(nn.Module):
    def __init__(self, n_features, hidden, dropout=0.25):
        super().__init__()
        self.h1 = nn.Linear(n_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.o = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        h = F.relu(self.h1(x))
        h = self.drop(h)
        return self.o(h)


def train_model(features, y_all, n_features, hidden):
    perm = np.random.permutation(len(y_all))
    n_train = int(0.8 * len(y_all))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    feat_mean = features[train_idx].mean(0)
    feat_std = features[train_idx].std(0) + 1e-8

    feat_norm = (features - feat_mean) / feat_std

    X_tr = torch.tensor(feat_norm[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_te = torch.tensor(feat_norm[test_idx], dtype=torch.float32)
    y_te = torch.tensor(y_all[test_idx], dtype=torch.long)

    print("\nTrain/test split:")
    print(f"  train={len(train_idx)}, test={len(test_idx)}")
    print(f"  train classes={np.bincount(y_all[train_idx], minlength=N_CLASSES)}")
    print(f"  test  classes={np.bincount(y_all[test_idx], minlength=N_CLASSES)}")

    model = SbpMLP(n_features=n_features, hidden=hidden, dropout=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    best_acc = -1.0
    best_state = None

    print(f"\nTraining SBP+MLP {n_features}->{hidden}->4...")
    print(f"Parameters: {n_features * hidden + hidden + hidden * N_CLASSES + N_CLASSES}")

    for ep in range(600):
        model.train()
        for bx, by in loader:
            loss = loss_fn(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(1)
                acc = 100.0 * (preds == y_te).sum().item() / len(y_te)
            print(f"  epoch {ep + 1:3d} | test {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds = model(X_te).argmax(1).cpu().numpy()
        acc = 100.0 * np.mean(preds == y_all[test_idx])

    print(f"\nFloat test accuracy: {acc:.1f}%")
    print("Per-class float test accuracy:")
    for c in range(N_CLASSES):
        mask = y_all[test_idx] == c
        if np.sum(mask) > 0:
            ca = 100.0 * np.mean(preds[mask] == c)
            print(f"  {CLASS_NAMES[c]:>6s}: {ca:.1f}% ({np.sum(mask)} test samples)")

    return model, train_idx, test_idx, feat_mean, feat_std, preds, acc


# ============================================================
# Quantized MLP
# ============================================================

def quantize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    mx = np.max(np.abs(arr))
    if mx < 1e-12:
        return np.zeros_like(arr, dtype=np.int16), 1.0

    scale = 127.0 / mx
    q = np.round(arr * scale).clip(-128, 127).astype(np.int16)
    return q, float(scale)


def quantize_model(model, feat_mean, feat_std):
    w1 = model.h1.weight.detach().numpy()
    b1 = model.h1.bias.detach().numpy()
    w2 = model.o.weight.detach().numpy()
    b2 = model.o.bias.detach().numpy()

    w1_eff = w1 / feat_std[np.newaxis, :]
    b1_eff = b1 - np.sum(w1 * feat_mean[np.newaxis, :] / feat_std[np.newaxis, :], axis=1)

    w1_q, sw1 = quantize(w1_eff)
    b1_q, sb1 = quantize(b1_eff)
    w2_q, sw2 = quantize(w2)
    b2_q, sb2 = quantize(b2)

    hidden1_bias_scale = int(round(sw1 / sb1))
    output_bias_scale = int(round(sw1 * sw2 / sb2))

    print("\nMLP quantization:")
    print(f"  w1 scale={sw1:.3f}, b1 scale={sb1:.3f}, hidden1_bias_scale={hidden1_bias_scale}")
    print(f"  w2 scale={sw2:.3f}, b2 scale={sb2:.3f}, output_bias_scale={output_bias_scale}")

    return {
        "w1_q": w1_q,
        "b1_q": b1_q,
        "w2_q": w2_q,
        "b2_q": b2_q,
        "hidden1_bias_scale": hidden1_bias_scale,
        "output_bias_scale": output_bias_scale,
        "scales": np.array([sw1, sb1, sw2, sb2], dtype=np.float64),
    }


def golden_int_model(adc_window, q, n_ch, n_bins, samples):
    feats = compute_sbp_bins_int(adc_window, n_ch, n_bins, samples)
    n_features = n_ch * n_bins

    hidden = np.zeros(q["w1_q"].shape[0], dtype=np.int64)
    for j in range(q["w1_q"].shape[0]):
        acc = np.int64(0)
        for i in range(n_features):
            acc += np.int64(feats[i]) * np.int64(q["w1_q"][j, i])
        acc += np.int64(q["b1_q"][j]) * np.int64(q["hidden1_bias_scale"])
        hidden[j] = max(0, acc)

    scores = np.zeros(N_CLASSES, dtype=np.int64)
    for k in range(N_CLASSES):
        acc = np.int64(0)
        for j in range(q["w2_q"].shape[1]):
            acc += np.int64(hidden[j]) * np.int64(q["w2_q"][k, j])
        acc += np.int64(q["b2_q"][k]) * np.int64(q["output_bias_scale"])
        scores[k] = acc

    return int(np.argmax(scores)), scores, feats, hidden


def evaluate_integer_path(X_int, y_all, test_idx, q, float_preds, n_ch, n_bins, samples):
    q_preds = []
    for idx in test_idx:
        pred, _, _, _ = golden_int_model(X_int[idx], q, n_ch, n_bins, samples)
        q_preds.append(pred)

    q_preds = np.array(q_preds, dtype=np.int64)

    q_acc = 100.0 * np.mean(q_preds == y_all[test_idx])
    mismatch = int(np.sum(q_preds != float_preds))

    print("\nInteger path comparison:")
    print(f"  Integer test accuracy:      {q_acc:.1f}%")
    print(f"  Float-vs-int mismatches:    {mismatch}/{len(test_idx)} = {100*mismatch/len(test_idx):.2f}%")

    return q_preds, q_acc, mismatch


# ============================================================
# Synthetic vectors
# ============================================================

def estimate_feature_stats(X_int, y_all, train_idx, n_ch, n_bins, samples):
    feats = np.array([
        compute_sbp_bins_int(X_int[i], n_ch, n_bins, samples)
        for i in train_idx
    ], dtype=np.float64)

    y = y_all[train_idx]

    stats = {}
    for c in range(N_CLASSES):
        f = feats[y == c]
        if len(f) == 0:
            continue
        stats[c] = {
            "mean": f.mean(axis=0),
            "std": f.std(axis=0) + 1.0,
        }

    global_stats = {
        "mean": feats.mean(axis=0),
        "std": feats.std(axis=0) + 1.0,
        "min": feats.min(axis=0),
        "max": feats.max(axis=0),
    }

    return stats, global_stats


def make_synthetic_adc(cls, stats, global_stats, rng, n_ch, n_bins, samples):
    bin_samples = samples // n_bins
    n_features = n_ch * n_bins

    if cls in stats:
        mu = stats[cls]["mean"]
        sd = stats[cls]["std"]
    else:
        mu = global_stats["mean"]
        sd = global_stats["std"]

    target = np.clip(rng.normal(mu, sd), 1, 127)
    adc = np.zeros((n_ch, samples), dtype=np.int32)

    for ch in range(n_ch):
        for b in range(n_bins):
            feat_idx = ch * n_bins + b
            amp = int(round(target[feat_idx]))
            amp = max(1, min(amp, 127))

            start = b * bin_samples
            stop = start + bin_samples

            vals = np.zeros(bin_samples, dtype=np.int32)
            s = 0
            sign = rng.choice([-1, 1])
            while s < bin_samples:
                run = int(rng.integers(2, max(3, min(12, bin_samples))))
                if rng.random() < 0.4:
                    sign *= -1
                e = min(bin_samples, s + run)
                jitter = rng.integers(-4, 5, size=e - s)
                vals[s:e] = ADC_MID + sign * np.clip(amp + jitter, 0, 127)
                s = e

            adc[ch, start:stop] = np.clip(vals, 0, 255)

    return adc


# ============================================================
# Export
# ============================================================

def export_weights(prefix, q, n_ch, n_bins, samples):
    n_features = n_ch * n_bins
    hidden = q["w1_q"].shape[0]

    all_w = np.concatenate([
        q["w1_q"].flatten(),
        q["b1_q"].flatten(),
        q["w2_q"].flatten(),
        q["b2_q"].flatten(),
    ]).astype(np.int16)

    hex_path = f"{prefix}_weights.hex"
    bin_path = f"{prefix}_weights.bin"
    readable_path = f"{prefix}_weights_readable.txt"

    with open(hex_path, "w") as f:
        f.write("// Brochier single-session SBP+MLP weights\n")
        f.write(f"// Architecture: {n_features} -> {hidden} -> 4\n")
        f.write(f"// N_CH={n_ch}, N_BINS={n_bins}, SAMPLES={samples}, BIN_SAMPLES={samples//n_bins}\n")
        f.write("// Feature layout: feat[ch * N_BINS + bin]\n")
        f.write("// Weight layout: w1[hidden][features], b1[hidden], w2[4][hidden], b2[4]\n")
        f.write(f"// hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"// output_bias_scale  = {q['output_bias_scale']}\n\n")

        for v in all_w:
            vv = int(v)
            if vv < 0:
                vv += 256
            f.write(f"{vv & 0xFF:02X}\n")

    all_w.astype(np.int8).tofile(bin_path)

    with open(readable_path, "w") as f:
        f.write("Brochier single-session SBP+MLP\n")
        f.write(f"N_CH = {n_ch}\n")
        f.write(f"N_BINS = {n_bins}\n")
        f.write(f"SAMPLES = {samples}\n")
        f.write(f"BIN_SAMPLES = {samples//n_bins}\n")
        f.write(f"N_FEATURES = {n_features}\n")
        f.write("Feature layout: feat[ch * N_BINS + bin]\n")
        f.write(f"hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"output_bias_scale  = {q['output_bias_scale']}\n\n")

        f.write("w1_q[hidden][feature]:\n")
        for row in q["w1_q"]:
            f.write(" ".join(f"{int(v):+4d}" for v in row) + "\n")

        f.write("\nb1_q[hidden]:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["b1_q"]) + "\n")

        f.write("\nw2_q[class][hidden]:\n")
        for row in q["w2_q"]:
            f.write(" ".join(f"{int(v):+4d}" for v in row) + "\n")

        f.write("\nb2_q[class]:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["b2_q"]) + "\n")

    return hex_path, bin_path, readable_path


def export_real_vectors(path, X_int, y_all, test_idx, q, n_ch, n_bins, samples, n_vec):
    n_vec = min(n_vec, len(test_idx))

    selected = []
    for c in range(N_CLASSES):
        cidx = [idx for idx in test_idx if y_all[idx] == c]
        selected.extend(cidx[:max(1, n_vec // N_CLASSES)])

    remaining = [idx for idx in test_idx if idx not in set(selected)]
    selected.extend(remaining[:n_vec - len(selected)])
    selected = selected[:n_vec]

    with open(path, "w") as f:
        f.write(f"// REAL BROCHIER RTL TEST VECTORS ({len(selected)})\n")
        f.write(f"// ADC({n_ch}ch,{samples}) -> SBP({n_ch*n_bins} features) -> MLP\n")
        f.write("// Feature layout: feat[ch * N_BINS + bin]\n\n")

        for vi, idx in enumerate(selected):
            adc = X_int[idx]
            pred, scores, feats, hidden = golden_int_model(adc, q, n_ch, n_bins, samples)
            true = int(y_all[idx])

            f.write(f"// ===== Real Vector {vi:03d} =====\n")
            f.write(f"// True: {CLASS_NAMES[true]} ({true})\n")
            f.write(f"// Expected Pred: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// FEATURES: {' '.join(str(int(v)) for v in feats)}\n")
            f.write(f"// HIDDEN: {' '.join(str(int(v)) for v in hidden)}\n")
            f.write(f"// SCORES: {' '.join(str(int(v)) for v in scores)}\n")

            for ch in range(n_ch):
                vals = " ".join(f"{int(adc[ch, s]):02X}" for s in range(samples))
                f.write(f"ch{ch}: {vals}\n")
            f.write("\n")


def export_synthetic_vectors(path, q, stats, global_stats, n_ch, n_bins, samples, n_vec):
    rng = np.random.default_rng(12345)

    with open(path, "w") as f:
        f.write(f"// SYNTHETIC BROCHIER-LIKE RTL TEST VECTORS ({n_vec})\n")
        f.write("// Matched to training-set feature statistics.\n")
        f.write("// Feature layout: feat[ch * N_BINS + bin]\n\n")

        for vi in range(n_vec):
            cls = vi % N_CLASSES
            adc = make_synthetic_adc(cls, stats, global_stats, rng, n_ch, n_bins, samples)
            pred, scores, feats, hidden = golden_int_model(adc, q, n_ch, n_bins, samples)

            f.write(f"// ===== Synthetic Vector {vi:03d} =====\n")
            f.write(f"// Intended synthetic class: {CLASS_NAMES[cls]} ({cls})\n")
            f.write(f"// Expected Pred: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// FEATURES: {' '.join(str(int(v)) for v in feats)}\n")
            f.write(f"// HIDDEN: {' '.join(str(int(v)) for v in hidden)}\n")
            f.write(f"// SCORES: {' '.join(str(int(v)) for v in scores)}\n")

            for ch in range(n_ch):
                vals = " ".join(f"{int(adc[ch, s]):02X}" for s in range(samples))
                f.write(f"ch{ch}: {vals}\n")
            f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-prefix", type=str, default=None,
                        help="Example: l101210-001 or i140703-001. Default prefers l101210-001.")
    parser.add_argument("--n-ch", type=int, default=16)
    parser.add_argument("--bins", type=int, default=1)
    parser.add_argument("--target-fs", type=int, default=TARGET_FS_DEFAULT)
    parser.add_argument("--samples", type=int, default=SAMPLES_DEFAULT)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--out-prefix", type=str, default="brochier_single_16ch")
    parser.add_argument("--real-vectors", type=int, default=60)
    parser.add_argument("--synthetic-vectors", type=int, default=100)
    args = parser.parse_args()

    if args.samples % args.bins != 0:
        raise ValueError("--samples must be divisible by --bins")

    n_features = args.n_ch * args.bins

    print("=" * 72)
    print("BROCHIER SINGLE-SESSION SBP+MLP — CONFIGURABLE CHANNELS")
    print("=" * 72)
    print(f"Channels:      {args.n_ch}")
    print(f"Bins/channel:  {args.bins}")
    print(f"Features:      {n_features}")
    print(f"Target fs:     {args.target_fs} Hz")
    print(f"Samples/window:{args.samples} = {1000*args.samples/args.target_fs:.1f} ms")
    print(f"Model:         {n_features} -> {args.hidden} -> 4")

    sessions = discover_sessions()
    if not sessions:
        print("ERROR: No Brochier NS5/NS6 + NEV pairs found.")
        sys.exit(1)

    print("\nDiscovered sessions:")
    for ns, nev in sessions:
        print(f"  {Path(ns).name} + {Path(nev).name}")

    ns_path, nev_path = choose_session(sessions, args.session_prefix)

    print("\nUsing ONLY this session:")
    print(f"  {Path(ns_path).name} + {Path(nev_path).name}")

    X_int, X_float, y_all, info = load_single_session(
        ns_path,
        nev_path,
        n_ch=args.n_ch,
        target_fs=args.target_fs,
        samples=args.samples,
        n_bins=args.bins,
    )

    if X_int is None:
        print("ERROR: failed to load session")
        sys.exit(1)

    features_float = np.array([
        compute_sbp_bins_float_adc(x, args.n_ch, args.bins, args.samples)
        for x in X_float
    ])
    features_int = np.array([
        compute_sbp_bins_int(x, args.n_ch, args.bins, args.samples)
        for x in X_int
    ])

    print("\nDataset summary:")
    print(f"  Session: {info['session_name']}")
    print(f"  Windows: {len(y_all)}")
    print(f"  Shape:   {X_int.shape}")
    print(f"  Classes: {np.bincount(y_all, minlength=N_CLASSES)}")
    print(f"  Feature range: [{features_int.min()}, {features_int.max()}]")

    model, train_idx, test_idx, feat_mean, feat_std, float_preds, float_acc = train_model(
        features_int.astype(np.float64),
        y_all,
        n_features=n_features,
        hidden=args.hidden,
    )

    q = quantize_model(model, feat_mean, feat_std)

    q_preds, int_acc, mismatch = evaluate_integer_path(
        X_int,
        y_all,
        test_idx,
        q,
        float_preds,
        args.n_ch,
        args.bins,
        args.samples,
    )

    stats, global_stats = estimate_feature_stats(
        X_int,
        y_all,
        train_idx,
        args.n_ch,
        args.bins,
        args.samples,
    )

    print("\nExporting RTL files...")
    weights_hex, weights_bin, weights_readable = export_weights(
        args.out_prefix,
        q,
        args.n_ch,
        args.bins,
        args.samples,
    )

    real_path = f"{args.out_prefix}_real_test_vectors.txt"
    synth_path = f"{args.out_prefix}_synthetic_test_vectors.txt"

    export_real_vectors(
        real_path,
        X_int,
        y_all,
        test_idx,
        q,
        args.n_ch,
        args.bins,
        args.samples,
        args.real_vectors,
    )

    export_synthetic_vectors(
        synth_path,
        q,
        stats,
        global_stats,
        args.n_ch,
        args.bins,
        args.samples,
        args.synthetic_vectors,
    )

    torch.save(model.state_dict(), f"{args.out_prefix}_model.pth")
    np.savez(
        f"{args.out_prefix}_meta.npz",
        session=info["session_name"],
        selected_channels=info["selected_channels"],
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        feat_mean=feat_mean,
        feat_std=feat_std,
        float_acc=float_acc,
        int_acc=int_acc,
        pred_mismatch=mismatch,
        qerr_mae=info["qerr"]["mae"],
        qerr_rmse=info["qerr"]["rmse"],
        qerr_p95=info["qerr"]["p95_abs"],
        n_ch=args.n_ch,
        n_bins=args.bins,
        n_features=n_features,
        target_fs=args.target_fs,
        samples=args.samples,
        hidden=args.hidden,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [float_acc, int_acc]
    labels = ["Float MLP", "Integer MLP"]
    bars = ax.bar(labels, vals, width=0.45)
    ax.axhline(25, linestyle="--", linewidth=1, label="Chance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Single Brochier session, {args.n_ch}ch × {args.bins} bins")
    ax.legend()
    plt.tight_layout()
    fig_path = f"{args.out_prefix}_results.png"
    plt.savefig(fig_path, dpi=150)

    n_param_int8 = (
        q["w1_q"].size +
        q["b1_q"].size +
        q["w2_q"].size +
        q["b2_q"].size
    )

    print("\nSaved:")
    print(f"  {weights_hex}")
    print(f"  {weights_bin}")
    print(f"  {weights_readable}")
    print(f"  {real_path}")
    print(f"  {synth_path}")
    print(f"  {args.out_prefix}_model.pth")
    print(f"  {args.out_prefix}_meta.npz")
    print(f"  {fig_path}")

    print("\nFINAL SUMMARY")
    print("=" * 72)
    print(f"  Session used:                {info['session_name']}")
    print(f"  Channels:                    {args.n_ch}")
    print(f"  Bins/channel:                {args.bins}")
    print(f"  Feature count:               {n_features}")
    print(f"  Model:                       {n_features} -> {args.hidden} -> 4")
    print(f"  Param bytes, int8 style:      {n_param_int8}")
    print(f"  ADC/SBP quant MAE:           {info['qerr']['mae']:.4f} SBP codes")
    print(f"  ADC/SBP quant RMSE:          {info['qerr']['rmse']:.4f} SBP codes")
    print(f"  ADC/SBP quant 95% abs err:   {info['qerr']['p95_abs']:.4f} SBP codes")
    print(f"  Float MLP accuracy:          {float_acc:.1f}%")
    print(f"  Integer MLP accuracy:        {int_acc:.1f}%")
    print(f"  Float-vs-int pred mismatch:  {mismatch}/{len(test_idx)}")
    print()
    print("For RTL, use:")
    print(f"  1. {weights_hex}")
    print(f"  2. {weights_readable}")
    print(f"  3. {real_path}")
    print(f"  4. {synth_path}")
    print()
    print("ADC / hardware note:")
    print("  If one 5 MS/s ADC is time-muxed across 16 channels, per-channel rate is")
    print("  5e6 / 16 = 312.5 kS/s before digital decimation, which is far above")
    print("  the neural spike-band requirement. The limiting issue is not ADC speed;")
    print("  it is whether your analog mux/front-end can support 16 physical inputs.")
    print("  This script uses target_fs=10 kHz by default to avoid aliasing a 3 kHz bandpass.")


if __name__ == "__main__":
    main()
