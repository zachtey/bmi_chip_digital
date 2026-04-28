#!/usr/bin/env python3
"""
BROCHIER SBP+MLP — BOTH SESSIONS + ADC/SBP QUANTIZATION ERROR + SYNTH VECTORS
=============================================================================

This version is for answering the hardware question:

  How much numerical error do we introduce when we mimic the ADC/downsample path?

It separates these metrics:

  1. ADC/SBP quantization error:
       Compare ideal floating ADC-code SBP before rounding
       vs integer RTL SBP after 8-bit ADC rounding.

  2. MLP integer decision difference:
       Compare float MLP vs integer golden MLP predictions.

  3. Final classification accuracy:
       Float and integer accuracy on held-out real Brochier trials.

Default:
  Loads ALL discovered Brochier sessions:
    i140703-001.ns6 + i140703-001-03.nev
    l101210-001.ns5 + l101210-001-02.nev

Run:
  python3 brochier_both_sessions_quant_error.py

Optional:
  python3 brochier_both_sessions_quant_error.py --session-prefix l101210-001
  python3 brochier_both_sessions_quant_error.py --session-prefix i140703-001
  python3 brochier_both_sessions_quant_error.py --hidden 16
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

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Hardware parameters
# ============================================================
TARGET_FS   = 5000
N_CH        = 8
SAMPLES     = 250
N_CLASSES   = 4
BP_LOW      = 300
BP_HIGH     = 3000
OFFSET_MS   = 100
ADC_MID     = 128

CLASS_NAMES = ["PG-LF", "PG-HF", "SG-LF", "SG-HF"]

MONKEY_L_GO = {"65381": 0, "65382": 1, "65385": 2, "65386": 3}

try:
    import neo
    import quantities as pq
except ImportError:
    print("ERROR: pip install neo quantities")
    sys.exit(1)


# ============================================================
# Session helpers
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


def filter_sessions(sessions, session_prefix):
    if session_prefix is None:
        return sessions
    out = []
    for ns, nev in sessions:
        if Path(ns).stem.startswith(session_prefix):
            out.append((ns, nev))
    if not out:
        raise FileNotFoundError(f"No session matching --session-prefix {session_prefix}")
    return out


def detect_goon_codes(events):
    counts = {k: len(v) for k, v in events.items()}
    keys = list(counts.keys())
    n = len(keys)

    best = None
    best_balance = 1e9

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
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

    best_sorted = sorted(best, key=lambda x: int(x))
    return {code: label for label, code in enumerate(best_sorted)}


# ============================================================
# Quantization-error metrics
# ============================================================
def adc_float_to_int(adc_float):
    return np.round(np.clip(adc_float, 0, 255)).astype(np.int32)


def compute_sbp_int(adc_window_int):
    """
    RTL-style integer SBP:
      sum(abs(adc - 128)) // 250
    """
    adc_window_int = np.asarray(adc_window_int, dtype=np.int32)
    sbp = np.zeros(N_CH, dtype=np.int32)
    for ch in range(N_CH):
        total = np.int32(0)
        for s in range(SAMPLES):
            total += abs(int(adc_window_int[ch, s]) - ADC_MID)
        sbp[ch] = total // SAMPLES
    return sbp


def compute_sbp_float_adc(adc_window_float):
    """
    Ideal SBP in ADC-code units before rounding/flooring:
      mean(abs(adc_float - 128))
    """
    return np.mean(np.abs(adc_window_float - ADC_MID), axis=1)


def summarize_quant_error(sbp_float_all, sbp_int_all, name):
    """
    sbp_float_all: (N, 8)
    sbp_int_all:   (N, 8)
    """
    err = sbp_int_all.astype(np.float64) - sbp_float_all.astype(np.float64)
    abs_err = np.abs(err)

    mae = abs_err.mean()
    rmse = np.sqrt(np.mean(err ** 2))
    max_abs = abs_err.max()
    rel = abs_err / (np.abs(sbp_float_all) + 1e-9)

    print(f"\nADC/SBP quantization error — {name}")
    print("-" * 70)
    print("This compares ideal pre-rounding ADC-code SBP vs RTL integer SBP.")
    print(f"  Overall MAE:      {mae:.4f} SBP codes")
    print(f"  Overall RMSE:     {rmse:.4f} SBP codes")
    print(f"  Max abs error:    {max_abs:.4f} SBP codes")
    print(f"  Mean rel error:   {100.0 * rel.mean():.3f}%")
    print(f"  95% abs error:    {np.percentile(abs_err, 95):.4f} SBP codes")
    print("  Per-channel MAE:")
    for ch in range(N_CH):
        print(f"    ch{ch}: {abs_err[:, ch].mean():.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_rel": rel.mean(),
        "p95_abs": np.percentile(abs_err, 95),
        "per_ch_mae": abs_err.mean(axis=0),
    }


# ============================================================
# Load one session
# ============================================================
def load_session(ns_path, nev_path):
    """
    Returns:
      X_int:       (N, 8, 250) int ADC windows for RTL
      X_float_adc: (N, 8, 250) float ADC-code windows before rounding
      y:           (N,) labels
      info:        dict
    """
    session_name = Path(ns_path).name
    print(f"\n{'=' * 70}")
    print(f"Loading session: {session_name}")
    print(f"{'=' * 70}")

    nsx_id = 6 if ns_path.endswith(".ns6") else 5

    try:
        reader = neo.io.BlackrockIO(filename=ns_path, nsx_to_load=nsx_id)
        block = reader.read_block(lazy=True, load_waveforms=False)
        raw_sig = next(
            (a for a in block.segments[0].analogsignals
             if float(a.sampling_rate.rescale("Hz").magnitude) >= 25000),
            None,
        )
    except Exception as e:
        print(f"ERROR reading NS file: {e}")
        return None, None, None, None

    if raw_sig is None:
        print("ERROR: No 30kHz signal found")
        return None, None, None, None

    T_raw = raw_sig.shape[0]
    n_raw = raw_sig.shape[1]
    fs_raw = float(raw_sig.sampling_rate.rescale("Hz").magnitude)
    DS = int(round(fs_raw / TARGET_FS))

    print(f"  {T_raw} samples × {n_raw} ch @ {fs_raw:.0f} Hz ({T_raw / fs_raw:.0f}s)")
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
            print("ERROR: Could not detect GO-ON codes")
            print(f"Counts: { {k: len(v) for k, v in events.items()} }")
            return None, None, None, None
        print(f"  Auto-detected GO-ON codes: {goon_map}")

    trials = sorted(
        [(t, label) for code, label in goon_map.items()
         for t in events.get(code, [])],
        key=lambda x: x[0],
    )

    counts = np.bincount([t[1] for t in trials], minlength=N_CLASSES)
    print("  Trials: " + " | ".join(f"{CLASS_NAMES[c]}:{counts[c]}" for c in range(N_CLASSES)))

    # Select channels using quick raw SBP statistic
    OFFSET_RAW = int(OFFSET_MS / 1000.0 * fs_raw)
    WIN_RAW = SAMPLES * DS

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
    top8 = np.sort(np.argsort(fisher)[-N_CH:])
    print(f"  Selected channels: {list(map(int, top8))}")

    # Filter selected channels at 30kHz, then downsample
    nyq = fs_raw / 2.0
    b, a = sp.butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")

    filtered_8 = np.zeros((T_raw, N_CH), dtype=np.float32)
    for ci, ch in enumerate(top8):
        filtered_8[:, ci] = sp.filtfilt(b, a, raw_full[:, ch].astype(np.float64))

    del raw_full

    v_min = np.percentile(filtered_8, 1)
    v_max = np.percentile(filtered_8, 99)
    print(f"  ADC rails: [{v_min:.3f}, {v_max:.3f}]")

    filt_ds = filtered_8[::DS]
    del filtered_8

    adc_float_ds = np.clip((filt_ds - v_min) / (v_max - v_min), 0, 1) * 255.0
    adc_int_ds = adc_float_to_int(adc_float_ds)

    T_ds = len(adc_int_ds)
    OFFSET_DS = int(OFFSET_MS / 1000.0 * TARGET_FS)

    X_int = []
    X_float_adc = []
    y = []

    for go_t, label in valid_trials:
        s = int(go_t * TARGET_FS) + OFFSET_DS
        e = s + SAMPLES
        if s < 0 or e > T_ds:
            continue

        # transpose to (8,250)
        X_float_adc.append(adc_float_ds[s:e, :].T)
        X_int.append(adc_int_ds[s:e, :].T)
        y.append(label)

    X_int = np.array(X_int, dtype=np.int32)
    X_float_adc = np.array(X_float_adc, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    sbp_float = np.array([compute_sbp_float_adc(x) for x in X_float_adc])
    sbp_int = np.array([compute_sbp_int(x) for x in X_int])
    qerr = summarize_quant_error(sbp_float, sbp_int, session_name)

    print(f"  Final windows: {X_int.shape}")

    info = {
        "session_name": session_name,
        "nev_name": Path(nev_path).name,
        "selected_channels": top8,
        "v_min": v_min,
        "v_max": v_max,
        "fs_raw": fs_raw,
        "ds": DS,
        "sbp_quant_error": qerr,
    }

    return X_int, X_float_adc, y, info


# ============================================================
# Model
# ============================================================
class SbpMLP(nn.Module):
    def __init__(self, hidden=8, dropout=0.3):
        super().__init__()
        self.h1 = nn.Linear(N_CH, hidden)
        self.drop = nn.Dropout(dropout)
        self.o = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        return self.o(self.drop(F.relu(self.h1(x))))


def train_model(sbp_int_features, y_all, hidden=8):
    perm = np.random.permutation(len(y_all))
    n_train = int(0.8 * len(y_all))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    sbp_mean = sbp_int_features[train_idx].mean(0)
    sbp_std = sbp_int_features[train_idx].std(0) + 1e-8
    sbp_norm = (sbp_int_features - sbp_mean) / sbp_std

    X_tr = torch.tensor(sbp_norm[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_te = torch.tensor(sbp_norm[test_idx], dtype=torch.float32)
    y_te = torch.tensor(y_all[test_idx], dtype=torch.long)

    print("\nTrain/test split:")
    print(f"  train={len(train_idx)}, test={len(test_idx)}")
    print(f"  train classes={np.bincount(y_all[train_idx], minlength=N_CLASSES)}")
    print(f"  test  classes={np.bincount(y_all[test_idx], minlength=N_CLASSES)}")

    model = SbpMLP(hidden=hidden, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    best_acc = -1
    best_state = None

    print(f"\nTraining SBP+MLP 8->{hidden}->4...")
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
            print(f"  epoch {ep+1:3d} | test {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_te)
        preds = logits.argmax(1).cpu().numpy()
        acc = 100.0 * np.mean(preds == y_all[test_idx])

    print(f"\nFloat test accuracy: {acc:.1f}%")
    return model, train_idx, test_idx, sbp_mean, sbp_std, preds, acc


# ============================================================
# MLP quantization
# ============================================================
def quantize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    mx = np.max(np.abs(arr))
    if mx < 1e-12:
        return np.zeros_like(arr, dtype=np.int16), 1.0
    sc = 127.0 / mx
    return np.round(arr * sc).clip(-128, 127).astype(np.int16), float(sc)


def quantize_model(model, sbp_mean, sbp_std):
    w1 = model.h1.weight.detach().numpy()
    b1 = model.h1.bias.detach().numpy()
    w2 = model.o.weight.detach().numpy()
    b2 = model.o.bias.detach().numpy()

    w1_eff = w1 / sbp_std[np.newaxis, :]
    b1_eff = b1 - np.sum(w1 * sbp_mean[np.newaxis, :] / sbp_std[np.newaxis, :], axis=1)

    w1_q, sw1 = quantize(w1_eff)
    b1_q, sb1 = quantize(b1_eff)
    w2_q, sw2 = quantize(w2)
    b2_q, sb2 = quantize(b2)

    hidden1_bias_scale = int(round(sw1 / sb1))
    output_bias_scale = int(round(sw1 * sw2 / sb2))

    print("\nMLP weight/bias quantization:")
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


def golden_int_model(adc_window, q):
    sbp = compute_sbp_int(adc_window)

    hidden = np.zeros(q["w1_q"].shape[0], dtype=np.int64)
    for j in range(q["w1_q"].shape[0]):
        acc = np.int64(0)
        for i in range(N_CH):
            acc += np.int64(sbp[i]) * np.int64(q["w1_q"][j, i])
        acc += np.int64(q["b1_q"][j]) * np.int64(q["hidden1_bias_scale"])
        hidden[j] = max(0, acc)

    scores = np.zeros(N_CLASSES, dtype=np.int64)
    for k in range(N_CLASSES):
        acc = np.int64(0)
        for j in range(q["w2_q"].shape[1]):
            acc += np.int64(hidden[j]) * np.int64(q["w2_q"][k, j])
        acc += np.int64(q["b2_q"][k]) * np.int64(q["output_bias_scale"])
        scores[k] = acc

    return int(np.argmax(scores)), scores, sbp, hidden


def evaluate_integer_path(X_int, y_all, test_idx, q, float_preds):
    q_preds = []
    for idx in test_idx:
        pred, _, _, _ = golden_int_model(X_int[idx], q)
        q_preds.append(pred)
    q_preds = np.array(q_preds, dtype=np.int64)

    q_acc = 100.0 * np.mean(q_preds == y_all[test_idx])
    mismatch = int(np.sum(q_preds != float_preds))

    print("\nInteger MLP path comparison:")
    print(f"  Integer test accuracy:        {q_acc:.1f}%")
    print(f"  Float-vs-int pred mismatch:   {mismatch}/{len(test_idx)} = {100*mismatch/len(test_idx):.2f}%")
    print("  Note: accuracy drop can be 0 even if some predictions changed.")

    return q_preds, q_acc, mismatch


# ============================================================
# Synthetic vectors
# ============================================================
def estimate_sbp_stats(X_int, y_all, train_idx):
    sbp = np.array([compute_sbp_int(X_int[i]) for i in train_idx], dtype=np.float64)
    y = y_all[train_idx]

    stats = {}
    for c in range(N_CLASSES):
        s = sbp[y == c]
        if len(s) == 0:
            continue
        stats[c] = {
            "mean": s.mean(axis=0),
            "std": s.std(axis=0) + 1.0,
        }

    global_stats = {
        "mean": sbp.mean(axis=0),
        "std": sbp.std(axis=0) + 1.0,
        "min": sbp.min(axis=0),
        "max": sbp.max(axis=0),
    }

    return stats, global_stats


def make_synthetic_adc(cls, stats, global_stats, rng):
    if cls in stats:
        mu = stats[cls]["mean"]
        sd = stats[cls]["std"]
    else:
        mu = global_stats["mean"]
        sd = global_stats["std"]

    target = np.clip(rng.normal(mu, sd), 1, 127)
    adc = np.zeros((N_CH, SAMPLES), dtype=np.int32)

    for ch in range(N_CH):
        amp = int(round(target[ch]))
        vals = np.zeros(SAMPLES, dtype=np.int32)

        s = 0
        sign = rng.choice([-1, 1])
        while s < SAMPLES:
            run = int(rng.integers(2, 12))
            if rng.random() < 0.4:
                sign *= -1
            e = min(SAMPLES, s + run)
            jitter = rng.integers(-4, 5, size=e - s)
            vals[s:e] = ADC_MID + sign * np.clip(amp + jitter, 0, 127)
            s = e

        adc[ch] = np.clip(vals, 0, 255)

    return adc


# ============================================================
# Export
# ============================================================
def export_weights(prefix, q):
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
        f.write("// Brochier SBP+MLP weights\n")
        f.write("// Layout: w1[hidden][8], b1[hidden], w2[4][hidden], b2[4]\n")
        f.write(f"// hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"// output_bias_scale  = {q['output_bias_scale']}\n\n")
        for v in all_w:
            vv = int(v)
            if vv < 0:
                vv += 256
            f.write(f"{vv & 0xFF:02X}\n")

    all_w.astype(np.int8).tofile(bin_path)

    with open(readable_path, "w") as f:
        f.write(f"hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"output_bias_scale  = {q['output_bias_scale']}\n\n")
        f.write("w1_q:\n")
        for row in q["w1_q"]:
            f.write(" ".join(f"{int(v):+4d}" for v in row) + "\n")
        f.write("\nb1_q:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["b1_q"]) + "\n")
        f.write("\nw2_q:\n")
        for row in q["w2_q"]:
            f.write(" ".join(f"{int(v):+4d}" for v in row) + "\n")
        f.write("\nb2_q:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["b2_q"]) + "\n")

    return hex_path, bin_path, readable_path


def export_real_vectors(path, X_int, y_all, test_idx, q, n_vec):
    n_vec = min(n_vec, len(test_idx))

    selected = []
    for c in range(N_CLASSES):
        cidx = [idx for idx in test_idx if y_all[idx] == c]
        selected.extend(cidx[: max(1, n_vec // N_CLASSES)])
    remaining = [idx for idx in test_idx if idx not in set(selected)]
    selected.extend(remaining[: n_vec - len(selected)])
    selected = selected[:n_vec]

    with open(path, "w") as f:
        f.write(f"// REAL BROCHIER RTL TEST VECTORS ({len(selected)})\n")
        f.write(f"// ADC(8ch,{SAMPLES}) -> SBP -> MLP\n\n")

        for vi, idx in enumerate(selected):
            adc = X_int[idx]
            pred, scores, sbp, hidden = golden_int_model(adc, q)
            true = int(y_all[idx])

            f.write(f"// ===== Real Vector {vi:03d} =====\n")
            f.write(f"// True: {CLASS_NAMES[true]} ({true})\n")
            f.write(f"// Expected Pred: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// SBP: {' '.join(str(int(v)) for v in sbp)}\n")
            f.write(f"// HIDDEN: {' '.join(str(int(v)) for v in hidden)}\n")
            f.write(f"// SCORES: {' '.join(str(int(v)) for v in scores)}\n")
            for ch in range(N_CH):
                f.write(f"ch{ch}: " + " ".join(f"{int(adc[ch, s]):02X}" for s in range(SAMPLES)) + "\n")
            f.write("\n")


def export_synthetic_vectors(path, q, stats, global_stats, n_vec):
    rng = np.random.default_rng(12345)

    with open(path, "w") as f:
        f.write(f"// SYNTHETIC BROCHIER-LIKE RTL TEST VECTORS ({n_vec})\n")
        f.write("// Matched to training-set SBP statistics.\n\n")

        for vi in range(n_vec):
            cls = vi % N_CLASSES
            adc = make_synthetic_adc(cls, stats, global_stats, rng)
            pred, scores, sbp, hidden = golden_int_model(adc, q)

            f.write(f"// ===== Synthetic Vector {vi:03d} =====\n")
            f.write(f"// Intended synthetic class: {CLASS_NAMES[cls]} ({cls})\n")
            f.write(f"// Expected Pred: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// SBP: {' '.join(str(int(v)) for v in sbp)}\n")
            f.write(f"// HIDDEN: {' '.join(str(int(v)) for v in hidden)}\n")
            f.write(f"// SCORES: {' '.join(str(int(v)) for v in scores)}\n")
            for ch in range(N_CH):
                f.write(f"ch{ch}: " + " ".join(f"{int(adc[ch, s]):02X}" for s in range(SAMPLES)) + "\n")
            f.write("\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-prefix", type=str, default=None,
                        help="Optional: l101210-001 or i140703-001. Default uses all discovered sessions.")
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--out-prefix", type=str, default="brochier_both")
    parser.add_argument("--real-vectors", type=int, default=60)
    parser.add_argument("--synthetic-vectors", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print("BROCHIER SBP+MLP — ADC/SBP QUANTIZATION ERROR ANALYSIS")
    print("=" * 70)

    sessions = discover_sessions()
    if not sessions:
        print("ERROR: No Brochier NS5/NS6 + NEV pairs found.")
        sys.exit(1)

    sessions = filter_sessions(sessions, args.session_prefix)

    print("Sessions to load:")
    for ns, nev in sessions:
        print(f"  {Path(ns).name} + {Path(nev).name}")

    X_parts = []
    Xf_parts = []
    y_parts = []
    infos = []

    for ns, nev in sessions:
        X, Xf, y, info = load_session(ns, nev)
        if X is None:
            continue
        X_parts.append(X)
        Xf_parts.append(Xf)
        y_parts.append(y)
        infos.append(info)

    if not X_parts:
        print("ERROR: no sessions loaded.")
        sys.exit(1)

    X_int = np.concatenate(X_parts, axis=0)
    X_float_adc = np.concatenate(Xf_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    print("\n" + "=" * 70)
    print("POOLED DATASET")
    print("=" * 70)
    print(f"  sessions: {len(infos)}")
    print(f"  windows:  {len(y_all)}")
    print(f"  shape:    {X_int.shape}")
    print(f"  classes:  {np.bincount(y_all, minlength=N_CLASSES)}")

    # Pooled ADC/SBP quantization error
    sbp_float = np.array([compute_sbp_float_adc(x) for x in X_float_adc])
    sbp_int = np.array([compute_sbp_int(x) for x in X_int])
    pooled_qerr = summarize_quant_error(sbp_float, sbp_int, "pooled")

    # Train on integer RTL SBP features
    model, train_idx, test_idx, sbp_mean, sbp_std, float_preds, float_acc = train_model(
        sbp_int.astype(np.float64), y_all, hidden=args.hidden
    )

    q = quantize_model(model, sbp_mean, sbp_std)
    q_preds, q_acc, pred_mismatch = evaluate_integer_path(X_int, y_all, test_idx, q, float_preds)

    stats, global_stats = estimate_sbp_stats(X_int, y_all, train_idx)

    print("\nExporting RTL files...")
    weights_hex, weights_bin, weights_readable = export_weights(args.out_prefix, q)

    real_path = f"{args.out_prefix}_real_test_vectors.txt"
    synth_path = f"{args.out_prefix}_synthetic_test_vectors.txt"
    export_real_vectors(real_path, X_int, y_all, test_idx, q, args.real_vectors)
    export_synthetic_vectors(synth_path, q, stats, global_stats, args.synthetic_vectors)

    torch.save(model.state_dict(), f"{args.out_prefix}_model.pth")
    np.savez(
        f"{args.out_prefix}_meta.npz",
        sessions=np.array([info["session_name"] for info in infos]),
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        sbp_mean=sbp_mean,
        sbp_std=sbp_std,
        float_acc=float_acc,
        int_acc=q_acc,
        pred_mismatch=pred_mismatch,
        adc_sbp_mae=pooled_qerr["mae"],
        adc_sbp_rmse=pooled_qerr["rmse"],
        adc_sbp_p95=pooled_qerr["p95_abs"],
    )

    # Plot summary
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [float_acc, q_acc]
    labels = ["Float MLP", "Integer MLP"]
    bars = ax.bar(labels, vals, width=0.45)
    ax.axhline(25, linestyle="--", linewidth=1, label="Chance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Brochier SBP+MLP pooled sessions")
    ax.legend()
    plt.tight_layout()
    fig_path = f"{args.out_prefix}_results.png"
    plt.savefig(fig_path, dpi=150)

    print("\nSaved:")
    print(f"  {weights_hex}")
    print(f"  {weights_bin}")
    print(f"  {weights_readable}")
    print(f"  {real_path}")
    print(f"  {synth_path}")
    print(f"  {args.out_prefix}_model.pth")
    print(f"  {args.out_prefix}_meta.npz")
    print(f"  {fig_path}")

    n_bytes = q["w1_q"].size + q["b1_q"].size + q["w2_q"].size + q["b2_q"].size

    print("\nFINAL SUMMARY")
    print("=" * 70)
    print(f"  Sessions used:               {len(infos)}")
    print(f"  Total trials/windows:        {len(y_all)}")
    print(f"  Model:                       8 -> {args.hidden} -> 4")
    print(f"  Param bytes, int8 style:      {n_bytes}")
    print(f"  ADC/SBP quant MAE:           {pooled_qerr['mae']:.4f} SBP codes")
    print(f"  ADC/SBP quant RMSE:          {pooled_qerr['rmse']:.4f} SBP codes")
    print(f"  ADC/SBP quant 95% abs err:   {pooled_qerr['p95_abs']:.4f} SBP codes")
    print(f"  Float MLP accuracy:          {float_acc:.1f}%")
    print(f"  Integer MLP accuracy:        {q_acc:.1f}%")
    print(f"  Float-vs-int pred mismatch:  {pred_mismatch}/{len(test_idx)}")
    print()
    print("For RTL, use:")
    print(f"  1. {weights_hex}")
    print(f"  2. {weights_readable}")
    print(f"  3. {real_path}")
    print(f"  4. {synth_path}")


if __name__ == "__main__":
    main()
