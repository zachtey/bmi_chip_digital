#!/usr/bin/env python3
"""
SBP + MLP PIPELINE — SINGLE BROCHIER SESSION + SYNTHETIC RTL VECTORS
====================================================================

This is a drop-in variant of your Brochier multi-session script.

What changed:
  1. Uses ONLY ONE Brochier dataset/session.
  2. Trains/tests SBP+MLP on that one session.
  3. Exports real Brochier ADC test vectors.
  4. After that, generates synthetic ADC test vectors with SBP/noise
     statistics matched to the selected Brochier session.

Default session selection:
  Prefer l101210-001.ns5 if present, otherwise first discovered NS5/NS6 pair.

Run:
  python3 brochier_single_session_plus_synth.py

Optional:
  python3 brochier_single_session_plus_synth.py --session-prefix l101210-001
  python3 brochier_single_session_plus_synth.py --session-prefix i140703-001
  python3 brochier_single_session_plus_synth.py --hidden 16
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

# ══════════════════════════════════════════════════════════════
# HARDWARE PARAMETERS — must match your Verilog exactly
# ══════════════════════════════════════════════════════════════
SOURCE_FS   = 30000
TARGET_FS   = 5000
N_CH        = 8
SAMPLES     = 250       # 50ms at 5kHz
N_CLASSES   = 4
BP_LOW      = 300
BP_HIGH     = 3000
OFFSET_MS   = 100       # ms after GO-ON
ADC_MID     = 128

CLASS_NAMES = ['PG-LF', 'PG-HF', 'SG-LF', 'SG-HF']

# Known GO-ON event codes per monkey
MONKEY_L_GO = {'65381': 0, '65382': 1, '65385': 2, '65386': 3}

try:
    import neo
    import quantities as pq
except ImportError:
    print("ERROR: pip install neo quantities")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# EVENT / SESSION HELPERS
# ══════════════════════════════════════════════════════════════
def detect_goon_codes(events):
    """
    Find 4 event codes that likely represent the four GO-ON trial types.
    Returns dict {code_str: class_label}.
    """
    counts = {k: len(v) for k, v in events.items()}
    keys = list(counts.keys())
    n = len(keys)
    best = None
    best_balance = 1e9

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    group = [keys[i], keys[j], keys[k], keys[l]]
                    total = sum(counts[g] for g in group)
                    if total < 40 or total > 1000:
                        continue
                    vals = [counts[g] for g in group]
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


def discover_sessions():
    ns_files = (
        glob.glob("*.ns5") + glob.glob("**/*.ns5", recursive=True) +
        glob.glob("*.ns6") + glob.glob("**/*.ns6", recursive=True)
    )
    ns_files = sorted(list({os.path.abspath(f) for f in ns_files}))

    sessions = []
    for ns in ns_files:
        base = ns.rsplit('.', 1)[0]
        for suffix in ['-02.nev', '-03.nev', '.nev']:
            nev = base + suffix
            if os.path.exists(nev):
                sessions.append((os.path.abspath(ns), os.path.abspath(nev)))
                break

    # dedupe
    sessions = sorted(list({(a, b) for a, b in sessions}))
    return sessions


def choose_one_session(sessions, session_prefix=None):
    if not sessions:
        return None

    if session_prefix:
        for ns, nev in sessions:
            if Path(ns).stem.startswith(session_prefix):
                return ns, nev
        raise FileNotFoundError(f"No session matching --session-prefix {session_prefix}")

    # Prefer your previous Monkey L session if present
    for ns, nev in sessions:
        if Path(ns).name.startswith("l101210-001"):
            return ns, nev

    return sessions[0]


# ══════════════════════════════════════════════════════════════
# LOAD SINGLE SESSION
# ══════════════════════════════════════════════════════════════
def load_session(ns_path, nev_path):
    """
    Returns:
      X_8: (N_trials, 8, 250) int32 ADC samples
      y:   (N_trials,) int labels 0-3
      info: dict with selected channels, rails, session name, etc.
    """
    session_name = os.path.basename(ns_path)
    print(f"\n{'─'*70}")
    print(f"Loading ONE selected session: {session_name}")
    print(f"{'─'*70}")

    nsx_id = 6 if ns_path.endswith('.ns6') else 5
    try:
        reader  = neo.io.BlackrockIO(filename=ns_path, nsx_to_load=nsx_id)
        block   = reader.read_block(lazy=True, load_waveforms=False)
        raw_sig = next((a for a in block.segments[0].analogsignals
                        if float(a.sampling_rate.rescale('Hz').magnitude) >= 25000), None)
    except Exception as e:
        print(f"  ERROR reading NS file: {e}")
        return None, None, None

    if raw_sig is None:
        print("  ERROR: No 30kHz signal found")
        return None, None, None

    T_raw  = raw_sig.shape[0]
    n_raw  = raw_sig.shape[1]
    fs_raw = float(raw_sig.sampling_rate.rescale('Hz').magnitude)
    DS     = int(round(fs_raw / TARGET_FS))

    print(f"  {T_raw} samples × {n_raw} ch @ {fs_raw:.0f} Hz ({T_raw/fs_raw:.0f}s)")
    print("  Loading raw array for this one session...")

    raw_full = raw_sig.load().magnitude.astype(np.float32)

    try:
        nev_reader = neo.io.BlackrockIO(filename=nev_path, nsx_to_load=None)
        nev_seg    = nev_reader.read_block(lazy=False, load_waveforms=False).segments[0]
    except Exception as e:
        print(f"  ERROR reading NEV file: {e}")
        return None, None, None

    events = {}
    for evt in nev_seg.events:
        times  = evt.times.rescale('s').magnitude
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
            print("  ERROR: Could not detect GO-ON codes")
            print(f"  Counts: { {k: len(v) for k, v in events.items()} }")
            return None, None, None
        print(f"  Auto-detected GO-ON codes: {goon_map}")

    trials = sorted(
        [(t, label) for code, label in goon_map.items()
         for t in events.get(code, [])],
        key=lambda x: x[0]
    )

    counts = np.bincount([t[1] for t in trials], minlength=N_CLASSES)
    print(f"  Trials: {len(trials)} — " +
          " | ".join(f"{CLASS_NAMES[c]}:{counts[c]}" for c in range(N_CLASSES)))

    if len(trials) < 40:
        print("  ERROR: Too few trials")
        return None, None, None

    # Channel selection from raw downsampled SBP-like statistic
    OFFSET_RAW = int(OFFSET_MS / 1000.0 * fs_raw)
    WIN_RAW    = SAMPLES * DS

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
        print("  ERROR: No valid windows fit in recording")
        return None, None, None

    raw_sbp_all = np.array(raw_sbp_all)
    y_raw = np.array([t[1] for t in valid_trials])

    cls_means = np.array([raw_sbp_all[y_raw == c].mean(0)
                          for c in range(N_CLASSES) if np.sum(y_raw == c) > 0])
    grand   = raw_sbp_all.mean(0)
    between = np.mean((cls_means - grand)**2, axis=0)
    within  = np.mean([raw_sbp_all[y_raw == c].var(0)
                       for c in range(N_CLASSES) if np.sum(y_raw == c) > 0], axis=0)
    fisher = between / (within + 1e-10)
    top8 = np.argsort(fisher)[-N_CH:]
    top8_sorted = np.sort(top8)
    print(f"  Selected channels: {list(map(int, top8_sorted))}")

    # Filter only selected channels
    print(f"  Filtering selected {N_CH} channels...")
    nyq = fs_raw / 2.0
    b, a = sp.butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype='band')

    filtered_8 = np.zeros((T_raw, N_CH), dtype=np.float32)
    for ci, ch in enumerate(top8_sorted):
        filtered_8[:, ci] = sp.filtfilt(b, a, raw_full[:, ch].astype(np.float64))

    del raw_full

    # ADC rails
    v_min = np.percentile(filtered_8, 1)
    v_max = np.percentile(filtered_8, 99)
    print(f"  ADC rails: [{v_min:.2f}, {v_max:.2f}]")

    filt_ds = filtered_8[::DS]
    del filtered_8
    T_ds = len(filt_ds)

    adc_ds = np.round(
        np.clip((filt_ds - v_min) / (v_max - v_min), 0, 1) * 255
    ).astype(np.int32)

    X_8, y_8 = [], []
    OFFSET_DS = int(OFFSET_MS / 1000.0 * TARGET_FS)
    for go_t, label in valid_trials:
        s = int(go_t * TARGET_FS) + OFFSET_DS
        e = s + SAMPLES
        if s < 0 or e > T_ds:
            continue
        X_8.append(adc_ds[s:e, :].T)  # (8,250)
        y_8.append(label)

    X_8 = np.array(X_8, dtype=np.int32)
    y_8 = np.array(y_8, dtype=np.int64)

    print(f"  Final: {len(y_8)} windows × {N_CH} ch × {SAMPLES} samples")

    info = {
        "session_name": session_name,
        "nev_name": os.path.basename(nev_path),
        "fs_raw": fs_raw,
        "ds": DS,
        "selected_channels": top8_sorted,
        "v_min": v_min,
        "v_max": v_max,
        "class_counts": np.bincount(y_8, minlength=N_CLASSES),
    }
    return X_8, y_8, info


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
class SbpMLP(nn.Module):
    def __init__(self, hidden=8, dropout=0.3):
        super().__init__()
        self.h1 = nn.Linear(N_CH, hidden)
        self.drop = nn.Dropout(dropout)
        self.o = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        return self.o(self.drop(F.relu(self.h1(x))))


def train_float_model(sbp_features, y_all, hidden=8):
    perm = np.random.permutation(len(y_all))
    n_train = int(0.8 * len(y_all))
    n_test = len(y_all) - n_train

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    sbp_mean = sbp_features[train_idx].mean(0)
    sbp_std  = sbp_features[train_idx].std(0) + 1e-8
    sbp_norm = (sbp_features - sbp_mean) / sbp_std

    X_tr = torch.tensor(sbp_norm[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_te = torch.tensor(sbp_norm[test_idx], dtype=torch.float32)
    y_te = torch.tensor(y_all[test_idx], dtype=torch.long)

    print(f"\nTrain: {n_train} trials | Test: {n_test} trials")
    print(f"Train classes: {np.bincount(y_all[train_idx], minlength=N_CLASSES)}")
    print(f"Test  classes: {np.bincount(y_all[test_idx], minlength=N_CLASSES)}")

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    model = SbpMLP(hidden=hidden, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1
    best_state = None

    print(f"\nTraining SBP+MLP (8→{hidden}→4)...")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_param}")

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
                pred = model(X_te).argmax(1)
                acc = 100.0 * (pred == y_te).sum().item() / len(y_te)
            print(f"  Epoch {ep+1:3d} | Test: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(1)
        acc_test = 100.0 * (preds == y_te).sum().item() / len(y_te)

    print(f"\n>>> Best float accuracy: {acc_test:.1f}% (chance=25%)")
    print("Per-class float accuracy:")
    for c in range(N_CLASSES):
        mask = (y_te == c)
        if mask.sum() > 0:
            ca = 100.0 * (preds[mask] == c).sum().item() / mask.sum().item()
            print(f"  {CLASS_NAMES[c]:>6s}: {ca:.1f}% ({mask.sum().item()} test samples)")

    return model, train_idx, test_idx, sbp_mean, sbp_std, acc_test


# ══════════════════════════════════════════════════════════════
# QUANTIZATION + GOLDEN MODEL
# ══════════════════════════════════════════════════════════════
def quantize(arr):
    arr = np.array(arr, dtype=np.float64)
    mx = np.abs(arr).max()
    if mx < 1e-10:
        return np.zeros_like(arr, dtype=np.int16), 1.0
    sc = 127.0 / mx
    return np.round(arr * sc).clip(-128, 127).astype(np.int16), float(sc)


def quantize_model(model, sbp_mean, sbp_std):
    hw1_float = model.h1.weight.detach().numpy()
    hb1_float = model.h1.bias.detach().numpy()
    ow_float  = model.o.weight.detach().numpy()
    ob_float  = model.o.bias.detach().numpy()

    hw1_eff = hw1_float / sbp_std[np.newaxis, :]
    hb1_eff = hb1_float - np.sum(hw1_float * sbp_mean[np.newaxis, :] / sbp_std[np.newaxis, :], axis=1)

    # Verify fold
    dummy = sbp_mean + 0.1 * sbp_std
    assert np.allclose(
        hw1_float @ ((dummy - sbp_mean) / sbp_std) + hb1_float,
        hw1_eff @ dummy + hb1_eff,
        atol=1e-6
    ), "Fold failed"

    hw1_q, sc_hw1 = quantize(hw1_eff)
    hb1_q, sc_hb1 = quantize(hb1_eff)
    ow_q, sc_ow = quantize(ow_float)
    ob_q, sc_ob = quantize(ob_float)

    hidden1_bias_scale = int(round(sc_hw1 / sc_hb1))
    output_bias_scale = int(round(sc_hw1 * sc_ow / sc_ob))

    print("\nQuantized model:")
    print(f"  hw1 scale: {sc_hw1:.1f} | hb1 scale: {sc_hb1:.1f}")
    print(f"  ow  scale: {sc_ow:.1f} | ob  scale: {sc_ob:.1f}")
    print(f"  hidden1_bias_scale = {hidden1_bias_scale}")
    print(f"  output_bias_scale  = {output_bias_scale}")

    return {
        "hw1_q": hw1_q,
        "hb1_q": hb1_q,
        "ow_q": ow_q,
        "ob_q": ob_q,
        "sc_hw1": sc_hw1,
        "sc_hb1": sc_hb1,
        "sc_ow": sc_ow,
        "sc_ob": sc_ob,
        "hidden1_bias_scale": hidden1_bias_scale,
        "output_bias_scale": output_bias_scale,
    }


def compute_sbp(adc_window):
    sbp = np.zeros(N_CH, dtype=np.int32)
    for ch in range(N_CH):
        total = np.int32(0)
        for s in range(SAMPLES):
            total += abs(int(adc_window[ch, s]) - ADC_MID)
        sbp[ch] = total // SAMPLES
    return sbp


def golden_model(adc_window, q):
    sbp = compute_sbp(adc_window)
    hidden = np.zeros(q["hw1_q"].shape[0], dtype=np.int64)

    for j in range(q["hw1_q"].shape[0]):
        acc = np.int64(0)
        for i in range(N_CH):
            acc += np.int64(sbp[i]) * np.int64(q["hw1_q"][j, i])
        acc += np.int64(q["hb1_q"][j]) * np.int64(q["hidden1_bias_scale"])
        hidden[j] = max(0, acc)

    scores = np.zeros(N_CLASSES, dtype=np.int64)
    for k in range(N_CLASSES):
        acc = np.int64(0)
        for j in range(q["ow_q"].shape[1]):
            acc += np.int64(hidden[j]) * np.int64(q["ow_q"][k, j])
        acc += np.int64(q["ob_q"][k]) * np.int64(q["output_bias_scale"])
        scores[k] = acc

    return int(np.argmax(scores)), scores, sbp, hidden


def evaluate_quantized(X_all, y_all, test_idx, q):
    correct = 0
    for i in test_idx:
        pred, _, _, _ = golden_model(X_all[i], q)
        correct += int(pred == int(y_all[i]))
    return 100.0 * correct / len(test_idx)


# ══════════════════════════════════════════════════════════════
# SYNTHETIC BROCHIER-LIKE VECTOR GENERATION
# ══════════════════════════════════════════════════════════════
def estimate_adc_stats_by_class(X_all, y_all):
    """
    Estimate per-class/per-channel SBP mean/std from the real ADC windows.
    Also estimate simple sign-flip probability and noise-like jitter scale.

    Synthetic vectors are NOT meant to be neuroscience-realistic.
    They are meant to look ADC-like and reproduce Brochier SBP ranges/classes.
    """
    sbp_all = np.array([compute_sbp(x) for x in X_all], dtype=np.float64)

    stats = {}
    for c in range(N_CLASSES):
        x_c = X_all[y_all == c]
        sbp_c = sbp_all[y_all == c]
        if len(x_c) == 0:
            continue

        stats[c] = {
            "sbp_mean": sbp_c.mean(axis=0),
            "sbp_std": sbp_c.std(axis=0) + 1.0,
        }

    global_stats = {
        "sbp_min": sbp_all.min(axis=0),
        "sbp_max": sbp_all.max(axis=0),
        "sbp_mean": sbp_all.mean(axis=0),
        "sbp_std": sbp_all.std(axis=0) + 1.0,
    }
    return stats, global_stats


def make_synthetic_adc_window_for_class(cls, stats, global_stats, rng):
    """
    Generate one synthetic ADC window shaped like:
      128 ± amplitude + small jitter

    The target amplitude per channel is sampled from that class's
    empirical Brochier SBP distribution.
    """
    if cls in stats:
        mu = stats[cls]["sbp_mean"]
        sd = stats[cls]["sbp_std"]
    else:
        mu = global_stats["sbp_mean"]
        sd = global_stats["sbp_std"]

    target_sbp = rng.normal(mu, sd)
    target_sbp = np.clip(target_sbp, 1, 120)

    adc = np.zeros((N_CH, SAMPLES), dtype=np.int32)

    for ch in range(N_CH):
        amp = int(round(target_sbp[ch]))
        amp = max(1, min(amp, 127))

        # Mix random sign flips with mild low-frequency segments.
        vals = np.zeros(SAMPLES, dtype=np.int32)
        s = 0
        current_sign = rng.choice([-1, 1])
        while s < SAMPLES:
            run_len = int(rng.integers(2, 12))
            current_sign *= rng.choice([-1, 1], p=[0.25, 0.75])
            e = min(SAMPLES, s + run_len)
            jitter = rng.integers(-4, 5, size=e-s)
            vals[s:e] = ADC_MID + current_sign * np.clip(amp + jitter, 0, 127)
            s = e

        adc[ch] = np.clip(vals, 0, 255)

    return adc


def export_synthetic_vectors(out_path, q, stats, global_stats, n_vec=80):
    rng = np.random.default_rng(12345)

    with open(out_path, "w") as f:
        f.write(f"// SYNTHETIC BROCHIER-LIKE RTL TEST VECTORS ({n_vec} vectors)\n")
        f.write(f"// Pipeline: ADC(8ch,{SAMPLES}samp) -> SBP=sum(abs(x-128))/{SAMPLES} -> MLP\n")
        f.write("// These are synthetic vectors matched to the selected Brochier session's SBP statistics.\n")
        f.write("// They are for RTL stress/functional testing, not real neural accuracy.\n\n")

        for vec in range(n_vec):
            cls = vec % N_CLASSES
            adc = make_synthetic_adc_window_for_class(cls, stats, global_stats, rng)
            pred, scores, sbp, hidden = golden_model(adc, q)

            f.write(f"// ===== Synthetic Vector {vec:03d} =====\n")
            f.write(f"// Synthetic intended class: {CLASS_NAMES[cls]} ({cls})\n")
            f.write(f"// Expected predicted class: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// SBP integers: {list(map(int, sbp))}\n")
            f.write(f"// Hidden accumulators: {list(map(int, hidden))}\n")
            f.write(f"// Scores: {list(map(int, scores))}\n")
            for ch in range(N_CH):
                vals = " ".join(f"{int(adc[ch, s]):02X}" for s in range(SAMPLES))
                f.write(f"ch{ch}: {vals}\n")
            f.write("\n")


# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════
def export_weights(q, hidden, out_prefix="single"):
    all_w = np.concatenate([
        q["hw1_q"].flatten(),
        q["hb1_q"].flatten(),
        q["ow_q"].flatten(),
        q["ob_q"].flatten(),
    ])

    hex_path = f"{out_prefix}_weights.hex"
    bin_path = f"{out_prefix}_weights.bin"
    readable_path = f"{out_prefix}_weights_readable.txt"

    with open(hex_path, "w") as f:
        f.write("// SBP+MLP weights — single Brochier session\n")
        f.write(f"// Architecture: 8 -> {hidden} -> 4\n")
        f.write(f"// Layout: hw1[{hidden}][8], hb1[{hidden}], ow[4][{hidden}], ob[4]\n")
        f.write(f"// hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"// output_bias_scale  = {q['output_bias_scale']}\n\n")
        for v in all_w:
            vv = int(v)
            if vv < 0:
                vv += 256
            f.write(f"{vv & 0xFF:02X}\n")

    all_w.astype(np.int8).tofile(bin_path)

    with open(readable_path, "w") as f:
        f.write(f"Architecture: 8 -> {hidden} -> 4\n")
        f.write(f"hidden1_bias_scale = {q['hidden1_bias_scale']}\n")
        f.write(f"output_bias_scale  = {q['output_bias_scale']}\n\n")

        f.write("hw1_q[hidden][input]:\n")
        for j in range(hidden):
            f.write(f"hw1[{j:02d}] = " + " ".join(f"{int(v):+4d}" for v in q["hw1_q"][j]) + "\n")

        f.write("\nhb1_q[hidden]:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["hb1_q"]) + "\n")

        f.write("\now_q[class][hidden]:\n")
        for k in range(N_CLASSES):
            f.write(f"ow[{k}] {CLASS_NAMES[k]:>6s} = " +
                    " ".join(f"{int(v):+4d}" for v in q["ow_q"][k]) + "\n")

        f.write("\nob_q[class]:\n")
        f.write(" ".join(f"{int(v):+4d}" for v in q["ob_q"]) + "\n")

    return hex_path, bin_path, readable_path


def export_real_test_vectors(out_path, X_all, y_all, test_idx, q, n_vec=40):
    n_vec = min(n_vec, len(test_idx))

    per_class = {c: [] for c in range(N_CLASSES)}
    for pos, idx in enumerate(test_idx):
        per_class[int(y_all[idx])].append(idx)

    selected = []
    for c in range(N_CLASSES):
        selected.extend(per_class[c][:max(1, n_vec // N_CLASSES)])
    remaining = [idx for idx in test_idx if idx not in set(selected)]
    selected.extend(remaining[:n_vec - len(selected)])
    selected = selected[:n_vec]

    with open(out_path, "w") as f:
        f.write(f"// REAL BROCHIER SINGLE-SESSION RTL TEST VECTORS ({len(selected)} vectors)\n")
        f.write(f"// Pipeline: ADC(8ch,{SAMPLES}samp) -> SBP=sum(abs(x-128))/{SAMPLES} -> MLP\n\n")

        for vec_num, idx in enumerate(selected):
            adc_win = X_all[idx]
            pred, scores, sbp, hidden = golden_model(adc_win, q)
            true_lbl = int(y_all[idx])

            f.write(f"// ===== Real Vector {vec_num:03d} =====\n")
            f.write(f"// True: {CLASS_NAMES[true_lbl]} ({true_lbl})\n")
            f.write(f"// Expected predicted class: {CLASS_NAMES[pred]} ({pred})\n")
            f.write(f"// Expected CLASS[1:0] = {pred:02b}\n")
            f.write(f"// SBP integers: {list(map(int, sbp))}\n")
            f.write(f"// Hidden accumulators: {list(map(int, hidden))}\n")
            f.write(f"// Scores: {list(map(int, scores))}\n")
            for ch in range(N_CH):
                vals = " ".join(f"{int(adc_win[ch, s]):02X}" for s in range(SAMPLES))
                f.write(f"ch{ch}: {vals}\n")
            f.write("\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-prefix", type=str, default=None,
                        help="Example: l101210-001 or i140703-001. Default prefers l101210-001 if present.")
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--real-vectors", type=int, default=40)
    parser.add_argument("--synthetic-vectors", type=int, default=80)
    parser.add_argument("--out-prefix", type=str, default="single_brochier")
    args = parser.parse_args()

    print("=" * 70)
    print("SBP+MLP — SINGLE BROCHIER SESSION + SYNTHETIC VECTORS")
    print("=" * 70)

    sessions = discover_sessions()
    if not sessions:
        print("ERROR: No NS5/NS6 + NEV file pairs found.")
        sys.exit(1)

    print("Discovered sessions:")
    for ns, nev in sessions:
        print(f"  {Path(ns).name} + {Path(nev).name}")

    ns_path, nev_path = choose_one_session(sessions, args.session_prefix)
    print(f"\nUsing ONLY this session:")
    print(f"  {Path(ns_path).name} + {Path(nev_path).name}")

    X_all, y_all, info = load_session(ns_path, nev_path)
    if X_all is None:
        print("ERROR: failed to load selected session.")
        sys.exit(1)

    print("\nDataset summary:")
    print(f"  Session:      {info['session_name']}")
    print(f"  Windows:      {len(y_all)}")
    print(f"  Shape:        {X_all.shape}")
    print(f"  Class counts: {np.bincount(y_all, minlength=N_CLASSES)}")

    print("\nComputing SBP features from ADC windows...")
    sbp_features = np.mean(np.abs(X_all.astype(np.float64) - ADC_MID), axis=2)
    print(f"  SBP range: [{sbp_features.min():.2f}, {sbp_features.max():.2f}]")

    model, train_idx, test_idx, sbp_mean, sbp_std, acc_float = train_float_model(
        sbp_features, y_all, hidden=args.hidden
    )

    q = quantize_model(model, sbp_mean, sbp_std)

    print("\nRunning integer golden model on held-out real Brochier test set...")
    acc_q = evaluate_quantized(X_all, y_all, test_idx, q)
    print(f"  Float:     {acc_float:.1f}%")
    print(f"  Quantized: {acc_q:.1f}%")
    print(f"  Drop:      {acc_float - acc_q:.1f}%")

    print("\nEstimating Brochier-like synthetic vector statistics...")
    stats, global_stats = estimate_adc_stats_by_class(X_all[train_idx], y_all[train_idx])

    print("\nExporting RTL files...")
    hex_path, bin_path, readable_path = export_weights(q, args.hidden, args.out_prefix)

    real_vec_path = f"{args.out_prefix}_real_test_vectors.txt"
    export_real_test_vectors(real_vec_path, X_all, y_all, test_idx, q, n_vec=args.real_vectors)

    synth_vec_path = f"{args.out_prefix}_synthetic_test_vectors.txt"
    export_synthetic_vectors(synth_vec_path, q, stats, global_stats, n_vec=args.synthetic_vectors)

    torch.save(model.state_dict(), f"{args.out_prefix}_model.pth")
    np.savez(
        f"{args.out_prefix}_meta.npz",
        session=info["session_name"],
        selected_channels=info["selected_channels"],
        v_min=info["v_min"],
        v_max=info["v_max"],
        sbp_mean=sbp_mean,
        sbp_std=sbp_std,
        train_idx=train_idx,
        test_idx=test_idx,
        float_acc=acc_float,
        quant_acc=acc_q,
        hidden=args.hidden,
        hidden1_bias_scale=q["hidden1_bias_scale"],
        output_bias_scale=q["output_bias_scale"],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Float real", "Quant real"], [acc_float, acc_q], width=0.45)
    ax.axhline(25, linestyle="--", linewidth=1, label="Chance (25%)")
    for b, v in zip(bars, [acc_float, acc_q]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Single Brochier session: {info['session_name']}")
    ax.legend()
    plt.tight_layout()
    fig_path = f"{args.out_prefix}_results.png"
    plt.savefig(fig_path, dpi=150)

    print(f"  {hex_path}")
    print(f"  {bin_path}")
    print(f"  {readable_path}")
    print(f"  {real_vec_path}")
    print(f"  {synth_vec_path}")
    print(f"  {args.out_prefix}_model.pth")
    print(f"  {args.out_prefix}_meta.npz")
    print(f"  {fig_path}")

    n_weights = q["hw1_q"].size + q["hb1_q"].size + q["ow_q"].size + q["ob_q"].size

    print("\nFINAL SUMMARY")
    print("=" * 70)
    print(f"  Session used:       {info['session_name']}")
    print(f"  Total trials:       {len(y_all)}")
    print(f"  Model:              8 -> {args.hidden} -> 4")
    print(f"  Weight bytes:       {n_weights}")
    print(f"  Float real acc:     {acc_float:.1f}%")
    print(f"  Quant real acc:     {acc_q:.1f}%")
    print()
    print("Files for RTL:")
    print(f"  1. {hex_path}")
    print(f"  2. {readable_path}")
    print(f"  3. {real_vec_path}")
    print(f"  4. {synth_vec_path}")
    print()
    print("Use real vectors to check your real Brochier ADC path.")
    print("Use synthetic vectors as extra stress tests with Brochier-like SBP statistics.")


if __name__ == "__main__":
    main()
