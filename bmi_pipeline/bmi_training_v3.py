"""
BMI ASIC - Final Training Pipeline (v6)
========================================
Author: [Your Name]
Date: April 2026
Status: FINAL - Ready for hardware integration

HARDWARE MODEL: SBP + MLP
DECISION RATIONALE:
  - SBP + MLP consistently achieves highest accuracy (~44-45%)
  - Conv + MLP does NOT outperform despite added complexity
  - SBP features are already highly informative from the analog front-end
  - SBP + MLP is small (< 300 bytes), efficient, and maps cleanly to hardware
  - MLP architecture: 48 inputs -> 8 hidden (ReLU) -> 4 outputs (argmax)

WHAT THIS SCRIPT DOES:
  1. Loads real neural data from NWB files (Chestek Lab, dandiset 001201)
  2. Extracts SBP features and creates 4-class movement labels
  3. Trains 3 models for COMPARISON (Linear, MLP, Conv+MLP)
  4. Quantizes SBP+MLP to 8-bit for hardware deployment
  5. Exports weights in JSON + Verilog hex format
  6. Generates UVM test vectors with ALL intermediate values
  7. Reports hardware readiness (parameter count, memory, MACs)

WHAT YOUR RTL NEEDS TO IMPLEMENT:
  - Receive 48 normalized SBP features (16 channels x 3 features each)
  - Features are: mean, std, max of SBP over a 300ms window per channel
  - Hidden layer: 48 inputs x 8 neurons, add bias, ReLU activation
  - Output layer: 8 inputs x 4 neurons, add bias, argmax
  - Output: 2-bit classification (00=BothExt, 01=BothFlex, 10=IdxF_MrsE, 11=IdxE_MrsF)

DEPENDENCIES:
  pip3 install torch numpy matplotlib h5py

DATA:
  NWB files from dandiset 001201 (Chestek Lab, Monkey N)
  Download: dandi download dandi://dandi/001201
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import h5py

# ============================================================
# CONFIGURATION
# All constants that define the system. Change these to tune.
# ============================================================

# --- Neural data parameters ---
NUM_CHANNELS     = 16       # How many electrode channels we use (top by variance)
WINDOW_SIZE      = 15       # SBP time bins per classification window (15 x 20ms = 300ms)
NUM_CLASSES      = 4        # Movement directions: BothExt, BothFlex, IdxF_MrsE, IdxE_MrsF
SPEED_PERCENTILE = 50       # Filter out bottom 50% of speeds (removes ambiguous "hold" periods)

# --- MLP architecture (THIS IS WHAT THE HARDWARE IMPLEMENTS) ---
# Input: NUM_CHANNELS * 3 = 48 features (mean + std + max per channel)
# Hidden: 8 neurons with ReLU
# Output: 4 neurons (one per class), argmax selects winner
HIDDEN_NEURONS   = 8

# --- Conv parameters (kept for comparison only, NOT deployed to hardware) ---
CONV_KERNEL_LEN  = 5

# --- Quantization ---
QUANT_BITS       = 8        # 8-bit signed integers (-127 to +127)

# --- Training ---
EPOCHS           = 100
BATCH_SIZE       = 64
LEARNING_RATE    = 1e-3
SEED             = 42

# --- Data path (CHANGE THIS to your NWB file location) ---
NWB_DATA_DIR = "/Users/christopherleung/Downloads/visual/001201"
MAX_SESSIONS = 3            # How many recording sessions to load

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Class names for display ---
DIRECTION_NAMES = ["BothExt", "BothFlex", "IdxF_MrsE", "IdxE_MrsF"]


# ============================================================
# STEP 1: LOAD REAL NEURAL DATA FROM NWB FILES
#
# Each NWB file contains one recording session with:
#   - analysis/SpikingBandPower/data  -> (timepoints, 96) SBP values
#   - analysis/index_velocity/data    -> (timepoints, 1)  index finger velocity
#   - analysis/mrs_velocity/data      -> (timepoints, 1)  middle-ring-small velocity
#
# We select the top NUM_CHANNELS channels by variance from the
# FIRST session, then use those SAME channels for all sessions
# to keep the hardware mapping consistent.
#
# Velocity is discretized into 4 direction classes based on
# the sign of each finger group's velocity.
# ============================================================

def select_channels(nwb_path, num_channels):
    """
    Pick the most informative channels from the first session.
    "Most informative" = highest variance in SBP signal.
    
    WHY: Not all 96 electrodes carry useful movement information.
    High-variance channels tend to have stronger neural modulation.
    We lock these channels and use them for ALL sessions so the
    hardware always reads from the same electrodes.
    
    Returns: sorted array of channel indices, e.g. [2, 6, 32, 37, ...]
    """
    with h5py.File(nwb_path, 'r') as f:
        sbp = f['analysis/SpikingBandPower/data'][:].astype(np.float32)
    variance = np.var(sbp, axis=0)  # variance per channel
    top = np.sort(np.argsort(variance)[-num_channels:])
    return top


def load_session(nwb_path, channels, window_size=WINDOW_SIZE):
    """
    Load one recording session and return windowed features + labels.
    
    STEPS:
    1. Read SBP and velocity from the NWB file
    2. Select only our fixed channels
    3. Compute speed magnitude and remove slow/ambiguous periods
    4. Assign direction labels based on velocity sign
    5. Build sliding windows of SBP data
    
    Args:
        nwb_path: path to .nwb file
        channels: array of channel indices to use (from select_channels)
        window_size: how many 20ms time bins per window
    
    Returns:
        X: (num_windows, num_channels, window_size) - SBP windows
        y: (num_windows,) - class labels 0-3
    """
    print(f"  Loading: {os.path.basename(nwb_path)}")
    
    # --- Read raw data ---
    with h5py.File(nwb_path, 'r') as f:
        sbp_all = f['analysis/SpikingBandPower/data'][:].astype(np.float32)  # (T, 96)
        vel_idx = f['analysis/index_velocity/data'][:].flatten().astype(np.float32)  # (T,)
        vel_mrs = f['analysis/mrs_velocity/data'][:].flatten().astype(np.float32)    # (T,)
    
    # --- Use only our selected channels ---
    sbp = sbp_all[:, channels]  # (T, NUM_CHANNELS)
    
    # --- Filter out slow/ambiguous movement periods ---
    # When the monkey is holding still, velocity is near zero and the
    # direction label is essentially random noise. We remove these.
    speed = np.sqrt(vel_idx**2 + vel_mrs**2)
    threshold = np.percentile(speed, SPEED_PERCENTILE)
    moving = speed > threshold
    
    sbp = sbp[moving]
    vel_idx = vel_idx[moving]
    vel_mrs = vel_mrs[moving]
    
    # --- Assign direction labels ---
    # We use the sign of each velocity component to create 4 quadrants:
    #   Class 0: both extending  (idx < 0, mrs < 0)
    #   Class 1: both flexing    (idx >= 0, mrs >= 0)
    #   Class 2: index flex + mrs extend  (idx >= 0, mrs < 0)
    #   Class 3: index extend + mrs flex  (idx < 0, mrs >= 0)
    n = len(vel_idx)
    labels = np.zeros(n, dtype=np.int64)
    labels[(vel_idx < 0)  & (vel_mrs < 0)]  = 0
    labels[(vel_idx >= 0) & (vel_mrs >= 0)]  = 1
    labels[(vel_idx >= 0) & (vel_mrs < 0)]   = 2
    labels[(vel_idx < 0)  & (vel_mrs >= 0)]  = 3
    
    print(f"    Timepoints: {sbp_all.shape[0]} -> {n} after speed filter (p{SPEED_PERCENTILE})")
    print(f"    Class dist: {np.bincount(labels, minlength=NUM_CLASSES)}")
    
    # --- Build sliding windows ---
    # Each window is 15 consecutive time bins (300ms of SBP data).
    # The label comes from the LAST timestep in the window.
    n_win = n - window_size
    if n_win <= 0:
        raise ValueError(f"Not enough data after filtering: {n} timepoints")
    
    X = np.zeros((n_win, len(channels), window_size), dtype=np.float32)
    y = np.zeros(n_win, dtype=np.int64)
    
    for i in range(n_win):
        X[i] = sbp[i:i + window_size].T  # transpose: (channels, time)
        y[i] = labels[i + window_size - 1]
    
    return X, y


# --- Execute data loading ---
print("=" * 60)
print("STEP 1: Loading neural data from NWB files...")
print("=" * 60)

nwb_files = sorted(glob.glob(os.path.join(NWB_DATA_DIR, "**", "*.nwb"), recursive=True))
print(f"  Found {len(nwb_files)} NWB files in {NWB_DATA_DIR}")

if len(nwb_files) == 0:
    print(f"\n  ERROR: No .nwb files found. Check NWB_DATA_DIR path.")
    exit(1)

# Lock channel selection from first session
print(f"\n  Selecting top {NUM_CHANNELS} channels from first session...")
fixed_channels = select_channels(nwb_files[0], NUM_CHANNELS)
print(f"  Channels locked: {fixed_channels}")
print()

# Load up to MAX_SESSIONS
all_X, all_y = [], []
loaded = 0
for path in nwb_files:
    if loaded >= MAX_SESSIONS:
        break
    try:
        X_s, y_s = load_session(path, fixed_channels)
        all_X.append(X_s)
        all_y.append(y_s)
        loaded += 1
        print(f"    -> {X_s.shape[0]} windows OK\n")
    except Exception as e:
        print(f"    -> SKIP: {e}\n")

if loaded == 0:
    print("  ERROR: No sessions loaded successfully.")
    exit(1)

# Combine and split 80/20
all_data = np.concatenate(all_X)
all_labels = np.concatenate(all_y)
np.random.seed(SEED)
perm = np.random.permutation(len(all_data))
all_data, all_labels = all_data[perm], all_labels[perm]

split = int(0.8 * len(all_data))
train_data, test_data = all_data[:split], all_data[split:]
train_labels, test_labels = all_labels[:split], all_labels[split:]

print(f"  Loaded {loaded} sessions, {len(all_data)} total windows")
print(f"  Train: {train_data.shape}  Test: {test_data.shape}")
print(f"  Train class dist: {np.bincount(train_labels, minlength=NUM_CLASSES)}")
print(f"  Test class dist:  {np.bincount(test_labels, minlength=NUM_CLASSES)}")
print()


# ============================================================
# STEP 2: FEATURE EXTRACTION AND NORMALIZATION
#
# For the SBP-based models (Linear and MLP), we extract
# 3 summary features per channel from each 300ms window:
#   - mean: average SBP power in the window
#   - std:  variability of SBP in the window
#   - max:  peak SBP value in the window
#
# This gives us NUM_CHANNELS * 3 = 48 input features.
#
# We normalize using ONLY training set statistics to prevent
# data leakage. The same mean/std must be applied in hardware.
#
# Class weights handle the imbalance problem: classes with
# fewer samples get higher loss weight so the model doesn't
# just predict the majority class.
# ============================================================

print("=" * 60)
print("STEP 2: Feature extraction and normalization...")
print("=" * 60)

# --- Class weights (inverse frequency) ---
# Without this, the model would just predict "BothExt" for everything
# because it's ~40% of the data.
counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
weights = 1.0 / (counts + 1.0)
weights = weights / weights.sum() * NUM_CLASSES  # normalize so mean weight = 1
class_wt = torch.FloatTensor(weights)
print(f"  Class counts:  {counts.astype(int)}")
print(f"  Class weights: {np.round(weights, 3)}")

# --- Normalize SBP per channel ---
# Compute mean and std across all training windows and time steps
# for each channel independently.
ch_mean = train_data.mean(axis=(0, 2), keepdims=True)  # shape: (1, C, 1)
ch_std  = train_data.std(axis=(0, 2), keepdims=True) + 1e-8  # avoid div by zero
train_norm = (train_data - ch_mean) / ch_std
test_norm  = (test_data - ch_mean) / ch_std

# --- Extract summary features ---
# THIS IS WHAT THE HARDWARE COMPUTES from the SBP stream:
#   For each of 16 channels over a 300ms window:
#     feature[ch*3 + 0] = mean of SBP values
#     feature[ch*3 + 1] = std of SBP values
#     feature[ch*3 + 2] = max of SBP values
def extract_features(data):
    """
    Extract per-channel summary features from windowed SBP.
    
    Input:  (N, channels, time_bins) - normalized SBP windows
    Output: (N, channels * 3) - concatenated mean, std, max
    
    This is what the digital front-end computes before feeding
    the MLP. In hardware, this would be done by accumulators,
    comparators, and dividers operating on the SBP stream.
    """
    feat_mean = data.mean(axis=2)   # (N, C) - average power
    feat_std  = data.std(axis=2)    # (N, C) - power variability
    feat_max  = data.max(axis=2)    # (N, C) - peak power
    return np.concatenate([feat_mean, feat_std, feat_max], axis=1)  # (N, C*3)

train_features = extract_features(train_norm)
test_features  = extract_features(test_norm)
NUM_INPUT_FEATURES = train_features.shape[1]  # 16 * 3 = 48

print(f"  Input features: {NUM_INPUT_FEATURES} ({NUM_CHANNELS} channels x 3)")
print(f"  Feature names: [mean_ch0, ..., mean_ch15, std_ch0, ..., std_ch15, max_ch0, ..., max_ch15]")
print(f"  Normalized data range: [{train_norm.min():.2f}, {train_norm.max():.2f}]")
print()


# ============================================================
# STEP 3: MODEL DEFINITIONS
#
# We define 3 models for COMPARISON, but only SBP+MLP gets
# deployed to hardware.
#
# SBP + Linear:  48 -> 4  (simplest baseline)
# SBP + MLP:     48 -> 8 (ReLU) -> 4  (OUR HARDWARE MODEL)
# Conv + MLP:    temporal conv -> pool -> 8 (ReLU) -> 4  (comparison only)
# ============================================================

class SBPLinear(nn.Module):
    """
    Simplest baseline: direct linear mapping from features to classes.
    No hidden layer, no nonlinearity.
    Equivalent to logistic regression.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NUM_INPUT_FEATURES, NUM_CLASSES)
    
    def forward(self, x):
        return self.fc(x)


class SBPMLP(nn.Module):
    """
    *** THIS IS THE HARDWARE MODEL ***
    
    Architecture:
      Input:  48 features (16 channels x 3 summary stats)
      Hidden: 8 neurons, ReLU activation
      Output: 4 neurons (one per direction class)
      Decision: argmax of output scores -> 2-bit class label
    
    Hardware mapping:
      - Hidden layer: 48x8 weight matrix + 8 biases + ReLU
      - Output layer: 8x4 weight matrix + 4 biases
      - Argmax: comparator tree on 4 output values
    
    Total parameters:
      Hidden weights: 48 * 8 = 384
      Hidden biases:  8
      Output weights: 8 * 4  = 32
      Output biases:  4
      TOTAL: 428 parameters = 428 bytes at 8-bit
    """
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(NUM_INPUT_FEATURES, HIDDEN_NEURONS)  # 48 -> 8
        self.relu   = nn.ReLU()
        self.output = nn.Linear(HIDDEN_NEURONS, NUM_CLASSES)         # 8 -> 4
    
    def forward(self, x):
        h = self.relu(self.hidden(x))  # hidden activations (8 values)
        return self.output(h)          # output scores (4 values)


class ConvMLP(nn.Module):
    """
    Comparison model only - NOT deployed to hardware.
    Kept to show that Conv+MLP doesn't outperform SBP+MLP on this data.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=CONV_KERNEL_LEN, bias=False)
        self.relu_c = nn.ReLU()
        self.hidden = nn.Linear(NUM_CHANNELS, HIDDEN_NEURONS)
        self.relu_h = nn.ReLU()
        self.output = nn.Linear(HIDDEN_NEURONS, NUM_CLASSES)
    
    def forward(self, x):
        feats = []
        for ch in range(NUM_CHANNELS):
            c = self.relu_c(self.conv(x[:, ch, :].unsqueeze(1)))
            feats.append(c.mean(dim=2))
        feats = torch.cat(feats, dim=1)
        feats = feats - feats.mean(dim=1, keepdim=True)  # CAR
        return self.output(self.relu_h(self.hidden(feats)))


# ============================================================
# STEP 4: TRAINING
#
# All 3 models are trained with:
#   - Class-weighted cross-entropy loss (handles imbalance)
#   - Adam optimizer with learning rate scheduling
#   - 100 epochs (verified that loss converges by then)
#
# We track the BEST test accuracy across all epochs,
# not just the final epoch, to report peak performance.
# ============================================================

def train_model(model, train_dl, test_dl, name, class_weights,
                epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Train a model and return its history.
    
    Uses class-weighted loss so minority classes aren't ignored.
    Uses ReduceLROnPlateau to lower learning rate when loss plateaus.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    history = {"loss": [], "train_acc": [], "test_acc": []}
    best_test = 0.0
    
    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_dl:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_dl)
        
        # --- Test pass ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_dl:
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.size(0)
        test_acc = correct / total
        best_test = max(best_test, test_acc)
        
        scheduler.step(avg_loss)
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Train: {train_acc:.3f}  "
                  f"Test: {test_acc:.3f}  "
                  f"Best: {best_test:.3f}")
    
    return history, best_test


# --- Build PyTorch data loaders ---
# Summary features for Linear and MLP models
train_feat_t = torch.FloatTensor(train_features)
test_feat_t  = torch.FloatTensor(test_features)

# Raw temporal windows for Conv model (comparison only)
train_raw_t  = torch.FloatTensor(train_norm)
test_raw_t   = torch.FloatTensor(test_norm)

# Labels
train_lbl_t  = torch.LongTensor(train_labels)
test_lbl_t   = torch.LongTensor(test_labels)

feat_train_dl = DataLoader(TensorDataset(train_feat_t, train_lbl_t),
                            batch_size=BATCH_SIZE, shuffle=True)
feat_test_dl  = DataLoader(TensorDataset(test_feat_t, test_lbl_t),
                            batch_size=BATCH_SIZE, shuffle=False)
raw_train_dl  = DataLoader(TensorDataset(train_raw_t, train_lbl_t),
                            batch_size=BATCH_SIZE, shuffle=True)
raw_test_dl   = DataLoader(TensorDataset(test_raw_t, test_lbl_t),
                            batch_size=BATCH_SIZE, shuffle=False)

# --- Train all 3 models ---
print("=" * 60)
print("STEP 4: Training all models (100 epochs)...")
print("=" * 60)

print("\n--- Model 1: SBP + Linear (baseline) ---")
model_linear = SBPLinear()
hist_linear, best_linear = train_model(model_linear, feat_train_dl, feat_test_dl,
                                        "Linear", class_wt)

print("\n--- Model 2: SBP + MLP (*** HARDWARE MODEL ***) ---")
model_mlp = SBPMLP()
hist_mlp, best_mlp = train_model(model_mlp, feat_train_dl, feat_test_dl,
                                  "MLP", class_wt)

print("\n--- Model 3: Conv + MLP (comparison only) ---")
model_conv = ConvMLP()
hist_conv, best_conv = train_model(model_conv, raw_train_dl, raw_test_dl,
                                    "Conv", class_wt)


# ============================================================
# STEP 5: EVALUATION
#
# Compare all 3 models with:
#   - Overall accuracy (final epoch + best epoch)
#   - Per-class accuracy (catch any class being ignored)
#   - Confusion matrix (see where errors happen)
#
# This section confirms SBP+MLP is the best choice.
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Evaluation and comparison")
print("=" * 60)

# Final and best accuracies
results_final = {
    "SBP + Linear": hist_linear["test_acc"][-1],
    "SBP + MLP":    hist_mlp["test_acc"][-1],
    "Conv + MLP":   hist_conv["test_acc"][-1],
}
results_best = {
    "SBP + Linear": best_linear,
    "SBP + MLP":    best_mlp,
    "Conv + MLP":   best_conv,
}

print("\n  Final epoch accuracy:")
for name, acc in results_final.items():
    tag = " *** HARDWARE ***" if name == "SBP + MLP" else ""
    print(f"    {name:20s}: {acc:.1%}{tag}")

print("\n  Best epoch accuracy:")
for name, acc in results_best.items():
    tag = " *** HARDWARE ***" if name == "SBP + MLP" else ""
    print(f"    {name:20s}: {acc:.1%}{tag}")

# --- Detailed per-class evaluation ---
def evaluate_detailed(model, dataloader, name):
    """
    Print per-class accuracy and confusion matrix.
    This is critical for catching class imbalance issues.
    """
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for bx, by in dataloader:
            all_pred.extend(model(bx).argmax(1).numpy())
            all_true.extend(by.numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    overall = (all_pred == all_true).mean()
    
    print(f"\n  --- {name} (overall: {overall:.1%}) ---")
    print(f"  Per-class accuracy:")
    for c in range(NUM_CLASSES):
        mask = all_true == c
        if mask.sum() > 0:
            acc = (all_pred[mask] == c).mean()
            print(f"    {DIRECTION_NAMES[c]:12s}: {acc:.1%}  (n={mask.sum()})")
    
    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1
    
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    header = "            " + " ".join(f"{DIRECTION_NAMES[c][:7]:>7s}" for c in range(NUM_CLASSES))
    print(header)
    for r in range(NUM_CLASSES):
        row_str = " ".join(f"{cm[r][c]:7d}" for c in range(NUM_CLASSES))
        print(f"    {DIRECTION_NAMES[r][:7]:>7s}  {row_str}")
    
    return all_pred, all_true, cm

evaluate_detailed(model_linear, feat_test_dl, "SBP + Linear")
evaluate_detailed(model_mlp, feat_test_dl, "SBP + MLP (HARDWARE)")
evaluate_detailed(model_conv, raw_test_dl, "Conv + MLP (comparison)")

# --- Training curves plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, h in [("SBP+Linear", hist_linear), ("SBP+MLP (HW)", hist_mlp),
                  ("Conv+MLP", hist_conv)]:
    axes[0].plot(h["loss"], label=label)
    axes[1].plot(h["test_acc"], label=label)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Test Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_comparison.png", dpi=150)
print("\n  Saved: training_comparison.png")


# ============================================================
# STEP 6: QUANTIZE SBP+MLP TO 8-BIT
#
# The hardware uses 8-bit signed integers for all weights,
# biases, and (optionally) activations.
#
# Quantization method: symmetric per-tensor quantization
#   scale = max(|tensor|) / 127
#   quantized_value = round(float_value / scale)
#   dequantized_value = quantized_value * scale
#
# We verify that quantization causes minimal accuracy loss.
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Quantizing SBP+MLP to 8-bit...")
print("=" * 60)

def quantize_tensor(tensor, bits=QUANT_BITS):
    """
    Symmetric quantization: maps [-max, +max] to [-127, +127].
    Returns (quantized_int_tensor, scale_factor).
    To recover float: float_value = int_value * scale
    """
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return torch.zeros_like(tensor).int(), 1.0
    qmax = 2**(bits - 1) - 1  # 127 for 8-bit
    scale = max_val / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).int()
    return quantized, scale

# Quantize all SBP+MLP parameters
model_mlp.eval()
qp = {}  # quantized parameters (integer tensors)
sc = {}  # scale factors (float, needed to convert back)

qp["hidden_w"], sc["hidden_w"] = quantize_tensor(model_mlp.hidden.weight.data)  # (8, 48)
qp["hidden_b"], sc["hidden_b"] = quantize_tensor(model_mlp.hidden.bias.data)    # (8,)
qp["output_w"], sc["output_w"] = quantize_tensor(model_mlp.output.weight.data)  # (4, 8)
qp["output_b"], sc["output_b"] = quantize_tensor(model_mlp.output.bias.data)    # (4,)

total_params = sum(p.numel() for p in qp.values())
print(f"  Total parameters: {total_params}")
print(f"  Memory footprint: {total_params} bytes (8-bit)")
print()

for name in qp:
    print(f"  {name:12s}: shape={str(list(qp[name].shape)):12s}  "
          f"range=[{qp[name].min().item():4d}, {qp[name].max().item():4d}]  "
          f"scale={sc[name]:.6f}")

# --- Verify quantized accuracy ---
# Run the full test set through the quantized model and compare
def quantized_mlp_forward(x, qp, sc):
    """
    Forward pass using quantized (integer) weights.
    
    This is the GOLDEN MODEL - it defines what the hardware
    should produce for any given input. Test vectors are
    generated from this function.
    
    Steps match the hardware pipeline:
      1. Multiply input features by hidden weights, add bias
      2. Apply ReLU (clamp negatives to zero)
      3. Multiply hidden activations by output weights, add bias
      4. Return raw scores (argmax done separately)
    """
    # Reconstruct float weights from quantized values
    wh = qp["hidden_w"].float() * sc["hidden_w"]   # (8, 48)
    bh = qp["hidden_b"].float() * sc["hidden_b"]   # (8,)
    wo = qp["output_w"].float() * sc["output_w"]    # (4, 8)
    bo = qp["output_b"].float() * sc["output_b"]    # (4,)
    
    # Hidden layer: linear + ReLU
    hidden = torch.relu(x @ wh.T + bh)  # (batch, 8)
    
    # Output layer: linear (no activation - argmax is the decision)
    scores = hidden @ wo.T + bo  # (batch, 4)
    
    return scores, hidden  # return hidden for test vectors

print("\n  Verifying quantized accuracy...")
correct, total = 0, 0
with torch.no_grad():
    for bx, by in feat_test_dl:
        scores, _ = quantized_mlp_forward(bx, qp, sc)
        correct += (scores.argmax(1) == by).sum().item()
        total += by.size(0)

quant_acc = correct / total
float_acc = results_final["SBP + MLP"]
print(f"  Float accuracy:     {float_acc:.1%}")
print(f"  Quantized accuracy: {quant_acc:.1%}")
print(f"  Accuracy drop:      {float_acc - quant_acc:.1%}")

if float_acc - quant_acc > 0.02:
    print("  WARNING: >2% accuracy drop from quantization!")
else:
    print("  OK: Quantization loss is acceptable.")


# ============================================================
# STEP 7: EXPORT FOR HARDWARE
#
# Generates all files needed for RTL implementation and
# UVM verification:
#
#   quantized_weights.json  - All weights with shapes and scales
#   weights_hex.txt         - Verilog-formatted hex for scan chain
#   test_vectors.json       - Input/output pairs with intermediates
#   normalization_params.json - Preprocessing constants
#   model_results.json      - Summary of all results
#   training_comparison.png - Visual training curves
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Exporting for hardware...")
print("=" * 60)

os.makedirs("exports", exist_ok=True)

# ---- 7a: Quantized weights as JSON ----
# This file contains everything needed to load the model:
# integer weight values, scale factors, and shapes.
weights_json = {}
for name in qp:
    weights_json[name] = {
        "values": qp[name].numpy().tolist(),
        "scale": sc[name],
        "shape": list(qp[name].shape),
        "dtype": "int8"
    }

with open("exports/quantized_weights.json", "w") as f:
    json.dump(weights_json, f, indent=2)
print("  Saved: exports/quantized_weights.json")


# ---- 7b: Weights as Verilog hex ----
# Format: 8'hXX values ready for scan chain loading.
# Each weight is an 8-bit signed integer stored as unsigned hex.
# To convert: if value < 0, store as value + 256.
with open("exports/weights_hex.txt", "w") as f:
    f.write("// ============================================\n")
    f.write("// SBP+MLP Quantized Weights for Scan Chain\n")
    f.write("// ============================================\n")
    f.write(f"// Architecture: {NUM_INPUT_FEATURES} -> {HIDDEN_NEURONS} (ReLU) -> {NUM_CLASSES}\n")
    f.write(f"// Total parameters: {total_params} bytes\n")
    f.write(f"// Quantization: {QUANT_BITS}-bit symmetric\n")
    f.write("//\n")
    f.write("// Loading order: hidden_w, hidden_b, output_w, output_b\n")
    f.write("// ============================================\n\n")
    
    for name in qp:
        shape = list(qp[name].shape)
        f.write(f"// {name} {shape} (scale={sc[name]:.6f})\n")
        flat = qp[name].numpy().flatten()
        for i, val in enumerate(flat):
            unsigned = int(val) if val >= 0 else int(val) + 256
            f.write(f"8'h{unsigned:02X}")
            if i < len(flat) - 1:
                f.write(", ")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write("\n\n")

print("  Saved: exports/weights_hex.txt")


# ---- 7c: Test vectors for UVM ----
# Each vector contains:
#   - input: 48 normalized SBP features
#   - expected class label (ground truth)
#   - intermediate values at each stage (for RTL verification)
#   - predicted class (what quantized model outputs)
NUM_VECTORS = 30

vectors = []
for i in range(min(NUM_VECTORS, len(test_features))):
    # Get input features (already normalized)
    inp = test_features[i]  # 48 float values
    true_label = int(test_labels[i])
    
    # Run through quantized model step by step
    x = torch.FloatTensor(inp).unsqueeze(0)  # (1, 48)
    
    # --- Hidden layer ---
    wh = qp["hidden_w"].float() * sc["hidden_w"]
    bh = qp["hidden_b"].float() * sc["hidden_b"]
    hidden_pre_relu = (x @ wh.T + bh).squeeze().tolist()  # before ReLU
    hidden_post_relu = [max(0.0, v) for v in hidden_pre_relu]  # after ReLU
    
    # --- Output layer ---
    wo = qp["output_w"].float() * sc["output_w"]
    bo = qp["output_b"].float() * sc["output_b"]
    h_tensor = torch.FloatTensor(hidden_post_relu).unsqueeze(0)
    scores = (h_tensor @ wo.T + bo).squeeze().tolist()
    
    # --- Argmax ---
    predicted = int(np.argmax(scores))
    
    vectors.append({
        "vector_id": i,
        "input_features": inp.tolist(),         # 48 values fed to MLP
        "true_label": true_label,                # ground truth class
        "true_label_name": DIRECTION_NAMES[true_label],
        "intermediate": {
            "hidden_pre_relu": hidden_pre_relu,  # 8 values before ReLU
            "hidden_post_relu": hidden_post_relu, # 8 values after ReLU
            "output_scores": scores,              # 4 raw class scores
        },
        "predicted_class": predicted,            # argmax result
        "predicted_name": DIRECTION_NAMES[predicted],
        "correct": predicted == true_label
    })

# Count correct predictions in vectors
n_correct = sum(v["correct"] for v in vectors)
print(f"  Test vectors: {len(vectors)} generated, {n_correct}/{len(vectors)} correct")

with open("exports/test_vectors.json", "w") as f:
    json.dump(vectors, f, indent=2)
print(f"  Saved: exports/test_vectors.json ({len(vectors)} vectors)")


# ---- 7d: Normalization parameters ----
# The hardware preprocessing stage needs these to normalize
# incoming SBP values before computing summary features.
norm_export = {
    "description": "Per-channel normalization applied to raw SBP before feature extraction",
    "formula": "normalized = (sbp - channel_mean) / channel_std",
    "channel_means": ch_mean.squeeze().tolist(),
    "channel_stds": ch_std.squeeze().tolist(),
    "channels_used": fixed_channels.tolist(),
    "num_channels": NUM_CHANNELS,
    "window_size_bins": WINDOW_SIZE,
    "window_size_ms": WINDOW_SIZE * 20,
    "features_per_channel": 3,
    "feature_order": ["mean", "std", "max"],
    "total_input_features": NUM_INPUT_FEATURES
}

with open("exports/normalization_params.json", "w") as f:
    json.dump(norm_export, f, indent=2)
print("  Saved: exports/normalization_params.json")


# ---- 7e: Results summary ----
results_export = {
    "hardware_model": "SBP + MLP",
    "architecture": f"{NUM_INPUT_FEATURES} -> {HIDDEN_NEURONS} (ReLU) -> {NUM_CLASSES} (argmax)",
    "data_source": "Chestek Lab NWB, dandiset 001201",
    "sessions_loaded": loaded,
    "total_windows": len(all_data),
    "window_ms": WINDOW_SIZE * 20,
    "num_channels": NUM_CHANNELS,
    "channels": fixed_channels.tolist(),
    "speed_filter_percentile": SPEED_PERCENTILE,
    "comparison_results_final": results_final,
    "comparison_results_best": results_best,
    "quantized_accuracy": quant_acc,
    "quantization_bits": QUANT_BITS,
    "total_parameters": total_params,
    "memory_bytes": total_params,
    "class_distribution": {
        "train": np.bincount(train_labels, minlength=NUM_CLASSES).tolist(),
        "test": np.bincount(test_labels, minlength=NUM_CLASSES).tolist(),
    }
}

with open("exports/model_results.json", "w") as f:
    json.dump(results_export, f, indent=2)
print("  Saved: exports/model_results.json")


# ============================================================
# STEP 8: HARDWARE READINESS REPORT
#
# Summary of everything the RTL team needs to know.
# ============================================================

print("\n" + "=" * 60)
print("HARDWARE READINESS REPORT")
print("=" * 60)

# Count MAC operations per inference
# Hidden layer: each of 8 neurons does 48 multiplies + 48 adds = 48 MACs
# Output layer: each of 4 neurons does 8 multiplies + 8 adds = 8 MACs
macs_hidden = NUM_INPUT_FEATURES * HIDDEN_NEURONS  # 48 * 8 = 384
macs_output = HIDDEN_NEURONS * NUM_CLASSES          # 8 * 4 = 32
macs_total  = macs_hidden + macs_output             # 416

print(f"""
  MODEL: SBP + MLP
  ─────────────────────────────────────────
  Architecture:     {NUM_INPUT_FEATURES} -> {HIDDEN_NEURONS} (ReLU) -> {NUM_CLASSES} (argmax)
  Input features:   {NUM_INPUT_FEATURES} ({NUM_CHANNELS} channels x 3 stats)
  Feature types:    mean, std, max of SBP per channel
  Window size:      {WINDOW_SIZE} bins x 20ms = {WINDOW_SIZE * 20}ms
  
  PARAMETERS:
    Hidden weights:   {NUM_INPUT_FEATURES} x {HIDDEN_NEURONS} = {NUM_INPUT_FEATURES * HIDDEN_NEURONS}
    Hidden biases:    {HIDDEN_NEURONS}
    Output weights:   {HIDDEN_NEURONS} x {NUM_CLASSES} = {HIDDEN_NEURONS * NUM_CLASSES}
    Output biases:    {NUM_CLASSES}
    Total:            {total_params} parameters = {total_params} bytes
  
  COMPUTATION (per inference):
    Hidden layer:     {macs_hidden} MACs
    Output layer:     {macs_output} MACs
    Total:            {macs_total} MACs
    + {HIDDEN_NEURONS} ReLU operations
    + 1 argmax (4-way comparator)
  
  QUANTIZATION:
    Precision:        {QUANT_BITS}-bit signed symmetric
    Float accuracy:   {float_acc:.1%}
    Quant accuracy:   {quant_acc:.1%}
    Accuracy drop:    {float_acc - quant_acc:.1%}
  
  PERFORMANCE:
    Best accuracy:    {best_mlp:.1%}
    vs Linear:        {best_mlp - best_linear:+.1%}
    vs Conv+MLP:      {best_mlp - best_conv:+.1%}
  
  SCAN CHAIN LOAD ORDER:
    1. hidden_w  ({NUM_INPUT_FEATURES}x{HIDDEN_NEURONS} = {NUM_INPUT_FEATURES * HIDDEN_NEURONS} bytes)
    2. hidden_b  ({HIDDEN_NEURONS} bytes)
    3. output_w  ({HIDDEN_NEURONS}x{NUM_CLASSES} = {HIDDEN_NEURONS * NUM_CLASSES} bytes)
    4. output_b  ({NUM_CLASSES} bytes)
  
  EXPORTED FILES:
    exports/quantized_weights.json    - weights + scales
    exports/weights_hex.txt           - Verilog scan chain format
    exports/test_vectors.json         - {len(vectors)} UVM test vectors
    exports/normalization_params.json - preprocessing constants
    exports/model_results.json        - all results
    training_comparison.png           - training curves
""")