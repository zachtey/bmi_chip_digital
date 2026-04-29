#!/usr/bin/env python3
"""
RTL SUBBLOCK VERIFICATION SCRIPT
==================================
Reads test_vectors.txt and weights_readable.txt produced by bmi_final.py
and computes the EXACT expected output of EVERY pipeline stage.

Use this to verify each Verilog subblock independently before integration.

STAGES VERIFIED:
  1. sample_collection  → raw ADC window (already in test_vectors.txt)
  2. sbp_extraction     → 8 SBP integers
  3. mlp_hidden         → 8 hidden activations (after ReLU)
  4. mlp_output         → 4 class scores
  5. argmax             → predicted class, CLASS[1:0]

OUTPUT FILES:
  subblock_vectors.txt  → all intermediate values, copy-paste into Verilog TB
  subblock_vectors.hex  → hex format for $readmemh-style loading

USAGE:
  python verify_subblocks.py
  (run in same folder as test_vectors.txt and weights.hex from bmi_final.py)
"""

import os, sys, re
import numpy as np

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — update these to match your bmi_final.py run
# ══════════════════════════════════════════════════════════════
N_CH      = 8
N_HIDDEN  = 8
N_OUT     = 4
SAMPLES   = 250
CLASS_NAMES = ['PG-LF', 'PG-HF', 'SG-LF', 'SG-HF']

# These come from your weights.hex header (printed at the end of bmi_final.py)
# The script will try to read them from test_vectors.txt automatically,
# but you can hardcode them here if the auto-detection fails.
HIDDEN_BIAS_SCALE = None   # will be read from test_vectors.txt
OUTPUT_BIAS_SCALE = None


# ══════════════════════════════════════════════════════════════
# 1. LOAD TEST VECTORS
# ══════════════════════════════════════════════════════════════
def load_test_vectors(path='test_vectors.txt'):
    """
    Parse test_vectors.txt and return list of dicts:
      { 'idx', 'true_class', 'pred_class', 'scores', 'sbp', 'adc' }
    Also extracts hidden_bias_scale and output_bias_scale from header.
    """
    global HIDDEN_BIAS_SCALE, OUTPUT_BIAS_SCALE

    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run bmi_final.py first.")
        sys.exit(1)

    vectors = []
    current = {}
    adc_rows = []

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            # Header: extract bias scales
            if 'hidden_bias_scale' in line:
                m = re.search(r'hidden_bias_scale\s*=\s*(\d+)', line)
                if m and HIDDEN_BIAS_SCALE is None:
                    HIDDEN_BIAS_SCALE = int(m.group(1))
            if 'output_bias_scale' in line:
                m = re.search(r'output_bias_scale\s*=\s*(\d+)', line)
                if m and OUTPUT_BIAS_SCALE is None:
                    OUTPUT_BIAS_SCALE = int(m.group(1))

            # New vector
            m = re.match(r'// Vec (\d+): true=(\S+)\s+pred=(\S+)', line)
            if m:
                if current and adc_rows:
                    current['adc'] = np.array(adc_rows, dtype=np.int32)
                    vectors.append(current)
                current = {
                    'idx':        int(m.group(1)),
                    'true_class': CLASS_NAMES.index(m.group(2)) if m.group(2) in CLASS_NAMES else -1,
                    'pred_class': CLASS_NAMES.index(m.group(3)) if m.group(3) in CLASS_NAMES else -1,
                }
                adc_rows = []
                continue

            # Expected CLASS
            m = re.match(r'// Expected CLASS\[1:0\] = (\d+) = (\d+)', line)
            if m:
                current['expected_class_bits'] = m.group(1)
                current['expected_class']       = int(m.group(2))
                continue

            # Scores
            m = re.match(r'// Scores: \[([^\]]+)\]', line)
            if m:
                current['golden_scores'] = [int(x.strip()) for x in m.group(1).split(',')]
                continue

            # SBP integers
            m = re.match(r'// SBP: \[([^\]]+)\]', line)
            if m:
                current['golden_sbp'] = [int(x.strip()) for x in m.group(1).split(',')]
                continue

            # ADC channel data
            m = re.match(r'// ch(\d+): ([0-9A-Fa-f ]+)', line)
            if m:
                ch   = int(m.group(1))
                vals = [int(h, 16) for h in m.group(2).strip().split()]
                adc_rows.append(vals)
                continue

    # Flush last vector
    if current and adc_rows:
        current['adc'] = np.array(adc_rows, dtype=np.int32)
        vectors.append(current)

    return vectors


# ══════════════════════════════════════════════════════════════
# 2. LOAD WEIGHTS FROM weights.hex
# ══════════════════════════════════════════════════════════════
def load_weights(path='weights.hex'):
    """
    Parse weights.hex (108 bytes) into hw, hb, ow, ob arrays.
    Layout: hw[N_HIDDEN][N_CH], hb[N_HIDDEN], ow[N_OUT][N_HIDDEN], ob[N_OUT]
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run bmi_final.py first.")
        sys.exit(1)

    bytes_list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('//') or not line:
                continue
            bytes_list.append(int(line, 16))

    # Convert unsigned bytes to signed int8
    w = np.array(bytes_list, dtype=np.uint8).view(np.int8)

    idx = 0
    hw = w[idx:idx + N_HIDDEN * N_CH].reshape(N_HIDDEN, N_CH).astype(np.int32)
    idx += N_HIDDEN * N_CH

    hb = w[idx:idx + N_HIDDEN].astype(np.int32)
    idx += N_HIDDEN

    ow = w[idx:idx + N_OUT * N_HIDDEN].reshape(N_OUT, N_HIDDEN).astype(np.int32)
    idx += N_OUT * N_HIDDEN

    ob = w[idx:idx + N_OUT].astype(np.int32)

    return hw, hb, ow, ob


# ══════════════════════════════════════════════════════════════
# 3. PIPELINE STAGES — integer-only, matches Verilog exactly
# ══════════════════════════════════════════════════════════════

def stage_sbp(adc_window):
    """
    D2: SBP feature extraction.
    Input:  (8, 250) int32, values 0-255
    Output: (8,) int32, values 0-127

    Verilog implementation:
      sum  = 0
      for s in 0..249: sum += |adc[ch][s] - 128|
      sbp[ch] = sum >> 8

    Note: right-shift by 8 (÷256) not divide by 250.
    This is the hardware-accurate version — matches RTL exactly.
    """
    sbp = np.zeros(N_CH, dtype=np.int64)
    for ch in range(N_CH):
        acc = np.int64(0)
        for s in range(SAMPLES):
            acc += abs(int(adc_window[ch, s]) - 128)
        sbp[ch] = acc >> 8    # right-shift, NOT divide by 250
    return sbp.astype(np.int32)


def stage_mlp_hidden(sbp, hw, hb, hidden_bias_scale):
    """
    D4: MLP hidden layer (8 neurons, ReLU).
    Input:  sbp (8,) int32
    Output: hidden_act (8,) int64  — stored as 32-bit in hardware

    Verilog FSM states: S_H_MAC (8 cycles) → S_H_BIAS (1 cycle) → S_H_RELU (1 cycle)
    Repeat for each of 8 neurons.

    Accumulator arithmetic:
      acc = 0
      for i in 0..7: acc += sbp[i] * hw[neuron][i]   (8-bit × 8-bit = 16-bit product)
      acc += hb[neuron] * HIDDEN_BIAS_SCALE
      hidden_act[neuron] = max(0, acc)
    """
    hidden = np.zeros(N_HIDDEN, dtype=np.int64)
    for j in range(N_HIDDEN):
        acc = np.int64(0)
        for i in range(N_CH):
            acc += np.int64(sbp[i]) * np.int64(hw[j, i])
        acc += np.int64(hb[j]) * np.int64(hidden_bias_scale)
        hidden[j] = max(np.int64(0), acc)    # ReLU
    return hidden


def stage_mlp_output(hidden, ow, ob, output_bias_scale):
    """
    D5: MLP output layer (4 neurons, no ReLU).
    Input:  hidden (8,) int64
    Output: scores (4,) int64

    Verilog FSM states: S_O_MAC (8 cycles) → S_O_BIAS (1 cycle) → S_O_STORE (1 cycle)
    Repeat for each of 4 output neurons.

    No ReLU — scores can be negative (that's fine, argmax still works).
    """
    scores = np.zeros(N_OUT, dtype=np.int64)
    for k in range(N_OUT):
        acc = np.int64(0)
        for j in range(N_HIDDEN):
            acc += np.int64(hidden[j]) * np.int64(ow[k, j])
        acc += np.int64(ob[k]) * np.int64(output_bias_scale)
        scores[k] = acc
    return scores


def stage_argmax(scores):
    """
    D6: Argmax — pick highest score.
    Input:  scores (4,) int64
    Output: (predicted_class int, CLASS[1:0] 2-bit string)

    Verilog: 3-comparator tournament.
      if score[0] >= score[1]: best01 = 0 else best01 = 1
      if score[2] >= score[3]: best23 = 2 else best23 = 3
      if score[best01] >= score[best23]: class = best01 else class = best23
    """
    pred = int(np.argmax(scores))
    return pred, f"{pred:02b}"


# ══════════════════════════════════════════════════════════════
# 4. RUN ALL TEST VECTORS
# ══════════════════════════════════════════════════════════════
def run_pipeline(vectors, hw, hb, ow, ob):
    results = []
    for vec in vectors:
        adc    = vec['adc']   # (8, 250) int32
        sbp    = stage_sbp(adc)
        hidden = stage_mlp_hidden(sbp, hw, hb, HIDDEN_BIAS_SCALE)
        scores = stage_mlp_output(hidden, ow, ob, OUTPUT_BIAS_SCALE)
        pred, class_bits = stage_argmax(scores)
        results.append({
            'idx':        vec['idx'],
            'true_class': vec['true_class'],
            'adc':        adc,
            'sbp':        sbp,
            'hidden':     hidden,
            'scores':     scores,
            'pred':       pred,
            'class_bits': class_bits,
        })
    return results


# ══════════════════════════════════════════════════════════════
# 5. WRITE VERIFICATION FILES
# ══════════════════════════════════════════════════════════════
def write_verification_file(results, path='subblock_vectors.txt'):
    """
    Write a file with expected outputs at each pipeline stage.
    Format: easy to grep in Verilog simulation output.
    """
    with open(path, 'w') as f:
        f.write("// ═══════════════════════════════════════════════════════\n")
        f.write("// RTL SUBBLOCK VERIFICATION — INTERMEDIATE EXPECTED VALUES\n")
        f.write("// ═══════════════════════════════════════════════════════\n")
        f.write("//\n")
        f.write("// HOW TO USE:\n")
        f.write("// For each vector:\n")
        f.write("//   1. Feed ADC input to sample_collection\n")
        f.write("//   2. After window_ready: compare SBP outputs\n")
        f.write("//   3. After sbp_done:     compare hidden_act values\n")
        f.write("//   4. After mlp_done:     compare class_scores\n")
        f.write("//   5. After done:         compare CLASS[1:0]\n")
        f.write("//\n")
        f.write(f"// hidden_bias_scale = {HIDDEN_BIAS_SCALE}\n")
        f.write(f"// output_bias_scale = {OUTPUT_BIAS_SCALE}\n")
        f.write(f"// Note: SBP uses >> 8 (divide by 256), not divide by 250\n")
        f.write("//\n\n")

        for r in results:
            f.write(f"// ══ Vector {r['idx']:02d} ".ljust(60, '═') + '\n')
            f.write(f"// True class: {CLASS_NAMES[r['true_class']]}  "
                    f"Predicted: {CLASS_NAMES[r['pred']]}  "
                    f"{'✓ CORRECT' if r['pred'] == r['true_class'] else '✗ WRONG'}\n\n")

            # ── Stage 1: ADC input ────────────────────────────
            f.write("// ── STAGE 1: sample_collection output (ADC window)\n")
            f.write("// Feed these hex values into adc_sample/adc_channel/adc_valid\n")
            for ch in range(N_CH):
                vals = ' '.join(f"{int(r['adc'][ch,s]):02X}" for s in range(SAMPLES))
                f.write(f"// ch{ch}: {vals}\n")

            # ── Stage 2: SBP ─────────────────────────────────
            f.write("\n// ── STAGE 2: sbp_feature_extraction output\n")
            f.write("// Check sbp_features[0..7] after sbp_done pulse\n")
            sbp_dec = ' '.join(f"{int(v):4d}" for v in r['sbp'])
            sbp_hex = ' '.join(f"{int(v) & 0xFF:02X}" for v in r['sbp'])
            f.write(f"// sbp_features (decimal): [{sbp_dec} ]\n")
            f.write(f"// sbp_features (hex):     [{sbp_hex}]\n")
            f.write("// Verilog check:\n")
            for ch in range(N_CH):
                f.write(f"//   assert(sbp_features[{ch}] === 8'h{int(r['sbp'][ch]) & 0xFF:02X})  "
                        f"// = {int(r['sbp'][ch])}\n")

            # ── Stage 3: MLP hidden ───────────────────────────
            f.write("\n// ── STAGE 3: mlp_inference hidden activations\n")
            f.write("// Check hidden_act[0..7] after S_H_RELU of last neuron\n")
            f.write("// (These are internal registers — expose via scan or add probe output)\n")
            for j in range(N_HIDDEN):
                v = int(r['hidden'][j])
                f.write(f"//   hidden_act[{j}] = 32'h{v & 0xFFFFFFFF:08X}  // = {v}\n")

            # ── Stage 4: MLP output scores ────────────────────
            f.write("\n// ── STAGE 4: mlp_inference class_scores output\n")
            f.write("// Check class_scores[0..3] after mlp_done pulse\n")
            for k in range(N_OUT):
                v = int(r['scores'][k])
                f.write(f"//   class_scores[{k}] ({CLASS_NAMES[k]:>6s}) = "
                        f"32'h{v & 0xFFFFFFFF:08X}  // = {v}\n")

            # ── Stage 5: Argmax ───────────────────────────────
            f.write("\n// ── STAGE 5: argmax output\n")
            f.write(f"//   predicted_class = 2'b{r['class_bits']}  // = {r['pred']} "
                    f"({CLASS_NAMES[r['pred']]})\n")
            f.write("// Verilog check:\n")
            f.write(f"//   assert(predicted_class === 2'b{r['class_bits']});\n")
            f.write('\n')


def write_tb_tasks(results, path='tb_tasks.sv'):
    """
    Write SystemVerilog tasks that you can call in your testbench.
    Each task feeds one vector through the DUT and checks at each stage.
    """
    with open(path, 'w') as f:
        f.write("// ═══════════════════════════════════════════════════════\n")
        f.write("// AUTO-GENERATED TESTBENCH TASKS\n")
        f.write("// Include this file in your top-level testbench.\n")
        f.write("// ═══════════════════════════════════════════════════════\n\n")
        f.write("`define CHECK(signal, expected, msg) \\\n")
        f.write("  if ((signal) !== (expected)) begin \\\n")
        f.write("    $display(\"FAIL: %s  got=%0d  exp=%0d\", msg, signal, expected); \\\n")
        f.write("    fail_count = fail_count + 1; \\\n")
        f.write("  end else begin \\\n")
        f.write("    $display(\"PASS: %s\", msg); \\\n")
        f.write("    pass_count = pass_count + 1; \\\n")
        f.write("  end\n\n")

        f.write("integer pass_count = 0;\n")
        f.write("integer fail_count = 0;\n\n")

        for r in results:
            f.write(f"// ─── Vector {r['idx']:02d}: {CLASS_NAMES[r['true_class']]} → {CLASS_NAMES[r['pred']]} ───\n")
            f.write(f"task run_vector_{r['idx']:02d};\n")
            f.write("begin\n")

            # Feed ADC samples
            f.write(f"  $display(\"--- Vector {r['idx']:02d}: true={CLASS_NAMES[r['true_class']]} ---\");\n")
            f.write("  // Feed ADC samples (round-robin ch 0..7, 250 rounds)\n")
            f.write("  for (int round = 0; round < 250; round++) begin\n")
            for ch in range(N_CH):
                # Write a few representative samples — full version would be too long
                # Instead reference the hex array
                pass
            f.write("    for (int ch = 0; ch < 8; ch++) begin\n")
            f.write(f"      @(posedge clk);\n")
            f.write(f"      adc_valid   = 1;\n")
            f.write(f"      adc_channel = ch;\n")
            f.write(f"      adc_sample  = adc_mem_{r['idx']:02d}[ch][round];\n")
            f.write("    end\n")
            f.write("    @(posedge clk); adc_valid = 0;\n")
            f.write("  end\n\n")

            # Wait for window_ready
            f.write("  // Wait for window_ready\n")
            f.write("  @(posedge window_ready);\n")
            f.write("  @(posedge clk);\n\n")

            # Check SBP outputs
            f.write("  // Check SBP features (after sbp_done)\n")
            f.write("  @(posedge sbp_done);\n")
            f.write("  @(posedge clk);\n")
            for ch in range(N_CH):
                f.write(f"  `CHECK(sbp_features[{ch}], 8'h{int(r['sbp'][ch])&0xFF:02X}, "
                        f"\"vec{r['idx']:02d} sbp[{ch}]={int(r['sbp'][ch])}\");\n")

            # Check MLP scores
            f.write("\n  // Check MLP class scores (after mlp_done)\n")
            f.write("  @(posedge mlp_done);\n")
            f.write("  @(posedge clk);\n")
            for k in range(N_OUT):
                v = int(r['scores'][k])
                f.write(f"  `CHECK(class_scores[{k}], 32'h{v&0xFFFFFFFF:08X}, "
                        f"\"vec{r['idx']:02d} score[{k}]={v}\");\n")

            # Check final prediction
            f.write(f"\n  // Check final prediction\n")
            f.write(f"  `CHECK(predicted_class, 2'b{r['class_bits']}, "
                    f"\"vec{r['idx']:02d} class={r['pred']} ({CLASS_NAMES[r['pred']]})\");\n")
            f.write("end\n")
            f.write("endtask\n\n")

        # ADC memory arrays (one per vector)
        f.write("\n// ─── ADC input memory arrays ───\n")
        f.write("// Declare in your testbench module:\n")
        for r in results:
            f.write(f"// reg [7:0] adc_mem_{r['idx']:02d} [0:7][0:249];\n")
        f.write("\n// Initialize with $readmemh or inline:\n")
        for r in results:
            f.write(f"task load_adc_{r['idx']:02d};\nbegin\n")
            for ch in range(N_CH):
                vals = r['adc'][ch]
                for s in range(SAMPLES):
                    f.write(f"  adc_mem_{r['idx']:02d}[{ch}][{s}] = 8'h{int(vals[s]):02X};\n")
            f.write("end\nendtask\n\n")

        # Main test runner
        f.write("// ─── Main test runner ───\n")
        f.write("task run_all_vectors;\nbegin\n")
        for r in results:
            f.write(f"  load_adc_{r['idx']:02d}();\n")
            f.write(f"  run_vector_{r['idx']:02d}();\n")
        f.write("  $display(\"\\n=== RESULTS: %0d PASS, %0d FAIL ===\", pass_count, fail_count);\n")
        f.write("end\nendtask\n")


# ══════════════════════════════════════════════════════════════
# 6. PRINT SUMMARY TABLE TO TERMINAL
# ══════════════════════════════════════════════════════════════
def print_summary(results):
    print(f"\n{'='*72}")
    print("PIPELINE VERIFICATION SUMMARY")
    print(f"{'='*72}")
    print(f"{'Vec':>4}  {'True':>6}  {'Pred':>6}  "
          f"{'SBP[0..7]':^36}  {'Scores':^24}  {'OK?':>5}")
    print("─"*72)

    n_correct = 0
    for r in results:
        sbp_str    = ' '.join(f"{int(v):3d}" for v in r['sbp'])
        scores_str = ' '.join(f"{int(v)//1000:5d}k" if abs(int(v)) > 999
                               else f"{int(v):6d}" for v in r['scores'])
        correct = (r['pred'] == r['true_class'])
        n_correct += correct
        mark = "✓" if correct else "✗"
        print(f"  {r['idx']:02d}  {CLASS_NAMES[r['true_class']]:>6}  "
              f"{CLASS_NAMES[r['pred']]:>6}  {sbp_str}  {scores_str}  {mark}")

    print("─"*72)
    print(f"  Accuracy on test vectors: {n_correct}/{len(results)} = "
          f"{100*n_correct/len(results):.1f}%")
    print(f"  (This should match the 'Quantized' accuracy from bmi_final.py)")

    print(f"\n{'─'*60}")
    print("EXPECTED INTERMEDIATE VALUES (first vector only):")
    r = results[0]
    adc_preview = ' '.join(f'{int(r["adc"][0,s]):02X}' for s in range(8))
    print(f"\n  [Stage 1] ADC input — ch0 first 8 samples:")
    print(f"    {adc_preview} ...")

    print(f"\n  [Stage 2] SBP features (after >> 8):")
    for ch in range(N_CH):
        print(f"    sbp[{ch}] = {int(r['sbp'][ch]):3d}  (0x{int(r['sbp'][ch])&0xFF:02X})")

    print(f"\n  [Stage 3] Hidden activations (after ReLU):")
    for j in range(N_HIDDEN):
        v = int(r['hidden'][j])
        print(f"    hidden[{j}] = {v:12d}  (0x{v&0xFFFFFFFF:08X})")

    print(f"\n  [Stage 4] Class scores:")
    for k in range(N_OUT):
        v = int(r['scores'][k])
        print(f"    score[{k}] ({CLASS_NAMES[k]:>6s}) = {v:15d}  (0x{v&0xFFFFFFFF:08X})")

    print(f"\n  [Stage 5] Argmax:")
    print(f"    CLASS[1:0] = 2'b{r['class_bits']} = {r['pred']} ({CLASS_NAMES[r['pred']]})")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading test vectors...")
    vectors = load_test_vectors('test_vectors.txt')
    print(f"  {len(vectors)} vectors loaded")

    if HIDDEN_BIAS_SCALE is None or OUTPUT_BIAS_SCALE is None:
        print("ERROR: could not read bias scales from test_vectors.txt")
        print("  Check the file header for hidden_bias_scale and output_bias_scale")
        print("  Or hardcode them at the top of this script")
        sys.exit(1)

    print(f"  hidden_bias_scale = {HIDDEN_BIAS_SCALE}")
    print(f"  output_bias_scale = {OUTPUT_BIAS_SCALE}")

    print("Loading weights...")
    hw, hb, ow, ob = load_weights('weights.hex')
    print(f"  hw: {hw.shape}  hb: {hb.shape}  ow: {ow.shape}  ob: {ob.shape}")

    print("Running pipeline stages...")
    results = run_pipeline(vectors, hw, hb, ow, ob)

    print_summary(results)

    print("\nWriting output files...")
    write_verification_file(results, 'subblock_vectors.txt')
    print("  subblock_vectors.txt — human-readable expected values per stage")

    write_tb_tasks(results, 'tb_tasks.sv')
    print("  tb_tasks.sv          — SystemVerilog tasks for your testbench")

    print("\nDONE. How to use:")
    print("  1. Verify SBP block:  compare sbp_features[0..7] against Stage 2 values")
    print("  2. Verify MLP hidden: add probe output for hidden_act, compare Stage 3")
    print("  3. Verify MLP output: compare class_scores[0..3] against Stage 4 values")
    print("  4. Verify argmax:     compare CLASS[1:0] against Stage 5 values")
    print("  5. All 40 vectors must match for RTL sign-off")