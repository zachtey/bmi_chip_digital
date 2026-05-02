#!/usr/bin/env python3
"""
parse_test_vectors.py

Reads test_vectors.txt produced by bmi_pipeline_final.py and converts
each test vector into $readmemh-compatible hex files for Verilog testbenches.

Output per vector (in ./vectors/ folder):
  vecNN_adc.hex      — 2000 lines, each line = 1 ADC byte (ch0s0, ch0s1, ... ch7s249)
  vecNN_sbp.hex      —    8 lines, each line = 1 SBP byte (expected, for checking)
  vecNN_expected.hex —    1 line,  expected class (0-3)
  vecNN_scores.hex   —    4 lines, expected scores (signed 64-bit, for debug)

Also writes:
  vectors/summary.txt — human-readable summary of all vectors

Usage:
  python3 parse_test_vectors.py [test_vectors.txt]
"""

import os
import sys
import re

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "test_vectors.txt"
OUT_DIR     = "vectors"
N_CH        = 8
N_SAMPLES   = 250

# ── Setup ────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parse ────────────────────────────────────────────────────────────────────
with open(INPUT_FILE, 'r') as f:
    lines = f.readlines()

vectors = []
current = None

for line in lines:
    line = line.strip()

    # Start of a new vector
    m = re.match(r'// Vec (\d+): true=(\S+)\s+pred=(\S+)', line)
    if m:
        if current is not None:
            vectors.append(current)
        current = {
            'idx':       int(m.group(1)),
            'true_cls':  m.group(2),
            'pred_cls':  m.group(3),
            'exp_class': None,
            'scores':    [],
            'sbp':       [],
            'adc':       [[] for _ in range(N_CH)],  # adc[ch] = list of 250 ints
        }
        continue

    if current is None:
        continue

    # Expected class
    m = re.match(r'// Expected CLASS\[1:0\] = \d+ = (\d+)', line)
    if m:
        current['exp_class'] = int(m.group(1))
        continue

    # Scores
    m = re.match(r'// Scores: \[(.+)\]', line)
    if m:
        current['scores'] = [int(x.strip()) for x in m.group(1).split(',')]
        continue

    # SBP
    m = re.match(r'// SBP: \[(.+)\]', line)
    if m:
        current['sbp'] = [int(x.strip()) for x in m.group(1).split(',')]
        continue

    # ADC channel data: "// chN: HH HH HH ..."
    m = re.match(r'// ch(\d+): (.+)', line)
    if m:
        ch   = int(m.group(1))
        vals = [int(v, 16) for v in m.group(2).split()]
        current['adc'][ch] = vals
        continue

# Don't forget the last vector
if current is not None:
    vectors.append(current)

print(f"Parsed {len(vectors)} vectors from {INPUT_FILE}")

# ── Write output files ───────────────────────────────────────────────────────
summary_lines = []
summary_lines.append(f"{'Vec':>4}  {'True':>6}  {'Pred':>6}  {'Class':>5}  {'SBP'}")
summary_lines.append("-" * 70)

errors = 0
for v in vectors:
    idx = v['idx']
    prefix = os.path.join(OUT_DIR, f"vec{idx:02d}")

    # Validate
    ok = True
    for ch in range(N_CH):
        if len(v['adc'][ch]) != N_SAMPLES:
            print(f"  WARNING vec{idx:02d} ch{ch}: got {len(v['adc'][ch])} samples, expected {N_SAMPLES}")
            ok = False
    if len(v['sbp']) != N_CH:
        print(f"  WARNING vec{idx:02d}: got {len(v['sbp'])} SBP values, expected {N_CH}")
        ok = False
    if not ok:
        errors += 1

    # ── vecNN_adc.hex ────────────────────────────────────────────────────
    # Format: one byte per line, ch0 first (all 250 samples), then ch1, etc.
    # Verilog: $readmemh("vecNN_adc.hex", adc_mem)
    # where adc_mem[ch*250 + s] = adc_byte
    with open(f"{prefix}_adc.hex", 'w') as f:
        f.write(f"// vec{idx:02d} ADC samples: {N_CH} channels x {N_SAMPLES} samples\n")
        f.write(f"// Layout: ch0[0..249], ch1[0..249], ..., ch7[0..249]\n")
        for ch in range(N_CH):
            f.write(f"// ch{ch}:\n")
            for s in range(N_SAMPLES):
                val = v['adc'][ch][s] if s < len(v['adc'][ch]) else 0
                f.write(f"{val:02X}\n")

    # ── vecNN_sbp.hex ────────────────────────────────────────────────────
    # 8 lines, one SBP value per line (expected output of SBP block)
    with open(f"{prefix}_sbp.hex", 'w') as f:
        f.write(f"// vec{idx:02d} expected SBP[0..7]\n")
        for s in v['sbp']:
            f.write(f"{s:02X}\n")

    # ── vecNN_expected.hex ───────────────────────────────────────────────
    # 1 line: expected class (0-3)
    with open(f"{prefix}_expected.hex", 'w') as f:
        f.write(f"// vec{idx:02d} expected CLASS[1:0]\n")
        f.write(f"{v['exp_class']:02X}\n")

    # ── vecNN_scores.hex ─────────────────────────────────────────────────
    # 4 lines: expected MLP output scores (signed int64, stored as hex)
    # These are large — use 16 hex digits (64-bit)
    with open(f"{prefix}_scores.hex", 'w') as f:
        f.write(f"// vec{idx:02d} expected scores[0..3] (signed int64 as 64-bit hex)\n")
        for sc in v['scores']:
            # Convert signed int64 to unsigned 64-bit hex for $readmemh
            val = sc & 0xFFFFFFFFFFFFFFFF
            f.write(f"{val:016X}\n")

    # Summary line
    sbp_str = '[' + ', '.join(str(x) for x in v['sbp']) + ']'
    summary_lines.append(
        f"{idx:>4}  {v['true_cls']:>6}  {v['pred_cls']:>6}  "
        f"{v['exp_class']:>5}  {sbp_str}"
    )

# ── Summary file ─────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')

# ── Also write a flat all-vectors ADC file for simple testbenches ─────────────
# vectors/all_adc.hex  — all vectors back to back, same layout
# vectors/all_expected.hex — one expected class per line
with open(os.path.join(OUT_DIR, "all_expected.hex"), 'w') as f:
    f.write("// expected class for each vector (one per line)\n")
    for v in vectors:
        f.write(f"{v['exp_class']:02X}\n")

# ── Print summary ─────────────────────────────────────────────────────────────
print('\n'.join(summary_lines))
print()
print(f"Output written to: {OUT_DIR}/")
print(f"  vec00_adc.hex .. vec{vectors[-1]['idx']:02d}_adc.hex   ({N_CH*N_SAMPLES} lines each)")
print(f"  vec00_sbp.hex .. vec{vectors[-1]['idx']:02d}_sbp.hex   (8 lines each)")
print(f"  vec00_expected.hex .. (1 line each)")
print(f"  vec00_scores.hex   .. (4 lines each)")
print(f"  summary.txt")
print(f"  all_expected.hex")
if errors:
    print(f"\nWARNING: {errors} vectors had data issues — check output")
else:
    print(f"\nAll {len(vectors)} vectors parsed cleanly.")