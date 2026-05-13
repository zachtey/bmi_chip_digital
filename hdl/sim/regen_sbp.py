#!/usr/bin/env python3
"""
regen_sbp.py

Recomputes vec*_sbp.hex from existing vec*_adc.hex using the
hardware-friendly formula:

    SBP[ch] = sum(|adc - 128|) >> 8

This matches what partner's sbp_feature_extractor.sv computes,
and what bmi_pipeline_final.py line 277 produces. Use this when
the SBP values in test_vectors.txt were generated from an older
version of the pipeline that used `// 250` instead.

Reads:
    vectors/vec00_adc.hex .. vec39_adc.hex   (2000 bytes each)

Writes:
    vectors/vec00_sbp.hex .. vec39_sbp.hex   (8 bytes each)

Run from hdl/sim/ folder (or anywhere the vectors/ folder is).

Usage:
    python3 regen_sbp.py
"""

import os
import sys

VECTORS_DIR = "vectors"
N_VECTORS   = 40
N_CH        = 8
N_SAMPLES   = 250


def read_hex_bytes(path):
    """Read a $readmemh-style hex file. Skip // comments and blank lines."""
    bytes_out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            # Each line is one byte in hex
            bytes_out.append(int(line, 16))
    return bytes_out


def compute_sbp(adc_bytes):
    """
    adc_bytes: flat list of 2000 ints (ch0[0..249], ch1[0..249], ..., ch7[0..249])
    Returns: list of 8 SBP values, one per channel
    """
    sbp = []
    for ch in range(N_CH):
        acc = 0
        for s in range(N_SAMPLES):
            sample = adc_bytes[ch * N_SAMPLES + s]
            acc += abs(sample - 128)
        sbp.append(acc >> 8)   # divide by 256 — matches RTL exactly
    return sbp


def main():
    if not os.path.isdir(VECTORS_DIR):
        print(f"ERROR: {VECTORS_DIR}/ folder not found.")
        print("Run this script from the folder containing vectors/.")
        sys.exit(1)

    print(f"Regenerating {N_VECTORS} SBP files from ADC vectors...")
    print(f"Formula: SBP[ch] = sum(|adc - 128|) >> 8")
    print()

    for i in range(N_VECTORS):
        adc_path = os.path.join(VECTORS_DIR, f"vec{i:02d}_adc.hex")
        sbp_path = os.path.join(VECTORS_DIR, f"vec{i:02d}_sbp.hex")

        # Read ADC bytes
        adc_bytes = read_hex_bytes(adc_path)
        if len(adc_bytes) != N_CH * N_SAMPLES:
            print(f"  WARNING vec{i:02d}: got {len(adc_bytes)} bytes, expected {N_CH * N_SAMPLES}")
            continue

        # Compute SBP
        sbp = compute_sbp(adc_bytes)

        # Write SBP file (same format as parse_test_vectors.py uses)
        with open(sbp_path, 'w') as f:
            f.write(f"// vec{i:02d} expected SBP[0..7] (regen with >> 8)\n")
            for v in sbp:
                f.write(f"{v:02X}\n")

        sbp_str = ', '.join(str(v) for v in sbp)
        print(f"  vec{i:02d}: [{sbp_str}]")

    print()
    print(f"Done. Updated {N_VECTORS} files in {VECTORS_DIR}/")


if __name__ == "__main__":
    main()