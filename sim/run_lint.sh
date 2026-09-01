#!/bin/sh
# Fast, synthesis-oriented RTL lint using Verilator.
# Run from any directory; warnings are treated as failures.

set -eu
cd "$(dirname "$0")"

verilator --lint-only --sv -Wall --top-module bmi_chip_top \
    ../rtl/streaming_sbp_frontend.sv \
    ../rtl/mlp_inference.sv \
    ../rtl/argmax.sv \
    ../rtl/output_formatter.sv \
    ../rtl/spi_slave.sv \
    ../rtl/bmi_chip_top.sv
