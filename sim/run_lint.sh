#!/bin/sh
# Fast, synthesis-oriented RTL lint using Verilator.
# Run from any directory; warnings are treated as failures.

set -eu
cd "$(dirname "$0")"

verilator --lint-only --sv -Wall --top-module bmi_chip_top \
    ../rtl/cdc_sync_2ff.sv \
    ../rtl/reset_synchronizer.sv \
    ../rtl/cdc_handshake.sv \
    ../rtl/cdc_async_fifo.sv \
    ../rtl/bmi_pipeline_ctrl.sv \
    ../rtl/streaming_sbp_frontend.sv \
    ../rtl/mlp_inference.sv \
    ../rtl/argmax.sv \
    ../rtl/apb_slave.sv \
    ../rtl/bmi_chip_top.sv

# The previous interface blocks are no longer instantiated by bmi_chip_top
# (the APB register block replaced them), but they are kept in rtl/ and still
# have unit tests, so lint each one on its own.
for block in weight_scan_loader output_formatter spi_slave; do
    verilator --lint-only --sv -Wall --top-module "$block" \
        "../rtl/$block.sv"
done
