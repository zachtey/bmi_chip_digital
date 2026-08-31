#!/bin/tcsh
# RTL-only simulation of bmi_chip_top with 40 test vectors.
# Can be run from any directory.

# Set up Cadence tools
setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

# Run from sim/ so relative paths to vectors/ and weights.hex resolve
cd `dirname $0`

xrun -64bit -access r -nokey -timescale 1ns/1ps \
    ../rtl/bmi_chip_top.sv          \
    ../rtl/sample_collection.sv     \
    ../rtl/sbp_feature_extractor.sv \
    ../rtl/mlp_inference.sv         \
    ../rtl/argmax.sv                \
    ../rtl/output_formatter.sv      \
    ../rtl/spi_slave.sv             \
    tb_output_formatter.sv
