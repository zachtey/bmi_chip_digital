#!/bin/tcsh
# Directed unit simulation for mlp_inference.
# Can be run from any directory.

# Northwestern Cadence installation and license configuration.
setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

# Resolve source paths relative to this script.
cd `dirname $0`

# Only the MLP RTL and its unit testbench are needed for this regression.
xrun -64bit -access r -nokey -timescale 1ns/1ps \
    ../rtl/mlp_inference.sv \
    tb_mlp_inference.sv
