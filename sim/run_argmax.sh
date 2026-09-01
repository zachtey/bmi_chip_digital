#!/bin/tcsh
# Directed unit simulation for argmax.
# Can be run from any directory.

# Set up Cadence tools
setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

# Resolve source paths relative to this script.
cd `dirname $0`

xrun -64bit -access r -nokey -timescale 1ns/1ps \
    ../rtl/argmax.sv \
    tb_argmax.sv
