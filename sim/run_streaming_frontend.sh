#!/bin/tcsh
# Directed unit simulation for the streaming ADC/SBP frontend.
# Can be run from any directory.

setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

cd `dirname $0`

xrun -64bit -access r -nokey -timescale 1ns/1ps \
    ../rtl/streaming_sbp_frontend.sv \
    tb_streaming_sbp_frontend.sv
