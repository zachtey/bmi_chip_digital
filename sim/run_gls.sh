#!/bin/tcsh
# Gate-level simulation of bmi_chip_top using the synthesized netlist.
# Functional only — no SDF timing annotation (+notimingchecks).

setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

cd `dirname $0`

set CELLS = /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/verilog
set NETLIST = ../syn/bmi_chip_top_syn.v

xrun -64bit -access r -nokey -timescale 1ns/1ps \
    +notimingchecks                             \
    +define+FUNCTIONAL                          \
    ${CELLS}/slow_vdd1v0_basicCells.v           \
    ${CELLS}/slow_vdd1v0_multibitsDFF.v         \
    ${NETLIST}                                   \
    tb_gls.sv
