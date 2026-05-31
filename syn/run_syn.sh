#!/bin/tcsh
# Cadence Genus synthesis for bmi_chip_top
# Run from any directory; outputs land in syn/

setenv PATH ${PATH}:/vol/cadence2018/GENUS181/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu

cd `dirname $0`

genus -no_gui -overwrite -files syn.tcl |& tee genus.log
