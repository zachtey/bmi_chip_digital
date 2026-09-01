#!/bin/tcsh
# Run the import/floorplan checkpoint in batch mode.
# Outputs are isolated under pnr/work, pnr/reports, and pnr/logs.

cd `dirname $0`
mkdir -p work reports/init logs

setenv PATH ${PATH}:/vol/cadence2018/INNOVUS181/bin:/vol/cadence2017/INNOVUS162/bin
if (! $?CDS_LIC_FILE) then
    setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
endif

innovus -no_gui -overwrite -init pnr_init.tcl -log logs/innovus_init

