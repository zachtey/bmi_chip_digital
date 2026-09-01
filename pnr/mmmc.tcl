# MMMC setup for the BMI digital block.
# This first backend pass uses the same slow GPDK45 library and SDC as Genus,
# so synthesis and Innovus analyze the same logical timing environment.

set pnr_dir   [file dirname [file normalize [info script]]]
set repo_root [file dirname $pnr_dir]

set slow_lib /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/timing/slow_vdd1v0_basicCells.lib
set design_sdc [file join $repo_root syn bmi_chip_top.sdc]

create_library_set -name slow_lib \
    -timing [list $slow_lib]

# The teaching PDK flow does not provide a separate QRC technology file here;
# Innovus uses its available/default pre-route RC model for this initial pass.
create_rc_corner -name rc_typ

create_delay_corner -name slow_corner \
    -library_set slow_lib \
    -rc_corner rc_typ

create_constraint_mode -name functional_mode \
    -sdc_files [list $design_sdc]

create_analysis_view -name slow_view \
    -constraint_mode functional_mode \
    -delay_corner slow_corner

set_analysis_view -setup [list slow_view] -hold [list slow_view]

