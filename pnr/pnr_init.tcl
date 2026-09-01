# BMI ASIC Innovus flow -- checkpoint 1: import and floorplan only.
# Run with ./run_pnr_init.sh from any directory.

set pnr_dir   [file dirname [file normalize [info script]]]
set repo_root [file dirname $pnr_dir]

set work_dir   [file join $pnr_dir work]
set report_dir [file join $pnr_dir reports init]
file mkdir $work_dir
file mkdir $report_dir

# -----------------------------------------------------------------------------
# 1. Design inputs
# -----------------------------------------------------------------------------
set init_top_cell bmi_chip_top
set init_verilog  [file join $repo_root syn bmi_chip_top_syn.v]
set init_mmmc_file [file join $pnr_dir mmmc.tcl]

set init_lef_file [list \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045_tech/lef/gsclib045_tech.lef \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef]

set init_pwr_net VDD
set init_gnd_net VSS

init_design
setDesignMode -process 45

# -----------------------------------------------------------------------------
# 2. Floorplan
# -----------------------------------------------------------------------------
# -r aspect_ratio target_utilization left bottom right top
# A 1.0 aspect ratio requests a square core. Sixty-percent utilization means
# standard cells initially occupy about 60% of the core, leaving whitespace for
# placement optimization, CTS buffers, hold fixes, and routing.
floorPlan -r 1.0 0.60 10 10 10 10

# -----------------------------------------------------------------------------
# 3. Logical pin groups
# -----------------------------------------------------------------------------
# Keep interfaces together so routing is understandable and repeatable. Metal4
# is used for boundary pins, matching the established GPDK45 teaching flow.
editPin -pin {adc_sample[0] adc_sample[1] adc_sample[2] adc_sample[3] \
              adc_sample[4] adc_sample[5] adc_sample[6] adc_sample[7] \
              adc_channel[0] adc_channel[1] adc_channel[2]} \
        -side TOP -layer Metal4 -spreadType SIDE

editPin -pin {spi_sclk spi_cs_n spi_miso} \
        -side RIGHT -layer Metal4 -spreadType SIDE

editPin -pin {scan_clk scan_en scan_in} \
        -side BOTTOM -layer Metal4 -spreadType SIDE

editPin -pin {clk rst_n} \
        -side LEFT -layer Metal4 -spreadType SIDE

# -----------------------------------------------------------------------------
# 4. Sanity reports and checkpoint
# -----------------------------------------------------------------------------
redirect [file join $report_dir check_design.rpt] {
    checkDesign -all
}

redirect [file join $report_dir design_summary.rpt] {
    summaryReport
}

set fp [open [file join $report_dir floorplan_summary.rpt] w]
puts $fp "Design: [dbGet top.name]"
puts $fp "Die box: [dbGet top.fPlan.box]"
puts $fp "Core box: [dbGet top.fPlan.coreBox]"
puts $fp "Instances: [llength [dbGet top.insts.name]]"
puts $fp "Nets: [llength [dbGet top.nets.name]]"
puts $fp "Ports: [dbGet top.terms.name]"
close $fp

saveDesign [file join $work_dir 01_floorplan.enc]

puts ""
puts "PNR checkpoint 1 complete."
puts "Saved database: [file join $work_dir 01_floorplan.enc]"
puts "Reports:       $report_dir"
puts "Inspect the floorplan before adding the power grid."

exit

