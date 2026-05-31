# ============================================================
# syn.tcl  -  Genus synthesis script for bmi_chip_top
# Run from inside the syn/ directory via run_syn.sh
# ============================================================

# -- Read RTL (submodules before top) -------------------------
read_hdl -sv ../rtl/sample_collection.sv
read_hdl -sv ../rtl/sbp_feature_extractor.sv
read_hdl -sv ../rtl/mlp_inference.sv
read_hdl -sv ../rtl/argmax.sv
read_hdl -sv ../rtl/output_formatter.sv
read_hdl -sv ../rtl/spi_slave.sv
read_hdl -sv ../rtl/bmi_chip_top.sv

# -- Cell library (GPDK045 RVT, slow corner at 1.0 V) ---------
set_db library     /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/timing/slow_vdd1v0_basicCells.lib
set_db lef_library [list \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef]

# -- Elaborate ------------------------------------------------
elaborate
current_design bmi_chip_top

# -- Timing constraints ---------------------------------------
read_sdc bmi_chip_top.sdc

# -- Synthesize -----------------------------------------------
syn_generic
syn_map
syn_opt

# -- Reports --------------------------------------------------
report_timing > timing.rpt
report_area   > area.rpt

# -- Write gate-level netlist ---------------------------------
write_hdl > bmi_chip_top_syn.v

quit
