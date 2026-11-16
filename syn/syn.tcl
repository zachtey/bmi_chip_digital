# ============================================================
# syn.tcl  -  Genus synthesis script for bmi_chip_top
# Run from inside the syn/ directory via run_syn.sh
# ============================================================

# -- Read RTL (submodules before top) -------------------------
read_hdl -sv ../rtl/cdc_sync_2ff.sv
read_hdl -sv ../rtl/reset_synchronizer.sv
read_hdl -sv ../rtl/cdc_handshake.sv
read_hdl -sv ../rtl/cdc_async_fifo.sv
read_hdl -sv ../rtl/streaming_sbp_frontend.sv
read_hdl -sv ../rtl/mlp_inference.sv
read_hdl -sv ../rtl/argmax.sv
read_hdl -sv ../rtl/bmi_pipeline_ctrl.sv
read_hdl -sv ../rtl/apb_slave.sv
# Previous interface (scan weight loader, packet formatter, SPI transmitter).
# These are not instantiated by bmi_chip_top any more; read them again only
# if that interface is restored in bmi_chip_top.sv.
# read_hdl -sv ../rtl/weight_scan_loader.sv
# read_hdl -sv ../rtl/output_formatter.sv
# read_hdl -sv ../rtl/spi_slave.sv
read_hdl -sv ../rtl/bmi_chip_top.sv

# -- Cell library (GPDK045 RVT, slow corner at 1.0 V) ---------
set_db library     /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/timing/slow_vdd1v0_basicCells.lib
set_db lef_library [list \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef \
    /vol/eecs391/GPDK045/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef]

# -- Elaborate ------------------------------------------------
elaborate
current_design bmi_chip_top

# Structural sanity check before constraints or optimization. This catches
# unresolved references and other elaborated-design problems independently of
# simulation and Verilator lint.
check_design -unresolved > check_design.rpt

# -- Timing constraints ---------------------------------------
read_sdc bmi_chip_top.sdc

# -- Synthesize -----------------------------------------------
syn_generic
syn_map
syn_opt

# -- Reports --------------------------------------------------
report_timing > timing.rpt
report timing -lint > timing_lint.rpt
report_area   > area.rpt

# -- Write gate-level netlist (for Innovus PNR) ---------------
write_hdl > bmi_chip_top_syn.v

# -- Write Virtuoso symbol stub (for schematic import only) ---
# This is a port-declaration-only file. It is NOT synthesized
# and NOT used by Innovus. Import this .v into Virtuoso to
# generate a symbol that includes VDD and VSS pins.
set f [open "bmi_chip_top_sym.v" w]
puts $f "// Virtuoso symbol import stub — do NOT use for synthesis or PNR"
puts $f "// Import into Virtuoso to create a symbol with VDD/VSS pins"
puts $f "module bmi_chip_top ("
puts $f "    input  wire        bmi_clk,"
puts $f "    input  wire        pclk,"
puts $f "    input  wire        presetn,"
puts $f "    input  wire \[7:0\]  adc_sample,"
puts $f "    output wire \[2:0\]  adc_channel,"
puts $f "    input  wire        psel,"
puts $f "    input  wire        penable,"
puts $f "    input  wire        pwrite,"
puts $f "    input  wire \[11:0\] paddr,"
puts $f "    input  wire \[31:0\] pwdata,"
puts $f "    output wire \[31:0\] prdata,"
puts $f "    output wire        pready,"
puts $f "    output wire        pslverr,"
puts $f "    output wire        irq,"
puts $f "    inout  wire        VDD,"
puts $f "    inout  wire        VSS"
puts $f ");"
puts $f "endmodule"
close $f

puts "\nOutputs:"
puts "  bmi_chip_top_syn.v  — gate-level netlist for Innovus"
puts "  bmi_chip_top_sym.v  — Virtuoso symbol stub (with VDD/VSS)"
puts "  check_design.rpt     — Genus structural design checks"
puts "  timing_lint.rpt       — Genus timing-constraint coverage checks"

quit
