# ============================================================
# bmi_chip_top.sdc  -  Timing constraints for Genus synthesis
#
# Synthesis target:  50 MHz (20 ns period)
#   Actual operating frequency will be 2-10 MHz in silicon,
#   giving 5-25x margin.  Synthesizing to 50 MHz ensures
#   Genus picks reasonably optimized cells without over-driving
#   for speed at the cost of area/power.
#
# SPI SCLK: up to 10 MHz. It is oversampled as asynchronous data by clk;
# it is not a clock for any sequential element in this design.
# Scan clock: 25 MHz (40 ns), asynchronous to the functional clock.
# ============================================================

# -- Clocks ---------------------------------------------------

# Main system clock — synthesize to 50 MHz, operate at 2-10 MHz
create_clock -period 20 -name clk [get_ports clk]

# Scan chain clock for weight loading at power-up (async)
create_clock -period 40 -name scan_clk [get_ports scan_clk]

# The scan and functional domains are unrelated. The scan-domain
# weights_loaded_scan level enters clk through a two-flop synchronizer.
set_clock_groups -asynchronous \
    -group [get_clocks clk] \
    -group [get_clocks scan_clk]

# -- Synchronous input delays ---------------------------------
# ADC samples are assumed synchronous to clk at the chip boundary.
# Maximum delay checks setup; minimum delay checks hold.
set_input_delay -clock clk -max 5.0 [get_ports adc_sample]
set_input_delay -clock clk -min 0.5 [get_ports adc_sample]

# Scan controls and data are launched by the external scan controller and
# captured by scan_clk, so they must be constrained against scan_clk.
set_input_delay -clock scan_clk -max 5.0 [get_ports {scan_en scan_in}]
set_input_delay -clock scan_clk -min 0.5 [get_ports {scan_en scan_in}]

# -- Asynchronous inputs --------------------------------------
# spi_sclk and spi_cs_n are sampled by the first stage of explicit two-flop
# synchronizers in u_spi. There is no meaningful setup/hold relationship to
# clk at those first-stage D pins, so static timing must not analyze it.
# Give the ports explicit external-delay metadata so timing lint knows their
# boundary environment. The false paths below override ordinary setup/hold
# analysis at the asynchronous first-stage synchronizer inputs.
set_input_delay -clock clk -max 5.0 [get_ports {spi_sclk spi_cs_n}]
set_input_delay -clock clk -min 0.5 [get_ports {spi_sclk spi_cs_n}]

set_false_path -from [get_ports spi_sclk] \
               -to [get_pins u_spi/sclk_sync1_reg/D]
set_false_path -from [get_ports spi_cs_n] \
               -to [get_pins u_spi/cs_n_sync1_reg/D]

# Reset is asynchronously asserted. Its safe deassertion is an RDC design
# requirement rather than a normal data-path setup check. Its delay entries
# describe the port environment only; the false path prevents synchronous
# setup/hold analysis from assigning meaning to its phase relative to clk.
set_input_delay -clock clk -max 5.0 [get_ports rst_n]
set_input_delay -clock clk -min 0.5 [get_ports rst_n]
set_false_path -from [get_ports rst_n]

# -- Output delays --------------------------------------------
set_output_delay -clock clk -max 5.0 [get_ports {adc_channel spi_miso}]
set_output_delay -clock clk -min 0.5 [get_ports {adc_channel spi_miso}]

# -- External signal slew -------------------------------------
# Preliminary 200 ps boundary slew assumption. Replace this with characterized
# pad/board values before signoff. Clock slew uses the clock-specific command.
set_clock_transition 0.2 [get_clocks {clk scan_clk}]
set_input_transition 0.2 \
    [get_ports {rst_n adc_sample spi_sclk spi_cs_n scan_en scan_in}]

# -- Load / drive strength ------------------------------------
set_load           0.005 [all_outputs]
set_max_fanout     20    [current_design]
set_max_transition 1.0   [current_design]
