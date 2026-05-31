# ============================================================
# bmi_chip_top.sdc  -  Timing constraints for Genus synthesis
#
# Synthesis target:  50 MHz (20 ns period)
#   Actual operating frequency will be 2-10 MHz in silicon,
#   giving 5-25x margin.  Synthesizing to 50 MHz ensures
#   Genus picks reasonably optimized cells without over-driving
#   for speed at the cost of area/power.
#
# SPI clock:   10 MHz (100 ns) - asynchronous to sys clk
# Scan clock:  25 MHz ( 40 ns) - asynchronous to sys clk
# ============================================================

# -- Clocks ---------------------------------------------------

# Main system clock — synthesize to 50 MHz, operate at 2-10 MHz
create_clock -period 20 -name clk [get_ports clk]

# SPI SCLK from external master (async to system clk)
create_clock -period 100 -name spi_sclk [get_ports spi_sclk]

# Scan chain clock for weight loading at power-up (async)
create_clock -period 40 -name scan_clk [get_ports scan_clk]

# Mark all three clocks as asynchronous to each other
set_clock_groups -asynchronous \
    -group [get_clocks clk] \
    -group [get_clocks spi_sclk] \
    -group [get_clocks scan_clk]

# -- Input delays (relative to system clock) ------------------
# 5 ns budget at 20 ns period (25%)
set_input_delay -clock clk -max 5.0 [get_ports {rst_n adc_sample}]
set_input_delay -clock clk -max 5.0 [get_ports {scan_en scan_in}]
set_input_delay -clock clk -max 5.0 [get_ports {spi_cs_n}]

# -- Output delays --------------------------------------------
set_output_delay -clock clk    -max 5.0 [get_ports {adc_channel}]
set_output_delay -clock spi_sclk -max 5.0 [get_ports {spi_miso}]

# -- Load / drive strength ------------------------------------
set_load           0.005 [all_outputs]
set_max_fanout     20    [current_design]
set_max_transition 1.0   [current_design]
