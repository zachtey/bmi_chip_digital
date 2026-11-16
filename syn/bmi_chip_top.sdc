# ============================================================
# bmi_chip_top.sdc  -  Timing constraints for Genus synthesis
#
# NOT YET RUN THROUGH GENUS OR INNOVUS. The instance names used in the
# comments below are the RTL names; check them against the elaborated design.
#
# Two unrelated clocks:
#   bmi_clk  Pipeline clock. Synthesis target 50 MHz (20 ns); it runs at
#            about 10 MHz in silicon.
#   pclk     APB clock. The real frequency is set by the SoC bus bridge and is
#            NOT known here. 100 MHz (10 ns) is a placeholder.
#
# Clock-domain crossings (all inside cdc_* instances in bmi_chip_top):
#   Synchronizer inputs (single bit or Gray pointer, flop to flop):
#     u_cmd_hs/req_q, u_weight_hs/req_q       -> u_*_hs/u_req_sync/meta_q
#     u_cmd_hs/ack_q, u_weight_hs/ack_q       -> u_*_hs/u_ack_sync/meta_q
#     u_result_fifo/wgray_q                   -> u_result_fifo/u_wptr_sync/meta_q
#     u_result_fifo/rgray_q                   -> u_result_fifo/u_rptr_sync/meta_q
#     u_ctrl/overflow_q                       -> u_ovf_sync/meta_q
#   Bundled data (sampled only while the handshake guarantees it is stable):
#     u_apb/cmd_code_q                        -> u_cmd_hs/dst_data_q
#     u_apb/weight_word[*] (864 bits)         -> u_weight_hs/dst_data_q
#     u_cmd_hs/dst_rsp_q                      -> u_cmd_hs/src_rsp_q
#     u_result_fifo/mem[*]                    -> u_apb/snap_q
#
# These paths are NOT declared false and the clocks are NOT put in
# asynchronous clock groups, because either would leave the crossings with no
# delay limit at all. What each crossing needs:
#   * a synchronizer input must settle well within one destination period;
#   * bundled data is sampled at least two destination periods after its
#     request/pointer was launched, so it is safe if both the data path and
#     the request path are each shorter than one destination period;
#   * the bits of a Gray pointer must have less than one source period of
#     skew between them, so that at most one bit is in flight when sampled.
# Limiting every clock-to-clock path to CDC_MAX_DELAY, the shorter of the two
# clock periods, satisfies all three. Hold is not meaningful between
# unrelated clocks and is disabled.
#
# This file does not prove the crossings are structurally correct; that needs
# a CDC tool. The synchronizer flops (module cdc_sync_2ff and
# reset_synchronizer) must also be kept out of retiming and placed adjacent.
# ============================================================

set BMI_CLK_PERIOD 20.0
set PCLK_PERIOD    10.0   ;# PLACEHOLDER: replace with the bridge's PCLK period
set CDC_MAX_DELAY  [expr {min($BMI_CLK_PERIOD, $PCLK_PERIOD)}]

# -- Clocks ---------------------------------------------------
create_clock -period $BMI_CLK_PERIOD -name bmi_clk [get_ports bmi_clk]
create_clock -period $PCLK_PERIOD    -name pclk    [get_ports pclk]

# -- Clock-domain crossings (see header) ----------------------
# If the tool supports it, add -ignore_clock_latency (Cadence) or
# -datapath_only so clock-tree insertion delay is not counted on these paths.
set_max_delay $CDC_MAX_DELAY -from [get_clocks pclk]    -to [get_clocks bmi_clk]
set_max_delay $CDC_MAX_DELAY -from [get_clocks bmi_clk] -to [get_clocks pclk]
set_false_path -hold -from [get_clocks pclk]    -to [get_clocks bmi_clk]
set_false_path -hold -from [get_clocks bmi_clk] -to [get_clocks pclk]

# -- Global reset ---------------------------------------------
# presetn is asynchronous. It only drives the asynchronous clear pins of the
# two reset synchronizers (u_prst_sync, u_bmi_rst_sync), which release the
# reset on a clock edge in each domain. Everything else is reset by their
# outputs, so recovery/removal there is an ordinary same-clock check.
set_false_path -from [get_ports presetn]

# -- ADC interface (bmi_clk) ----------------------------------
set_input_delay  -clock bmi_clk -max 5.0 [get_ports adc_sample]
set_input_delay  -clock bmi_clk -min 0.5 [get_ports adc_sample]
set_output_delay -clock bmi_clk -max 5.0 [get_ports adc_channel]
set_output_delay -clock bmi_clk -min 0.5 [get_ports adc_channel]

# -- APB slave interface (pclk) -------------------------------
# A budget of 30% of the placeholder period on each side; rescale it with the
# real PCLK period. prdata and pslverr are combinational functions of the APB
# inputs, so input-to-output paths exist: 10 - 3 - 3 = 4 ns through the block.
set_input_delay  -clock pclk -max 3.0 [get_ports {psel penable pwrite paddr pwdata}]
set_input_delay  -clock pclk -min 0.3 [get_ports {psel penable pwrite paddr pwdata}]
set_output_delay -clock pclk -max 3.0 [get_ports {prdata pready pslverr irq}]
set_output_delay -clock pclk -min 0.3 [get_ports {prdata pready pslverr irq}]

# -- External signal slew -------------------------------------
# Preliminary 200 ps boundary slew assumption. Replace with characterized
# values before signoff.
set_clock_transition 0.2 [get_clocks {bmi_clk pclk}]
set_input_transition 0.2 \
    [get_ports {presetn adc_sample psel penable pwrite paddr pwdata}]

# -- Load / drive strength ------------------------------------
set_load           0.005 [all_outputs]
set_max_fanout     20    [current_design]
set_max_transition 1.0   [current_design]
