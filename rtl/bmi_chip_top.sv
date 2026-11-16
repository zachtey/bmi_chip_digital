// bmi_chip_top.sv
// Top level of the BMI classifier.
//
//   ADC -> streaming_sbp_frontend -> mlp_inference -> argmax -> result FIFO
//                                                                   |
//   APB3 host <-> apb_slave <-- snapshot ---------------------------+
//
// Two unrelated clocks, one reset:
//   pclk     APB clock. Clocks apb_slave: registers, staged weights, result
//            snapshot and the interrupt.
//   bmi_clk  Pipeline clock (CLK_FREQ_HZ, nominally 10 MHz). Clocks the
//            frontend, MLP, argmax, the active weights and bmi_pipeline_ctrl.
//   presetn  Global asynchronous reset, active low. Asserted asynchronously
//            and released synchronously, separately in each domain
//            (u_prst_sync, u_bmi_rst_sync). APB transfers must not start
//            until two pclk edges after presetn is released.
//
// Every crossing is inside a cdc_* instance; no other module has a
// synchronizer. Signals carry a _p (pclk) or _b (bmi_clk) suffix where the
// domain is not obvious.
//
//   u_cmd_hs       pclk -> bmi_clk  START/STOP/RESTART mailbox. The
//                                   acknowledgment carries RUNNING back.
//   u_weight_hs    pclk -> bmi_clk  Weight commit. Its destination register
//                                   is the active weight image the MLP uses;
//                                   all 864 bits are captured on one edge.
//   u_result_fifo  bmi_clk -> pclk  Depth-4 FIFO of {scores, class}.
//   u_ovf_sync     bmi_clk -> pclk  Sticky result-overflow level.
//

module bmi_chip_top #(
    parameter N_CH              = 8,
    parameter N_SAMPLES         = 250,
    parameter ADC_WIDTH         = 8,
    parameter CLK_FREQ_HZ        = 10_000_000, // frequency of bmi_clk
    parameter ADC_SAMPLE_FREQ_HZ = 2_500_000,  // max aggregate ADC capture rate
    parameter SBP_WIDTH         = 8,
    parameter N_IN              = 8,
    parameter N_HIDDEN          = 8,
    parameter N_OUT             = 4,
    parameter W_WIDTH           = 8,
    parameter ACC_WIDTH         = 32,
    parameter SCORE_WIDTH       = 32,
    parameter HIDDEN_BIAS_SCALE = 85,
    parameter OUTPUT_BIAS_SCALE = 109,
    parameter APB_ADDR_WIDTH    = 12           // width of paddr (byte address)
)(
    input  wire                    bmi_clk,    // pipeline clock
    input  wire                    pclk,       // APB clock, unrelated to bmi_clk
    input  wire                    presetn,    // global async reset, active low

    // ANALOG pins: ADC interface (bmi_clk). adc_channel drives the analog MUX select.
    input  wire [ADC_WIDTH-1:0]    adc_sample,
    output wire [$clog2(N_CH)-1:0] adc_channel,

    // APB3 slave interface (pclk)
    input  wire                      psel,
    input  wire                      penable,
    input  wire                      pwrite,
    input  wire [APB_ADDR_WIDTH-1:0] paddr,
    input  wire [31:0]               pwdata,
    output wire [31:0]               prdata,
    output wire                      pready,
    output wire                      pslverr,

    // Level interrupt (pclk): RESULT_VALID & IRQ_EN
    output wire                      irq
);

    // Packed weight image: hw, hb, ow, ob (byte i at bits [i*8 +: 8]).
    localparam integer TOTAL_WEIGHT_BITS = (N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_HIDDEN + N_OUT) * W_WIDTH;
    localparam integer RESULT_WIDTH = N_OUT*SCORE_WIDTH + 2;

    // ---- resets ----
    wire prst_n;      // pclk domain
    wire bmi_rst_n;   // bmi_clk domain

    reset_synchronizer u_prst_sync (
        .clk(pclk), .arst_n(presetn), .rst_n_out(prst_n)
    );

    reset_synchronizer u_bmi_rst_sync (
        .clk(bmi_clk), .arst_n(presetn), .rst_n_out(bmi_rst_n)
    );

    // ---- pclk side ----
    wire                          cmd_send_p, cmd_busy_p, cmd_running_p;
    wire [1:0]                    cmd_code_p;
    wire                          wt_send_p, wt_busy_p, wt_done_p;
    wire [TOTAL_WEIGHT_BITS-1:0]  staged_weights_p;
    wire                          fifo_empty_p, fifo_pop_p;
    wire [RESULT_WIDTH-1:0]       fifo_rdata_p;
    wire                          overflow_p;

    // ---- bmi_clk side ----
    wire                          cmd_valid_b, cmd_done_b, cmd_running_b;
    wire [1:0]                    cmd_code_b;
    wire                          wt_valid_b, wt_done_b, weights_valid_b;
    wire [TOTAL_WEIGHT_BITS-1:0]  active_weights_b;
    wire                          fifo_push_b, fifo_full_b;
    wire [RESULT_WIDTH-1:0]       fifo_wdata_b;
    wire                          overflow_b;

    wire                          frontend_enable, pipe_clear;
    wire                          sbp_done;
    wire [SBP_WIDTH-1:0]          sbp_features [0:N_CH-1];
    wire                          mlp_done;
    wire signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1];
    wire [1:0]                    predicted_class;
    wire                          decision_valid;

    // ---- host interface (pclk) ----
    apb_slave #(
        .N_OUT(N_OUT), .SCORE_WIDTH(SCORE_WIDTH),
        .WEIGHT_BITS(TOTAL_WEIGHT_BITS), .ADDR_WIDTH(APB_ADDR_WIDTH)
    ) u_apb (
        .clk(pclk), .rst_n(prst_n),
        .psel(psel), .penable(penable), .pwrite(pwrite),
        .paddr(paddr), .pwdata(pwdata),
        .prdata(prdata), .pready(pready), .pslverr(pslverr),
        .irq(irq),
        .cmd_send(cmd_send_p), .cmd_code(cmd_code_p),
        .cmd_busy(cmd_busy_p), .cmd_running(cmd_running_p),
        .wt_send(wt_send_p), .weight_image(staged_weights_p),
        .wt_busy(wt_busy_p), .wt_done(wt_done_p),
        .fifo_empty(fifo_empty_p), .fifo_rdata(fifo_rdata_p),
        .fifo_pop(fifo_pop_p),
        .overflow(overflow_p)
    );

    // ---- crossings ----
    wire unused_cmd_done_p;

    cdc_handshake #(.REQ_WIDTH(2), .RSP_WIDTH(1)) u_cmd_hs (
        .src_clk(pclk), .src_rst_n(prst_n),
        .src_send(cmd_send_p), .src_data(cmd_code_p),
        .src_busy(cmd_busy_p), .src_done(unused_cmd_done_p),
        .src_rsp(cmd_running_p),
        .dst_clk(bmi_clk), .dst_rst_n(bmi_rst_n),
        .dst_valid(cmd_valid_b), .dst_data(cmd_code_b),
        .dst_done(cmd_done_b), .dst_rsp(cmd_running_b)
    );

    wire unused_wt_rsp_p;

    cdc_handshake #(.REQ_WIDTH(TOTAL_WEIGHT_BITS), .RSP_WIDTH(1)) u_weight_hs (
        .src_clk(pclk), .src_rst_n(prst_n),
        .src_send(wt_send_p), .src_data(staged_weights_p),
        .src_busy(wt_busy_p), .src_done(wt_done_p),
        .src_rsp(unused_wt_rsp_p),
        .dst_clk(bmi_clk), .dst_rst_n(bmi_rst_n),
        .dst_valid(wt_valid_b), .dst_data(active_weights_b),
        .dst_done(wt_done_b), .dst_rsp(1'b0)
    );

    cdc_async_fifo #(.WIDTH(RESULT_WIDTH), .ADDR_WIDTH(2)) u_result_fifo (
        .wclk(bmi_clk), .wrst_n(bmi_rst_n),
        .wr_en(fifo_push_b), .wdata(fifo_wdata_b), .wfull(fifo_full_b),
        .rclk(pclk), .rrst_n(prst_n),
        .rd_en(fifo_pop_p), .rdata(fifo_rdata_p), .rempty(fifo_empty_p)
    );

    // Sticky: set once, cleared only by the global reset.
    cdc_sync_2ff u_ovf_sync (
        .clk(pclk), .rst_n(prst_n), .d(overflow_b), .q(overflow_p)
    );

    // ---- pipeline (bmi_clk) ----
    bmi_pipeline_ctrl #(
        .N_OUT(N_OUT), .SCORE_WIDTH(SCORE_WIDTH)
    ) u_ctrl (
        .clk(bmi_clk), .rst_n(bmi_rst_n),
        .cmd_valid(cmd_valid_b), .cmd_code(cmd_code_b),
        .cmd_done(cmd_done_b), .cmd_running(cmd_running_b),
        .wt_valid(wt_valid_b), .wt_done(wt_done_b),
        .weights_valid(weights_valid_b),
        .frontend_enable(frontend_enable), .pipe_clear(pipe_clear),
        .features_done(sbp_done), .decision_valid(decision_valid),
        .predicted_class(predicted_class), .class_scores(class_scores),
        .fifo_push(fifo_push_b), .fifo_wdata(fifo_wdata_b),
        .fifo_full(fifo_full_b),
        .overflow(overflow_b)
    );

    // The frontend pauses after each window while the MLP reads its
    // features, and is resumed by the decision itself, so it never waits for
    // software.
    streaming_sbp_frontend #(
        .N_CH(N_CH), .N_SAMPLES(N_SAMPLES),
        .ADC_WIDTH(ADC_WIDTH), .SBP_WIDTH(SBP_WIDTH),
        .CLK_FREQ_HZ(CLK_FREQ_HZ), .ADC_SAMPLE_FREQ_HZ(ADC_SAMPLE_FREQ_HZ)
    ) u_frontend (
        .clk(bmi_clk), .rst_n(bmi_rst_n),
        .enable(frontend_enable), .clear(pipe_clear),
        .adc_sample(adc_sample),
        .adc_channel(adc_channel),
        .resume(decision_valid),
        .features_done(sbp_done),
        .sbp_features(sbp_features)
    );

    mlp_inference #(
        .N_IN(N_IN), .N_HIDDEN(N_HIDDEN), .N_OUT(N_OUT),
        .IN_WIDTH(SBP_WIDTH), .W_WIDTH(W_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .SCORE_WIDTH(SCORE_WIDTH),
        .HIDDEN_BIAS_SCALE(HIDDEN_BIAS_SCALE),
        .OUTPUT_BIAS_SCALE(OUTPUT_BIAS_SCALE)
    ) u_mlp (
        .clk(bmi_clk), .rst_n(bmi_rst_n), .clear(pipe_clear),
        .start(sbp_done), .done(mlp_done),
        .weight_image(active_weights_b),
        .weights_loaded(weights_valid_b),
        .sbp_features(sbp_features),
        .class_scores(class_scores)
    );

    argmax #(
        .N_CLASSES(N_OUT), .SCORE_WIDTH(SCORE_WIDTH)
    ) u_argmax (
        .clk(bmi_clk), .rst_n(bmi_rst_n), .clear(pipe_clear),
        .class_scores(class_scores),
        .scores_valid(mlp_done),
        .predicted_class(predicted_class),
        .decision_valid(decision_valid)
    );

    wire unused_ok = &{1'b0, unused_cmd_done_p, unused_wt_rsp_p};

endmodule
