// bmi_chip_top.sv
// Top-level structural wrapper for the BMI digital signal-processing chain.
//
// Pipeline:
//   ADC -> sample_collection -> sbp_feature_extraction
//       -> mlp_inference -> argmax -> output_formatter -> spi_slave

module bmi_chip_top #(
    parameter N_CH              = 8,
    parameter N_SAMPLES         = 250,
    parameter ADC_WIDTH         = 8,
    parameter SBP_WIDTH         = 8,
    parameter N_IN              = 8,
    parameter N_HIDDEN          = 8,
    parameter N_OUT             = 4,
    parameter W_WIDTH           = 8,
    parameter ACC_WIDTH         = 32,
    parameter SCORE_WIDTH       = 32,
    parameter HIDDEN_BIAS_SCALE = 85,
    parameter OUTPUT_BIAS_SCALE = 109,
    parameter PKT_BYTES         = 10
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ADC interface — adc_channel drives the analog MUX select
    input  wire [ADC_WIDTH-1:0]    adc_sample,
    output wire [$clog2(N_CH)-1:0] adc_channel,

    // SPI output
    input  wire                    spi_sclk,
    input  wire                    spi_cs_n,
    output wire                    spi_miso,

    // Scan chain for MLP weight loading
    input  wire                    scan_en,
    input  wire                    scan_clk,
    input  wire                    scan_in
);

    // -- Internal signals -----------------------------------------

    wire                          window_ready;
    wire [ADC_WIDTH-1:0]          sample_window [0:N_CH-1][0:N_SAMPLES-1];

    wire                          sbp_done;
    wire [SBP_WIDTH-1:0]          sbp_features [0:N_CH-1];

    wire                          mlp_done;
    wire signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1];

    wire [1:0]                    predicted_class;
    wire                          decision_valid;

    wire [PKT_BYTES*8-1:0]        packet_data;
    wire                          packet_valid;
    wire                          packet_ready;

    // -- Block instances ------------------------------------------

    sample_collection #(
        .N_CH(N_CH), .N_SAMPLES(N_SAMPLES), .ADC_WIDTH(ADC_WIDTH)
    ) u_sc (
        .clk(clk), .rst_n(rst_n),
        .adc_sample(adc_sample),
        .adc_channel(adc_channel),
        .resume(packet_ready),
        .window_ready(window_ready),
        .sample_window(sample_window)
    );

    sbp_feature_extraction #(
        .N_CH(N_CH), .N_SAMPLES(N_SAMPLES),
        .ADC_WIDTH(ADC_WIDTH), .SBP_WIDTH(SBP_WIDTH)
    ) u_sbp (
        .clk(clk), .rst_n(rst_n),
        .start(window_ready), .done(sbp_done),
        .sample_window(sample_window),
        .sbp_features(sbp_features)
    );

    mlp_inference #(
        .N_IN(N_IN), .N_HIDDEN(N_HIDDEN), .N_OUT(N_OUT),
        .IN_WIDTH(SBP_WIDTH), .W_WIDTH(W_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .SCORE_WIDTH(SCORE_WIDTH),
        .HIDDEN_BIAS_SCALE(HIDDEN_BIAS_SCALE),
        .OUTPUT_BIAS_SCALE(OUTPUT_BIAS_SCALE)
    ) u_mlp (
        .clk(clk), .rst_n(rst_n),
        .start(sbp_done), .done(mlp_done),
        .sbp_features(sbp_features),
        .class_scores(class_scores),
        .scan_en(scan_en), .scan_clk(scan_clk), .scan_in(scan_in)
    );

    argmax #(
        .N_CLASSES(N_OUT), .SCORE_WIDTH(SCORE_WIDTH)
    ) u_argmax (
        .clk(clk), .rst_n(rst_n),
        .class_scores(class_scores),
        .scores_valid(mlp_done),
        .predicted_class(predicted_class),
        .decision_valid(decision_valid)
    );

    output_formatter #(
        .N_CLASSES(N_OUT), .SCORE_WIDTH(SCORE_WIDTH), .PKT_BYTES(PKT_BYTES)
    ) u_fmt (
        .clk(clk), .rst_n(rst_n),
        .predicted_class(predicted_class),
        .class_scores(class_scores),
        .decision_valid(decision_valid),
        .packet_data(packet_data),
        .packet_valid(packet_valid),
        .packet_ready(packet_ready)
    );

    spi_slave #(
        .PKT_BYTES(PKT_BYTES)
    ) u_spi (
        .clk(clk), .rst_n(rst_n),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n), .spi_miso(spi_miso),
        .packet_data(packet_data),
        .packet_valid(packet_valid),
        .packet_ready(packet_ready)
    );

endmodule
