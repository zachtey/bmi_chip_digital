// streaming_sbp_frontend.sv
//
// Streaming replacement for sample_collection + sbp_feature_extraction.
// A single time-multiplexed ADC supplies one sample every system clock. The
// module rotates adc_channel through all channels and accumulates
// abs(adc_sample - midpoint) immediately, so no sample window is stored or
// reread. After N_SAMPLES have been accepted for every channel, features_done
// pulses for one clock and collection pauses until resume is asserted.
//
// SBP[ch] = sum(abs(sample - 2^(ADC_WIDTH-1))) >> FEATURE_SHIFT

module streaming_sbp_frontend #(
    parameter integer N_CH          = 8,
    parameter integer N_SAMPLES     = 250,
    parameter integer ADC_WIDTH     = 8,
    parameter integer SBP_WIDTH     = 8,
    parameter integer FEATURE_SHIFT = 8,
    parameter integer CH_IDX_WIDTH  = (N_CH > 1) ? $clog2(N_CH) : 1
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire [ADC_WIDTH-1:0]         adc_sample,
    output logic [CH_IDX_WIDTH-1:0]     adc_channel,

    // Collection remains paused after features_done until the completed
    // output packet has been consumed and resume is pulsed.
    input  wire                         resume,

    output logic                        features_done,
    output logic [SBP_WIDTH-1:0]        sbp_features [0:N_CH-1]
);

    localparam integer SAMPLE_CNT_WIDTH = (N_SAMPLES > 1) ? $clog2(N_SAMPLES) : 1;
    localparam integer ACC_WIDTH = $clog2(N_SAMPLES * (1 << (ADC_WIDTH-1)) + 1); //max accumulation value is half the adc * number of samples
    localparam logic [ADC_WIDTH-1:0] ADC_MIDPOINT = {1'b1, {(ADC_WIDTH-1){1'b0}}};

    typedef enum logic {COLLECT, PAUSE} state_t;
    state_t state;

    logic [SAMPLE_CNT_WIDTH-1:0] sample_cnt [0:N_CH-1];
    logic [ACC_WIDTH-1:0] deviation_sum [0:N_CH-1];
    logic [ADC_WIDTH-1:0] abs_deviation;
    logic [ACC_WIDTH-1:0] sum_with_sample;

    always_comb begin
        if (adc_sample >= ADC_MIDPOINT)
            abs_deviation = adc_sample - ADC_MIDPOINT;
        else
            abs_deviation = ADC_MIDPOINT - adc_sample;
        // Include the currently presented sample when producing the final feature. Using deviation_sum alone here would drop sample 249.
        sum_with_sample = deviation_sum[adc_channel] + {{(ACC_WIDTH-ADC_WIDTH){1'b0}}, abs_deviation};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= COLLECT;
            adc_channel   <= '0;
            features_done <= 1'b0;
            for (int ch = 0; ch < N_CH; ch++) begin
                sample_cnt[ch]    <= '0;
                deviation_sum[ch] <= '0;
                sbp_features[ch]  <= '0;
            end
        end else begin
            features_done <= 1'b0;

            unique case (state)
                COLLECT: begin
                    deviation_sum[adc_channel] <= sum_with_sample;
                    if (sample_cnt[adc_channel] == N_SAMPLES-1) begin
                        sbp_features[adc_channel] <= sum_with_sample >> FEATURE_SHIFT;
                        if (adc_channel == N_CH-1) begin
                            features_done <= 1'b1;
                            state         <= PAUSE;
                        end
                    end else begin
                        sample_cnt[adc_channel] <= sample_cnt[adc_channel] + 1'b1;
                    end
                    if (adc_channel == N_CH-1)
                        adc_channel <= '0;
                    else
                        adc_channel <= adc_channel + 1'b1;
                end

                PAUSE: begin
                    if (resume) begin
                        state       <= COLLECT;
                        adc_channel <= '0;
                        for (int ch = 0; ch < N_CH; ch++) begin
                            sample_cnt[ch]    <= '0;
                            deviation_sum[ch] <= '0;
                        end
                    end
                end

                default: begin
                    state       <= COLLECT;
                    adc_channel <= '0;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (N_CH < 1 || N_SAMPLES < 1)
            $fatal(1, "N_CH and N_SAMPLES must be at least one");
        if (ADC_WIDTH < 2 || SBP_WIDTH < 1)
            $fatal(1, "Invalid frontend datapath width");
        if (FEATURE_SHIFT < 0 || FEATURE_SHIFT >= ACC_WIDTH)
            $fatal(1, "FEATURE_SHIFT must select bits within the accumulator");
        if ((ACC_WIDTH-FEATURE_SHIFT) > SBP_WIDTH)
            $fatal(1, "SBP_WIDTH cannot hold the maximum shifted sum");
    end
`endif

endmodule
