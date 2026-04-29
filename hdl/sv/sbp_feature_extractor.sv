// sbp_feature_extraction.v
//
// Computes SBP (Spiking Band Power) for 8 channels.
// SBP[ch] = sum(|sample - 128|) >> 8   (≈ mean absolute deviation)
//
// Uses ONE shared accumulator and processes channels sequentially.
// Latency: 8 channels × 250 samples = 2000 clock cycles.
// At 1 MHz: 2 ms — well within the 50 ms budget.
//
// Area: 1 accumulator (16-bit), 2 counters, small FSM.
// No multipliers. No SRAM. Pure combinational + flip-flops.
module sbp_feature_extraction #(
    parameter N_CH      = 8,
    parameter N_SAMPLES = 250,
    parameter ADC_WIDTH = 8,
    parameter SBP_WIDTH = 8     // output feature width (sum >> 8 fits in 8 bits)
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // Handshake
    input  wire                              start,         // 1-cycle pulse from sample_collection
    output reg                               done,          // 1-cycle pulse when all 8 features ready

    // Input: the full sample window from sample_collection
    // 8 channels × 250 samples × 8 bits
    input  wire [ADC_WIDTH-1:0]              sample_window [0:N_CH-1][0:N_SAMPLES-1],

    // Output: one SBP value per channel
    output reg  [SBP_WIDTH-1:0]              sbp_features [0:N_CH-1]
);

    // Accumulator 
    // Max value: 128 (max |x - 128|) × 250 samples = 32,000
    // 15 bits needed (2^15 = 32768 > 32000). Use 16 for safety.
    localparam ACC_WIDTH = 16;
    reg [ACC_WIDTH-1:0] acc;

    // Counters 
    reg [$clog2(N_CH)-1:0]      ch_idx;      // current channel being accumulated (0-7)
    reg [$clog2(N_SAMPLES)-1:0] samp_idx;    // current sample within that channel (0-249)

    //  Absolute deviation 
    // |sample - 128| without subtraction overflow:
    //   if sample >= 128: deviation = sample - 128
    //   if sample <  128: deviation = 128 - sample
    // This is a single mux, zero hardware cost.
    wire [ADC_WIDTH-1:0] current_sample;
    wire [ADC_WIDTH-1:0] abs_dev;

    assign current_sample = sample_window[ch_idx][samp_idx];
    assign abs_dev = (current_sample >= 8'd128)
                   ? (current_sample - 8'd128)
                   : (8'd128 - current_sample);

    // FSM 
    // IDLE: waiting for start pulse
    // RUN:  accumulating samples, one per clock
    // DONE: features are valid, pulse done for one cycle
    //
    // Timing:
    //   cycle 0:        start pulse arrives, enter RUN
    //   cycles 1-250:   accumulate channel 0 (samples 0-249)
    //   cycle 251:      store sbp[0], reset acc, move to channel 1
    //   cycles 252-501: accumulate channel 1
    //   ...
    //   cycle 2001:     store sbp[7], pulse done, back to IDLE

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    reg [1:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= IDLE;
            done     <= 1'b0;
            acc      <= '0;
            ch_idx   <= '0;
            samp_idx <= '0;
            for (int i = 0; i < N_CH; i++)
                sbp_features[i] <= '0;

        end else begin
            done <= 1'b0;    // default: deassert (done is a 1-cycle pulse)

            case (state)

                IDLE: begin
                    if (start) begin
                        acc      <= '0;
                        ch_idx   <= '0;
                        samp_idx <= '0;
                        state    <= RUN;
                    end
                end

                RUN: begin
                    // ── Accumulate one sample per clock ──────
                    acc <= acc + {{(ACC_WIDTH-ADC_WIDTH){1'b0}}, abs_dev};

                    if (samp_idx == N_SAMPLES - 1) begin
                        // ── Last sample of this channel ───────
                        // Store the result: shift right by 8 = divide by 256
                        // This matches the hardware-friendly golden model.
                        // Max possible value: (acc + 128) >> 8 = 32128 >> 8 = 125
                        // which fits comfortably in SBP_WIDTH = 8 bits.
                        sbp_features[ch_idx] <= (acc + {{(ACC_WIDTH-ADC_WIDTH){1'b0}}, abs_dev}) >> 8;

                        samp_idx <= '0;
                        acc      <= '0;

                        if (ch_idx == N_CH - 1) begin
                            // ── All channels done ─────────────
                            done     <= 1'b1;
                            ch_idx   <= '0;
                            state    <= IDLE;
                        end else begin
                            ch_idx <= ch_idx + 1;
                        end

                    end else begin
                        samp_idx <= samp_idx + 1;
                    end
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule