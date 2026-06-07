// sbp_feature_extraction.sv
// Computes Spiking Band Power (SBP) for each of N_CH channels.
// SBP[ch] = sum(|sample - 128|) >> 8  — mean absolute deviation proxy.
//
// Single shared accumulator, channels processed sequentially.
// Latency: N_CH * N_SAMPLES = 2000 clock cycles.
// No multipliers; absolute value is a subtraction mux.

module sbp_feature_extraction #(
    parameter N_CH      = 8,
    parameter N_SAMPLES = 250,
    parameter ADC_WIDTH = 8,
    parameter SBP_WIDTH = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         start,
    output logic                        done,

    input  wire [ADC_WIDTH-1:0]         sample_window [0:N_CH-1][0:N_SAMPLES-1],
    output logic [SBP_WIDTH-1:0]        sbp_features  [0:N_CH-1]
);

    // Accumulator headroom: max |x-128| = 128; 128*250 = 32000 < 2^15. 16 bits.
    localparam ACC_WIDTH = 16;

    typedef enum logic { IDLE, RUN } state_t;

    state_t                          state,      next_state;
    logic [ACC_WIDTH-1:0]            acc;
    logic [$clog2(N_CH)-1:0]         ch_idx;
    logic [$clog2(N_SAMPLES)-1:0]    samp_idx;

    // Next-cycle values
    logic [ACC_WIDTH-1:0]            next_acc;
    logic [$clog2(N_CH)-1:0]         next_ch_idx;
    logic [$clog2(N_SAMPLES)-1:0]    next_samp_idx;
    logic [SBP_WIDTH-1:0]            next_sbp_features [0:N_CH-1];
    logic                             next_done;

    // -------------------------------------------------------------------------
    // Combinational: absolute deviation and next-state logic
    // -------------------------------------------------------------------------
    logic [ADC_WIDTH-1:0] current_sample;
    logic [ADC_WIDTH-1:0] abs_dev;
    logic [ACC_WIDTH-1:0] acc_with_dev;

    always_comb begin
        current_sample = sample_window[ch_idx][samp_idx];
        abs_dev        = (current_sample >= 8'd128) ? (current_sample - 8'd128)
                                                     : (8'd128 - current_sample);
        acc_with_dev   = acc + {{(ACC_WIDTH-ADC_WIDTH){1'b0}}, abs_dev};

        // Defaults — hold all state
        next_state    = state;
        next_acc      = acc;
        next_ch_idx   = ch_idx;
        next_samp_idx = samp_idx;
        next_done     = 1'b0;
        for (int i = 0; i < N_CH; i++)
            next_sbp_features[i] = sbp_features[i];

        unique case (state)
            IDLE: begin
                if (start) begin
                    next_acc      = '0;
                    next_ch_idx   = '0;
                    next_samp_idx = '0;
                    next_state    = RUN;
                end
            end

            RUN: begin
                next_acc = acc_with_dev;

                if (samp_idx == N_SAMPLES - 1) begin
                    // Capture final accumulated value for this channel
                    next_sbp_features[ch_idx] = acc_with_dev[ACC_WIDTH-1:8];
                    next_acc      = '0;
                    next_samp_idx = '0;

                    if (ch_idx == N_CH - 1) begin
                        next_done   = 1'b1;
                        next_ch_idx = '0;
                        next_state  = IDLE;
                    end else begin
                        next_ch_idx = ch_idx + 1;
                    end
                end else begin
                    next_samp_idx = samp_idx + 1;
                end
            end

            default: next_state = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential: registers
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            done     <= 1'b0;
            acc      <= '0;
            ch_idx   <= '0;
            samp_idx <= '0;
            for (int i = 0; i < N_CH; i++)
                sbp_features[i] <= '0;
        end else begin
            state    <= next_state;
            done     <= next_done;
            acc      <= next_acc;
            ch_idx   <= next_ch_idx;
            samp_idx <= next_samp_idx;
            for (int i = 0; i < N_CH; i++)
                sbp_features[i] <= next_sbp_features[i];
        end
    end

endmodule
