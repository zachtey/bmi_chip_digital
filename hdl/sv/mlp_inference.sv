// mlp_inference.v
//
// 8→8(ReLU)→4 MLP classifier.
// One shared MAC unit, time-multiplexed across all neurons.
//
// Latency: ~110 clock cycles from start to done.
// Memory:  108 bytes of weight registers (no SRAM).
//
// Weight loading: shift all 108 bytes in via scan chain
//   before inference begins. See scan chain section below.
//
// Bias scaling: hidden_bias_scale and output_bias_scale are
//   constants derived from the Python training pipeline.
//   Hard-code them as parameters — they are NOT weights,
//   they are fixed architectural constants that account for
//   the quantization scale mismatch between weights and biases.

module mlp_inference #(
    parameter N_IN     = 8,     // SBP input features
    parameter N_HIDDEN = 8,     // hidden neurons
    parameter N_OUT    = 4,     // output classes
    parameter IN_WIDTH = 8,     // SBP feature width (unsigned)
    parameter W_WIDTH  = 8,     // weight width (signed)
    parameter ACC_WIDTH = 32,   // accumulator width (signed)
    parameter SCORE_WIDTH = 32, // output score width (signed)

    // ── From Python pipeline output ───────────────────────────
    // These come from your weights.hex header:
    //   // hidden_bias_scale = <N>
    //   // output_bias_scale = <N>
    // Fill them in after running bmi_final.py
    parameter HIDDEN_BIAS_SCALE = 228,
    parameter OUTPUT_BIAS_SCALE = 193
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // Handshake
    input  wire                              start,           // 1-cycle pulse from SBP block
    output reg                               done,            // 1-cycle pulse when scores valid

    // Input features from SBP block
    input  wire [IN_WIDTH-1:0]               sbp_features [0:N_IN-1],

    // Output scores (one per class, signed 32-bit)
    output reg  signed [SCORE_WIDTH-1:0]     class_scores [0:N_OUT-1],

    // ── Scan chain for weight loading ─────────────────────────
    // Shift 108 bytes in LSB-first before asserting start.
    // Sequence: hw[0][0..7], hw[1][0..7], ..., hw[7][0..7],
    //           hb[0..7], ow[0][0..7], ..., ow[3][0..7], ob[0..3]
    input  wire                              scan_en,         // enable scan shift
    input  wire                              scan_clk,        // scan clock (separate from sys clk)
    input  wire                              scan_in          // serial scan data in
);

    // ════════════════════════════════════════════════════════
    // WEIGHT REGISTERS
    // ════════════════════════════════════════════════════════
    // All signed 8-bit. Loaded once via scan chain at power-up.
    // After loading, read-only during inference.

    reg signed [W_WIDTH-1:0] hw [0:N_HIDDEN-1][0:N_IN-1];    // hidden weights  [neuron][input]
    reg signed [W_WIDTH-1:0] hb [0:N_HIDDEN-1];               // hidden biases   [neuron]
    reg signed [W_WIDTH-1:0] ow [0:N_OUT-1][0:N_IN-1];        // output weights  [class][hidden]
    reg signed [W_WIDTH-1:0] ob [0:N_OUT-1];                  // output biases   [class]

    // ── Scan chain ────────────────────────────────────────────
    // 108 bytes = 864 bits total. One big shift register.
    // Layout matches weights.hex: hw[0..7][0..7], hb[0..7],
    //                             ow[0..3][0..7], ob[0..3]
    // Each byte shifts in LSB-first, one bit per scan_clk.
    localparam TOTAL_WEIGHT_BITS = (N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_IN + N_OUT) * W_WIDTH;

    reg [TOTAL_WEIGHT_BITS-1:0] scan_reg;

    always @(posedge scan_clk) begin
        if (scan_en)
            scan_reg <= {scan_in, scan_reg[TOTAL_WEIGHT_BITS-1:1]};
    end

    // Unpack scan_reg into weight registers (combinational)
    // This is a generate loop — zero hardware cost, just wiring.
    genvar gi, gj;
    generate
        for (gi = 0; gi < N_HIDDEN; gi++) begin : unpack_hw
            for (gj = 0; gj < N_IN; gj++) begin : unpack_hw_in
                assign hw[gi][gj] = scan_reg[(gi*N_IN + gj)*W_WIDTH +: W_WIDTH];
            end
        end
        for (gi = 0; gi < N_HIDDEN; gi++) begin : unpack_hb
            assign hb[gi] = scan_reg[(N_HIDDEN*N_IN + gi)*W_WIDTH +: W_WIDTH];
        end
        for (gi = 0; gi < N_OUT; gi++) begin : unpack_ow
            for (gj = 0; gj < N_IN; gj++) begin : unpack_ow_in
                assign ow[gi][gj] = scan_reg[((N_HIDDEN*N_IN + N_HIDDEN) + gi*N_IN + gj)*W_WIDTH +: W_WIDTH];
            end
        end
        for (gi = 0; gi < N_OUT; gi++) begin : unpack_ob
            assign ob[gi] = scan_reg[((N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_IN) + gi)*W_WIDTH +: W_WIDTH];
        end
    endgenerate

    // ════════════════════════════════════════════════════════
    // INTERNAL REGISTERS
    // ════════════════════════════════════════════════════════

    // Accumulator for current neuron's dot product
    reg signed [ACC_WIDTH-1:0] acc;

    // Hidden layer activations (after ReLU), stored as 32-bit
    // so output layer has enough dynamic range to work with
    reg signed [ACC_WIDTH-1:0] hidden_act [0:N_HIDDEN-1];

    // Neuron and weight counters
    reg [$clog2(N_HIDDEN)-1:0]  neuron_idx;   // which neuron we're computing
    reg [$clog2(N_IN)-1:0]      weight_idx;   // which weight within that neuron

    // ════════════════════════════════════════════════════════
    // FSM
    // ════════════════════════════════════════════════════════
    //
    //  IDLE
    //    │  start=1
    //    ▼
    //  H_MAC ──────────────────────────────────────────────┐
    //    │  weight_idx == N_IN-1                           │
    //    ▼                                                 │
    //  H_BIAS  (add hb[neuron] × HIDDEN_BIAS_SCALE)       │
    //    │                                                 │ neuron_idx < N_HIDDEN-1
    //    ▼                                                 │ (next neuron)
    //  H_RELU  (if acc<0: acc=0, store to hidden_act)     │
    //    │  neuron_idx == N_HIDDEN-1  ─────────────────────┘
    //    │  (all hidden neurons done)
    //    ▼
    //  O_MAC ──────────────────────────────────────────────┐
    //    │  weight_idx == N_HIDDEN-1                       │
    //    ▼                                                 │ neuron_idx < N_OUT-1
    //  O_BIAS  (add ob[neuron] × OUTPUT_BIAS_SCALE)       │
    //    │                                                 │
    //  O_STORE (store score, advance neuron) ─────────────┘
    //    │  neuron_idx == N_OUT-1
    //    ▼
    //  DONE (pulse done=1, back to IDLE)

    localparam [2:0]
        S_IDLE    = 3'd0,
        S_H_MAC   = 3'd1,
        S_H_BIAS  = 3'd2,
        S_H_RELU  = 3'd3,
        S_O_MAC   = 3'd4,
        S_O_BIAS  = 3'd5,
        S_O_STORE = 3'd6,
        S_DONE    = 3'd7;

    reg [2:0] state;

    // ── MAC inputs (registered for timing) ────────────────────
    // Sign-extend the unsigned SBP input to ACC_WIDTH before multiply
    wire signed [ACC_WIDTH-1:0] sbp_signed [0:N_IN-1];
    genvar gk;
    generate
        for (gk = 0; gk < N_IN; gk++) begin : sign_extend_sbp
            assign sbp_signed[gk] = {{(ACC_WIDTH-IN_WIDTH){1'b0}}, sbp_features[gk]};
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            acc        <= '0;
            neuron_idx <= '0;
            weight_idx <= '0;
            for (int i = 0; i < N_HIDDEN; i++) hidden_act[i]    <= '0;
            for (int i = 0; i < N_OUT;    i++) class_scores[i]  <= '0;

        end else begin
            done <= 1'b0;    // default deassert

            case (state)

                // ── Wait for start signal ────────────────────
                S_IDLE: begin
                    if (start) begin
                        acc        <= '0;
                        neuron_idx <= '0;
                        weight_idx <= '0;
                        state      <= S_H_MAC;
                    end
                end

                // ── Hidden layer: accumulate dot product ─────
                // Each cycle: acc += sbp[weight_idx] * hw[neuron_idx][weight_idx]
                // Processes one weight per clock. Takes N_IN=8 cycles per neuron.
                S_H_MAC: begin
                    acc <= acc + sbp_signed[weight_idx] *
                                 {{(ACC_WIDTH-W_WIDTH){hw[neuron_idx][weight_idx][W_WIDTH-1]}},
                                   hw[neuron_idx][weight_idx]};

                    if (weight_idx == N_IN - 1) begin
                        weight_idx <= '0;
                        state      <= S_H_BIAS;
                    end else begin
                        weight_idx <= weight_idx + 1;
                    end
                end

                // ── Add hidden bias (scaled) ──────────────────
                // acc += hb[neuron] * HIDDEN_BIAS_SCALE
                // HIDDEN_BIAS_SCALE is a constant — synthesizer
                // turns this into a series of shifts and adds.
                S_H_BIAS: begin
                    acc   <= acc + ({{(ACC_WIDTH-W_WIDTH){hb[neuron_idx][W_WIDTH-1]}},
                                      hb[neuron_idx]} *
                                    $signed({{(ACC_WIDTH-16){1'b0}},
                                             HIDDEN_BIAS_SCALE[15:0]}));
                    state <= S_H_RELU;
                end

                // ── ReLU + store hidden activation ────────────
                // if acc < 0: store 0 (ReLU clips negatives)
                // if acc >= 0: store acc
                // Then move to next neuron or proceed to output layer
                S_H_RELU: begin
                    hidden_act[neuron_idx] <= acc[ACC_WIDTH-1] ? '0 : acc;

                    acc <= '0;    // reset accumulator for next neuron

                    if (neuron_idx == N_HIDDEN - 1) begin
                        // All hidden neurons computed → start output layer
                        neuron_idx <= '0;
                        state      <= S_O_MAC;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= S_H_MAC;
                    end
                end

                // ── Output layer: accumulate dot product ──────
                // Each cycle: acc += hidden_act[weight_idx] * ow[neuron_idx][weight_idx]
                // No ReLU on output layer — scores can be negative.
                S_O_MAC: begin
                    acc <= acc + hidden_act[weight_idx] *
                                 {{(ACC_WIDTH-W_WIDTH){ow[neuron_idx][weight_idx][W_WIDTH-1]}},
                                   ow[neuron_idx][weight_idx]};

                    if (weight_idx == N_HIDDEN - 1) begin
                        weight_idx <= '0;
                        state      <= S_O_BIAS;
                    end else begin
                        weight_idx <= weight_idx + 1;
                    end
                end

                // ── Add output bias (scaled) ──────────────────
                S_O_BIAS: begin
                    acc   <= acc + ({{(ACC_WIDTH-W_WIDTH){ob[neuron_idx][W_WIDTH-1]}},
                                      ob[neuron_idx]} *
                                    $signed({{(ACC_WIDTH-16){1'b0}},
                                             OUTPUT_BIAS_SCALE[15:0]}));
                    state <= S_O_STORE;
                end

                // ── Store output score ─────────────────────────
                S_O_STORE: begin
                    class_scores[neuron_idx] <= acc;
                    acc <= '0;

                    if (neuron_idx == N_OUT - 1) begin
                        state <= S_DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= S_O_MAC;
                    end
                end

                // ── Pulse done, return to IDLE ─────────────────
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule