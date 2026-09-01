// mlp_inference.sv
// 8->8(ReLU)->4 fully-connected MLP classifier.
// Single shared MAC, time-multiplexed across all neurons.
//
// Weights are loaded once at power-up via a serial scan chain. For the default
// dimensions this is 864 bits (108 bytes): hw[8][8], hb[8], ow[4][8], ob[4].
// Do not assert start until the scan chain has been fully loaded.
//
// HIDDEN_BIAS_SCALE / OUTPUT_BIAS_SCALE come from the Python training
// pipeline (printed in weights.hex header). They correct the quantization
// scale mismatch between int8 weights and float32 biases.
//
// Latency: 64 (H_MAC) + 8 (H_BIAS) + 8 (H_RELU)
//        + 32 (O_MAC)  + 4 (O_BIAS) + 4 (O_STORE) + 1 (DONE)
//        = 121 clock cycles from start to done.

module mlp_inference #(
    parameter N_IN              = 8,
    parameter N_HIDDEN          = 8,
    parameter N_OUT             = 4,
    parameter IN_WIDTH          = 8,
    parameter W_WIDTH           = 8,
    parameter ACC_WIDTH         = 32,
    parameter SCORE_WIDTH       = 32,
    parameter HIDDEN_BIAS_SCALE = 228,
    parameter OUTPUT_BIAS_SCALE = 193
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          start,
    output logic                         done,

    input  wire [IN_WIDTH-1:0]           sbp_features [0:N_IN-1],
    output logic signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1],

    // Scan chain: shift 108 bytes LSB-first on scan_clk before asserting start.
    // Order: hw[0..7][0..7], hb[0..7], ow[0..3][0..7], ob[0..3]
    input  wire                          scan_en,
    input  wire                          scan_clk,
    input  wire                          scan_in
);

    // Derived widths use a minimum of one bit so legal one-element layers do
    // not create zero-width vectors through $clog2(1).
    localparam integer HIDDEN_IDX_WIDTH = (N_HIDDEN > 1) ? $clog2(N_HIDDEN) : 1;
    localparam integer OUTPUT_IDX_WIDTH = (N_OUT    > 1) ? $clog2(N_OUT)    : 1;
    localparam integer INPUT_IDX_WIDTH  = (N_IN     > 1) ? $clog2(N_IN)     : 1;

    localparam integer TOTAL_WEIGHT_BITS =
        (N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_HIDDEN + N_OUT) * W_WIDTH;

    // Conservative mathematical bounds for every legal input/weight value.
    // longint keeps this calculation independent of the datapath width. The
    // scales are expected to be nonnegative, but their absolute values make
    // the check safe if a signed scale is supplied.
    localparam longint signed MAX_INPUT_VALUE = (64'd1 << IN_WIDTH) - 1;
    localparam longint signed MAX_POS_WEIGHT  = (64'd1 << (W_WIDTH-1)) - 1;
    localparam longint signed MAX_ABS_WEIGHT  =  64'd1 << (W_WIDTH-1);
    localparam longint signed ABS_HIDDEN_SCALE =
        (HIDDEN_BIAS_SCALE < 0) ? -HIDDEN_BIAS_SCALE : HIDDEN_BIAS_SCALE;
    localparam longint signed ABS_OUTPUT_SCALE =
        (OUTPUT_BIAS_SCALE < 0) ? -OUTPUT_BIAS_SCALE : OUTPUT_BIAS_SCALE;

    localparam longint signed HIDDEN_MAX_VALUE =
        N_IN*MAX_INPUT_VALUE*MAX_POS_WEIGHT +
        MAX_POS_WEIGHT*ABS_HIDDEN_SCALE;
    localparam longint signed HIDDEN_MIN_VALUE =
        -(N_IN*MAX_INPUT_VALUE*MAX_ABS_WEIGHT +
          MAX_ABS_WEIGHT*ABS_HIDDEN_SCALE);
    localparam longint signed OUTPUT_MAX_VALUE =
        N_HIDDEN*HIDDEN_MAX_VALUE*MAX_POS_WEIGHT +
        MAX_POS_WEIGHT*ABS_OUTPUT_SCALE;
    localparam longint signed OUTPUT_MIN_VALUE =
        -(N_HIDDEN*HIDDEN_MAX_VALUE*MAX_ABS_WEIGHT +
          MAX_ABS_WEIGHT*ABS_OUTPUT_SCALE);

    // Return true when [minimum, maximum] fits in a signed two's-complement
    // vector of width bits. This is evaluated at elaboration time.
    function automatic integer signed_range_fits;
        input integer width;
        input longint signed minimum;
        input longint signed maximum;
        longint signed signed_minimum;
        longint signed signed_maximum;
        begin
            signed_minimum = -(64'sd1 << (width-1));
            signed_maximum =  (64'sd1 << (width-1)) - 1;
            signed_range_fits =
                (minimum >= signed_minimum) && (maximum <= signed_maximum);
        end
    endfunction

    logic [TOTAL_WEIGHT_BITS-1:0] scan_reg;

    always_ff @(posedge scan_clk)
        if (scan_en)
            scan_reg <= {scan_in, scan_reg[TOTAL_WEIGHT_BITS-1:1]};

    // Combinationally unpack scan_reg into named weight arrays.
    logic signed [W_WIDTH-1:0] hw [0:N_HIDDEN-1][0:N_IN-1];
    logic signed [W_WIDTH-1:0] hb [0:N_HIDDEN-1];
    logic signed [W_WIDTH-1:0] ow [0:N_OUT-1][0:N_HIDDEN-1];
    logic signed [W_WIDTH-1:0] ob [0:N_OUT-1];

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
            for (gj = 0; gj < N_HIDDEN; gj++) begin : unpack_ow_hidden
                assign ow[gi][gj] = scan_reg[((N_HIDDEN*N_IN + N_HIDDEN) + gi*N_HIDDEN + gj)*W_WIDTH +: W_WIDTH];
            end
        end
        for (gi = 0; gi < N_OUT; gi++) begin : unpack_ob
            assign ob[gi] = scan_reg[((N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_HIDDEN) + gi)*W_WIDTH +: W_WIDTH];
        end
    endgenerate

    // FSM
    typedef enum logic [2:0] {
        S_IDLE,
        S_H_MAC,
        S_H_BIAS,
        S_H_RELU,
        S_O_MAC,
        S_O_BIAS,
        S_O_STORE,
        S_DONE
    } state_t;

    state_t state, next_state;

    logic signed [ACC_WIDTH-1:0] acc;
    logic signed [ACC_WIDTH-1:0] hidden_act [0:N_HIDDEN-1];
    logic [HIDDEN_IDX_WIDTH-1:0] hidden_idx;
    logic [OUTPUT_IDX_WIDTH-1:0] output_idx;
    logic [INPUT_IDX_WIDTH-1:0]  input_idx;
    logic [HIDDEN_IDX_WIDTH-1:0] hidden_weight_idx;

    // Next-cycle values
    logic signed [ACC_WIDTH-1:0] next_acc;
    logic signed [ACC_WIDTH-1:0] next_hidden_act [0:N_HIDDEN-1];
    logic signed [SCORE_WIDTH-1:0] next_class_scores [0:N_OUT-1];
    logic [HIDDEN_IDX_WIDTH-1:0] next_hidden_idx;
    logic [OUTPUT_IDX_WIDTH-1:0] next_output_idx;
    logic [INPUT_IDX_WIDTH-1:0]  next_input_idx;
    logic [HIDDEN_IDX_WIDTH-1:0] next_hidden_weight_idx;
    logic next_done;

    // Zero-extend unsigned SBP inputs to accumulator width for signed multiply
    genvar gk;
    logic signed [ACC_WIDTH-1:0] sbp_signed [0:N_IN-1];
    generate
        for (gk = 0; gk < N_IN; gk++) begin : sign_extend_sbp
            assign sbp_signed[gk] = {{(ACC_WIDTH-IN_WIDTH){1'b0}}, sbp_features[gk]};
        end
    endgenerate

    // Combinational
    always_comb begin
        // Defaults — hold everything
        next_state      = state;
        next_acc        = acc;
        next_hidden_idx        = hidden_idx;
        next_output_idx        = output_idx;
        next_input_idx         = input_idx;
        next_hidden_weight_idx = hidden_weight_idx;
        next_done       = 1'b0;
        for (int i = 0; i < N_HIDDEN; i++) next_hidden_act[i]   = hidden_act[i];
        for (int i = 0; i < N_OUT;    i++) next_class_scores[i] = class_scores[i];

        unique case (state)
            S_IDLE: begin
                if (start) begin
                    next_acc        = '0;
                    next_hidden_idx        = '0;
                    next_output_idx        = '0;
                    next_input_idx         = '0;
                    next_hidden_weight_idx = '0;
                    next_state      = S_H_MAC;
                end
            end

            // Hidden layer dot product — one weight per cycle
            S_H_MAC: begin
                next_acc = acc + sbp_signed[input_idx] *
                           $signed(hw[hidden_idx][input_idx]);
                if (input_idx == N_IN - 1) begin
                    next_input_idx = '0;
                    next_state      = S_H_BIAS;
                end else begin
                    next_input_idx = input_idx + 1'b1;
                end
            end

            // Add scaled hidden bias
            S_H_BIAS: begin
                next_acc  = acc + ({{(ACC_WIDTH-W_WIDTH){hb[hidden_idx][W_WIDTH-1]}},
                                     hb[hidden_idx]} *
                                   $signed({{(ACC_WIDTH-16){1'b0}},
                                            HIDDEN_BIAS_SCALE[15:0]}));
                next_state = S_H_RELU;
            end

            // ReLU + store hidden activation; advance to next neuron
            S_H_RELU: begin
                next_hidden_act[hidden_idx] = acc[ACC_WIDTH-1] ? '0 : acc;
                next_acc = '0;

                if (hidden_idx == N_HIDDEN - 1) begin
                    next_hidden_idx        = '0;
                    next_hidden_weight_idx = '0;
                    next_state      = S_O_MAC;
                end else begin
                    next_hidden_idx = hidden_idx + 1'b1;
                    next_state      = S_H_MAC;
                end
            end

            // Output layer dot product — one hidden activation per cycle
            S_O_MAC: begin
                next_acc = acc + hidden_act[hidden_weight_idx] *
                           $signed(ow[output_idx][hidden_weight_idx]);
                if (hidden_weight_idx == N_HIDDEN - 1) begin
                    next_hidden_weight_idx = '0;
                    next_state      = S_O_BIAS;
                end else begin
                    next_hidden_weight_idx = hidden_weight_idx + 1'b1;
                end
            end

            // Add scaled output bias
            S_O_BIAS: begin
                next_acc  = acc + ({{(ACC_WIDTH-W_WIDTH){ob[output_idx][W_WIDTH-1]}},
                                     ob[output_idx]} *
                                   $signed({{(ACC_WIDTH-16){1'b0}},
                                            OUTPUT_BIAS_SCALE[15:0]}));
                next_state = S_O_STORE;
            end

            // Latch output score; advance to next class or finish
            S_O_STORE: begin
                next_class_scores[output_idx] = acc[SCORE_WIDTH-1:0];
                next_acc = '0;

                if (output_idx == N_OUT - 1) begin
                    next_state = S_DONE;
                end else begin
                    next_output_idx = output_idx + 1'b1;
                    next_state      = S_O_MAC;
                end
            end

            S_DONE: begin
                next_done  = 1'b1;
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    // Sequential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            acc         <= '0;
            hidden_idx        <= '0;
            output_idx        <= '0;
            input_idx         <= '0;
            hidden_weight_idx <= '0;
            for (int i = 0; i < N_HIDDEN; i++) hidden_act[i]   <= '0;
            for (int i = 0; i < N_OUT;    i++) class_scores[i] <= '0;
        end else begin
            state       <= next_state;
            done        <= next_done;
            acc         <= next_acc;
            hidden_idx        <= next_hidden_idx;
            output_idx        <= next_output_idx;
            input_idx         <= next_input_idx;
            hidden_weight_idx <= next_hidden_weight_idx;
            for (int i = 0; i < N_HIDDEN; i++) hidden_act[i]   <= next_hidden_act[i];
            for (int i = 0; i < N_OUT;    i++) class_scores[i] <= next_class_scores[i];
        end
    end

`ifndef SYNTHESIS
    // These are configuration errors, not runtime conditions. Keeping them
    // close to the RTL makes illegal parameter combinations fail immediately.
    initial begin
        if (N_IN < 1 || N_HIDDEN < 1 || N_OUT < 1)
            $fatal(1, "MLP layer sizes must all be at least one");
        if (IN_WIDTH > 62 || W_WIDTH > 62 || ACC_WIDTH > 63 || SCORE_WIDTH > 63)
            $fatal(1, "MLP bound calculation supports datapath widths up to 63 bits");
        if (HIDDEN_BIAS_SCALE < 0 || HIDDEN_BIAS_SCALE > 65535 ||
            OUTPUT_BIAS_SCALE < 0 || OUTPUT_BIAS_SCALE > 65535)
            $fatal(1, "MLP bias scales must be unsigned 16-bit values");
        if (ACC_WIDTH < IN_WIDTH || ACC_WIDTH < W_WIDTH || ACC_WIDTH < 16)
            $fatal(1, "ACC_WIDTH is too small for the configured operands");
        if (SCORE_WIDTH > ACC_WIDTH)
            $fatal(1, "SCORE_WIDTH must not exceed ACC_WIDTH");
        if (!signed_range_fits(ACC_WIDTH, HIDDEN_MIN_VALUE, HIDDEN_MAX_VALUE))
            $fatal(1, "ACC_WIDTH cannot hold hidden range [%0d, %0d]",
                   HIDDEN_MIN_VALUE, HIDDEN_MAX_VALUE);
        if (!signed_range_fits(ACC_WIDTH, OUTPUT_MIN_VALUE, OUTPUT_MAX_VALUE))
            $fatal(1, "ACC_WIDTH cannot hold output range [%0d, %0d]",
                   OUTPUT_MIN_VALUE, OUTPUT_MAX_VALUE);
        if (!signed_range_fits(SCORE_WIDTH, OUTPUT_MIN_VALUE, OUTPUT_MAX_VALUE))
            $fatal(1, "SCORE_WIDTH cannot hold output range [%0d, %0d]",
                   OUTPUT_MIN_VALUE, OUTPUT_MAX_VALUE);
    end
`endif
endmodule
