`default_nettype none
`timescale 1ns/1ps

// ============================================================
// argmax.sv
//
// Finds the index of the largest of 4 signed 40-bit scores.
// Tie-breaking: lowest index wins (matches np.argmax default).
//
// Interface:
//   start_i   : pulse high for one cycle to begin compare
//   score0..3 : signed 40-bit scores from MLP output layer
//   class_o   : index of largest score (0-3)
//   valid_o   : pulses high in DONE state for one cycle
//
// FSM:
//   IDLE → COMPARE → DONE → IDLE
//
// Two-process FSM (sequential + combinational).
// ============================================================

module argmax (
    input  logic                clk,
    input  logic                rst_n,

    // Control
    input  logic                start_i,

    // Scores from MLP (signed 40-bit)
    input  logic signed [39:0]  score0_i,
    input  logic signed [39:0]  score1_i,
    input  logic signed [39:0]  score2_i,
    input  logic signed [39:0]  score3_i,

    // Result
    output logic        [1:0]   class_o,
    output logic                valid_o
);

    // --------------------------------------------------------
    // FSM state definition
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        COMPARE = 2'b01,
        DONE    = 2'b10
    } state_t;

    state_t state, next_state;

    // --------------------------------------------------------
    // Datapath registers
    // --------------------------------------------------------
    logic [1:0] class_reg;

    // --------------------------------------------------------
    // Combinational compare logic
    // Lowest index wins on ties because of strict > comparison.
    //
    // Walk through the 4 scores, keeping the running max:
    //   start with score0 as the leader
    //   if score1 > leader → leader = score1, idx = 1
    //   if score2 > leader → leader = score2, idx = 2
    //   if score3 > leader → leader = score3, idx = 3
    //
    // Implemented as a 4-way compare tree with priority on
    // the lowest index when scores are equal.
    // --------------------------------------------------------
    logic signed [39:0] max01_val;
    logic        [0:0]  max01_idx;
    logic signed [39:0] max23_val;
    logic        [0:0]  max23_idx;
    logic        [1:0]  final_idx;

    always_comb begin
        // Compare score0 vs score1
        if (score1_i > score0_i) begin
            max01_val = score1_i;
            max01_idx = 1'b1;
        end else begin
            max01_val = score0_i;
            max01_idx = 1'b0;
        end

        // Compare score2 vs score3
        if (score3_i > score2_i) begin
            max23_val = score3_i;
            max23_idx = 1'b1;
        end else begin
            max23_val = score2_i;
            max23_idx = 1'b0;
        end

        // Compare the two winners
        if (max23_val > max01_val) begin
            final_idx = {1'b1, max23_idx};   // 2 or 3
        end else begin
            final_idx = {1'b0, max01_idx};   // 0 or 1
        end
    end

    // --------------------------------------------------------
    // Process 1 : sequential — state + datapath
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            class_reg <= 2'b00;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    // wait for start
                end

                COMPARE: begin
                    // Latch the combinational compare result
                    class_reg <= final_idx;
                end

                DONE: ;

                default: ;
            endcase
        end
    end

    // --------------------------------------------------------
    // Process 2 : combinational — next-state logic
    // --------------------------------------------------------
    always_comb begin
        next_state = state;
        case (state)
            IDLE:    if (start_i) next_state = COMPARE;
            COMPARE:              next_state = DONE;
            DONE:                 next_state = IDLE;
            default:              next_state = IDLE;
        endcase
    end

    // --------------------------------------------------------
    // Output assignments
    // --------------------------------------------------------
    assign class_o = class_reg;
    assign valid_o = (state == DONE);

endmodule

`default_nettype wire
