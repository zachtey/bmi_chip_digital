// argmax.v
//
// Finds the index of the largest signed 32-bit score among 4.
//
// Combinational comparison tree, registered output.
// Latency: 1 clock cycle (just pipeline register, no compute).
//
// Tournament:
//   Round 1: best_01 = (score[0] >= score[1]) ? 0 : 1
//            best_23 = (score[2] >= score[3]) ? 2 : 3
//   Round 2: best    = (score[best_01] >= score[best_23])
//                    ? best_01 : best_23
module argmax #(
    parameter N_CLASSES   = 4,
    parameter SCORE_WIDTH = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Scores from MLP (signed 32-bit, one per class)
    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1],

    // Strobe: asserted for exactly one cycle when scores are valid
    // Connect directly to mlp_done
    input  wire                          scores_valid,

    // Result — stable until next scores_valid pulse
    output reg  [1:0]                    predicted_class,
    output reg                           decision_valid    // 1-cycle pulse
);

    // ── Round 1: compare pairs ────────────────────────────────
    wire [1:0] best_01 = (class_scores[0] >= class_scores[1]) ? 2'd0 : 2'd1;
    wire [1:0] best_23 = (class_scores[2] >= class_scores[3]) ? 2'd2 : 2'd3;

    // ── Round 2: compare winners ──────────────────────────────
    // Use the winning index to mux the score — one more comparison.
    wire signed [SCORE_WIDTH-1:0] score_best_01 = class_scores[best_01];
    wire signed [SCORE_WIDTH-1:0] score_best_23 = class_scores[best_23];
    wire [1:0] best = (score_best_01 >= score_best_23) ? best_01 : best_23;

    // ── Register on scores_valid pulse ───────────────────────
    // Latch result the cycle MLP finishes.
    // predicted_class holds its value until the next inference.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            predicted_class <= 2'b00;
            decision_valid  <= 1'b0;
        end else begin
            decision_valid <= 1'b0;
            if (scores_valid) begin
                predicted_class <= best;
                decision_valid  <= 1'b1;
            end
        end
    end

endmodule