// argmax.sv
// Tournament comparator: returns the index of the largest signed score.
//
// Round 1: best_01 = winner of scores[0] vs scores[1]
//          best_23 = winner of scores[2] vs scores[3]
// Round 2: best    = winner of best_01 vs best_23
//
// Combinational comparison tree, result registered on scores_valid.
// Latency: 1 clock cycle.

module argmax #(
    parameter N_CLASSES   = 4,
    parameter SCORE_WIDTH = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1],
    input  wire                          scores_valid,

    output logic [1:0]                   predicted_class,
    output logic                         decision_valid
);

    // Combinational: tournament comparison tree
    logic [1:0]                   best_01, best_23, best;
    logic signed [SCORE_WIDTH-1:0] score_best_01, score_best_23;

    always_comb begin
        best_01       = (class_scores[0] >= class_scores[1]) ? 2'd0 : 2'd1;
        best_23       = (class_scores[2] >= class_scores[3]) ? 2'd2 : 2'd3;
        score_best_01 = class_scores[best_01];
        score_best_23 = class_scores[best_23];
        best          = (score_best_01 >= score_best_23) ? best_01 : best_23;
    end

    // Sequential: register result on scores_valid
    always_ff @(posedge clk or negedge rst_n) begin
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
