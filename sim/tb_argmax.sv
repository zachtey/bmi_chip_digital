`timescale 1ns/1ps

module tb_argmax;

    localparam int SCORE_WIDTH = 32;
    localparam int CLK_PERIOD = 10; //tb runs on 100 MHz clock

    //dut inputs 
    logic clk, rst_n;
    logic signed [SCORE_WIDTH-1:0] class_scores [0:3];
    logic scores_valid;
    //dut outputs
    logic [1:0] predicted_class;
    logic decision_valid;

    // Testbench result counters
    integer pass_count;
    integer fail_count;

    //dut instantiation
    argmax #(
    .N_CLASSES  (4),
    .SCORE_WIDTH(SCORE_WIDTH)
) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .class_scores   (class_scores),
    .scores_valid   (scores_valid),
    .predicted_class(predicted_class),
    .decision_valid (decision_valid)
);

    //process 1: clock generation (100 MHz)
    initial clk = 1'b0;
    always # (CLK_PERIOD/2) clk = ~clk;

    //process 2: apply reset and test sequence, prints results
    initial begin
        rst_n        = 1'b0;
        scores_valid = 1'b0;
        class_scores[0] = '0;
        class_scores[1] = '0;
        class_scores[2] = '0;
        class_scores[3] = '0;
        pass_count = 0;
        fail_count = 0;

        // Hold reset for several clock cycles.
        repeat (3) @(posedge clk);
        #1;

        //check for reset correctness
        if (predicted_class !== 2'd0 ||
            decision_valid  !== 1'b0) begin
            $fatal(
                1,
                "Reset values incorrect: class=%b valid=%b",
                predicted_class,
                decision_valid
            );
        end

        // release reset away from the rising clock edge
        @(negedge clk);
        rst_n = 1'b1;

        // ARG-001: each class is the unique maximum
        check_case(1,  40,  30,  20,  10, 2'd0);
        check_case(2,  10,  40,  30,  20, 2'd1);
        check_case(3,  20,  10,  40,  30, 2'd2);
        check_case(4,  30,  20,  10,  40, 2'd3);

        // ARG-002: signed comparisons and negative scores
        check_case(5, -10, -20, -30, -40, 2'd0);
        check_case(6, -40,  -5, -20, -30, 2'd1);
        check_case(7, -40, -30,  -5, -20, 2'd2);
        check_case(8, -40, -30, -20,  -5, 2'd3);

        // Signed 32-bit boundaries:
        // +2147483647 must beat 0, -1, and -2147483648.
        check_case(
            9,
            32'sh7fff_ffff,
            32'sh0000_0000,
            -32'sd1,
            32'sh8000_0000,
            2'd0
        );

        // Most-negative signed number must lose to every ordinary
        // negative score.
        check_case(
            10,
            32'sh8000_0000,
            -32'sd100,
            -32'sd50,
            -32'sd1,
            2'd3
        );

        // ARG-003: documented tie policy is lowest index
        check_case(11, 10, 10,  5,  5, 2'd0);
        check_case(12,  5,  5, 10, 10, 2'd2);
        check_case(13, 10,  5, 10,  5, 2'd0);
        check_case(14,  5, 10,  5, 10, 2'd1);
        check_case(15, 10, 10, 10, 10, 2'd0);

        // ARG-005: check invalid scores
        check_invalid_scores(
            16,
            -32'sd100,
            -32'sd100,
            -32'sd100,
            32'sd100
        );

        //check async reset operation
        check_async_reset(17);

        //End of tests, print results
        $display("----------------------------------------");
        $display(
            "ARGMAX TEST: %0d passed, %0d failed",
            pass_count,
            fail_count
        );
        $display("----------------------------------------");

        if (fail_count != 0)
            $fatal(1, "Argmax regression failed");

        $display("ALL ARGMAX TESTS PASS");
        $finish;
    
    end

    //process 3: watch for timeout
    initial begin
        #(1000 * CLK_PERIOD);
        $fatal(1, "Argmax testbench timeout");
    end


    // tasks!!
    // apply one score vector and check the registered result.
    task automatic check_case;
        input integer test_id;
        input logic signed [SCORE_WIDTH-1:0] score0;
        input logic signed [SCORE_WIDTH-1:0] score1;
        input logic signed [SCORE_WIDTH-1:0] score2;
        input logic signed [SCORE_WIDTH-1:0] score3;
        input logic [1:0] expected_class;

        logic case_pass; //accumulative case pass; takes one fail to set low per task

        begin
            case_pass = 1'b1;
            @(negedge clk);
            class_scores[0] = score0;
            class_scores[1] = score1;
            class_scores[2] = score2;
            class_scores[3] = score3;
            scores_valid    = 1'b1;

            @(posedge clk);
            #1;

            if (decision_valid !== 1'b1) begin
                $display(
                    "FAIL test %0d: decision_valid got=%b expected=1",
                    test_id,
                    decision_valid
                );
                case_pass = 1'b0;
            end else if (predicted_class !== expected_class) begin
                $display(
                    "FAIL test %0d: scores=[%0d,%0d,%0d,%0d] got=%0d expected=%0d",
                    test_id,
                    $signed(score0),
                    $signed(score1),
                    $signed(score2),
                    $signed(score3),
                    predicted_class,
                    expected_class
                );
                case_pass = 1'b0;
            end

            // Remove valid and make sure decision_valid is a pulse.
            @(negedge clk);
            scores_valid = 1'b0;

            @(posedge clk);
            #1;

            if (decision_valid !== 1'b0) begin
                $display(
                    "FAIL test %0d: decision_valid did not return low",
                    test_id
                );
                case_pass = 1'b0;
            end

            if (case_pass) begin
                $display(
                    "PASS test %0d: scores=[%0d,%0d,%0d,%0d] class=%0d",
                    test_id,
                    $signed(score0),
                    $signed(score1),
                    $signed(score2),
                    $signed(score3),
                    predicted_class
                );
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    // change scores while scores_valid is low and check output stability
    task automatic check_invalid_scores;
        input integer test_id;

        input logic signed [SCORE_WIDTH-1:0] score0;
        input logic signed [SCORE_WIDTH-1:0] score1;
        input logic signed [SCORE_WIDTH-1:0] score2;
        input logic signed [SCORE_WIDTH-1:0] score3;

        logic [1:0] held_class;

        begin
            // Remember the currently registered result.
            held_class = predicted_class;

            // Change scores while declaring them invalid.
            @(negedge clk);

            class_scores[0] = score0;
            class_scores[1] = score1;
            class_scores[2] = score2;
            class_scores[3] = score3;
            scores_valid    = 1'b0;

            // Give the DUT a rising edge, then sample registered outputs.
            @(posedge clk);
            #1;

            if (decision_valid !== 1'b0) begin
                $display(
                    "FAIL test %0d: decision_valid=%b while scores_valid=0",
                    test_id,
                    decision_valid
                );
                fail_count = fail_count + 1;
            end else if (predicted_class !== held_class) begin
                $display(
                    "FAIL test %0d: class changed while invalid, got=%0d held=%0d",
                    test_id,
                    predicted_class,
                    held_class
                );
                fail_count = fail_count + 1;
            end else begin
                $display(
                    "PASS test %0d: invalid scores ignored, class held at %0d",
                    test_id,
                    predicted_class
                );
                pass_count = pass_count + 1;
            end
        end
    endtask

    //checks for async reset correctness 
    task automatic check_async_reset;
        input integer test_id;

        begin
            // Assert reset away from the DUT's active rising clock edge.
            @(negedge clk);
            rst_n = 1'b0;

            // The reset is asynchronous, so outputs should reset without
            // waiting for another rising edge.
            #1;

            if (predicted_class !== 2'd0) begin
                $display(
                    "FAIL test %0d: reset class got=%b expected=00",
                    test_id,
                    predicted_class
                );
                fail_count = fail_count + 1;
            end else if (decision_valid !== 1'b0) begin
                $display(
                    "FAIL test %0d: reset valid got=%b expected=0",
                    test_id,
                    decision_valid
                );
                fail_count = fail_count + 1;
            end else begin
                $display(
                    "PASS test %0d: asynchronous reset",
                    test_id
                );
                pass_count = pass_count + 1;
            end

            // Leave the DUT ready for additional tests.
            @(negedge clk);
            rst_n = 1'b1;
        end
    endtask

endmodule


/*
argmax.sv
Tournament comparator: returns the index of the largest signed score.

Round 1: best_01 = winner of scores[0] vs scores[1]
         best_23 = winner of scores[2] vs scores[3]
Round 2: best    = winner of best_01 vs best_23

Combinational comparison tree, result registered on scores_valid.
Latency: 1 clock cycle.

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
*/
