`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_argmax.sv  
//
// Loads scores from vectors/vecNN_scores.hex (4 x signed int64),
// truncates to 32 bits, drives them into class_scores[0..3],
// pulses scores_valid, checks predicted_class against
// vecNN_expected.hex.
//
// Compile + run from hdl/sim/
//   iverilog -g2012 -o argmax_sim tb_argmax.sv ../sv/argmax.sv
//   vvp argmax_sim
//   surfer argmax_waves.vcd
// ============================================================

module tb_argmax;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int CLK_PERIOD  = 10;   // 100 MHz
    localparam int N_VECTORS   = 40;
    localparam int SCORE_WIDTH = 32;

    // --------------------------------------------------------
    // DUT signals
    // --------------------------------------------------------
    logic                          clk;
    logic                          rst_n;
    logic signed [SCORE_WIDTH-1:0] class_scores [0:3];
    logic                          scores_valid;
    logic [1:0]                    predicted_class;
    logic                          decision_valid;

    // --------------------------------------------------------
    // DUT
    // --------------------------------------------------------
    argmax #(
        .N_CLASSES   (4),
        .SCORE_WIDTH (32)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .class_scores    (class_scores),
        .scores_valid    (scores_valid),
        .predicted_class (predicted_class),
        .decision_valid  (decision_valid)
    );

    // --------------------------------------------------------
    // Clock
    // --------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // --------------------------------------------------------
    // Storage
    // --------------------------------------------------------
    logic [63:0] scores_mem    [0:N_VECTORS*4-1];
    logic [7:0]  expected_class[0:N_VECTORS-1];

    // --------------------------------------------------------
    // Test tracking
    // --------------------------------------------------------
    int tests_run    = 0;
    int tests_passed = 0;

    // --------------------------------------------------------
    // Task : run one vector
    // --------------------------------------------------------
    task automatic run_vector(
        input int          vec_idx,
        input logic [63:0] s0_64,
        input logic [63:0] s1_64,
        input logic [63:0] s2_64,
        input logic [63:0] s3_64,
        input logic [1:0]  exp_class
    );
        logic [1:0] got_class;
        int         timeout;

        tests_run++;

        // Drive scores (truncate 64-bit to 32-bit; values within range)
        @(posedge clk); #1;
        class_scores[0] = s0_64[31:0];
        class_scores[1] = s1_64[31:0];
        class_scores[2] = s2_64[31:0];
        class_scores[3] = s3_64[31:0];
        scores_valid    = 1'b1;
        @(posedge clk); #1;
        scores_valid    = 1'b0;

        // Wait for decision_valid
        timeout = 0;
        while (!decision_valid && timeout < 10) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!decision_valid) begin
            $display("[FAIL] Vec %02d | decision_valid never asserted", vec_idx);
            return;
        end

        got_class = predicted_class;

        if (got_class === exp_class) begin
            $display("[PASS] Vec %02d | scores=[%0d, %0d, %0d, %0d]  class=%0d",
                     vec_idx,
                     $signed(s0_64[31:0]), $signed(s1_64[31:0]),
                     $signed(s2_64[31:0]), $signed(s3_64[31:0]),
                     got_class);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | expected class=%0d  got=%0d",
                     vec_idx, exp_class, got_class);
        end

        repeat(3) @(posedge clk);
    endtask

    // --------------------------------------------------------
    // Main
    // --------------------------------------------------------
    initial begin
        // Init
        rst_n        = 1'b0;
        scores_valid = 1'b0;
        for (int i = 0; i < 4; i++) class_scores[i] = '0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Load scores and expected classes from vector files
        for (int i = 0; i < N_VECTORS; i++) begin
            string fname_scores;
            string fname_expected;
            logic [63:0] s_mem [0:3];
            logic [7:0]  e_mem [0:0];

            $sformat(fname_scores,   "vectors/vec%02d_scores.hex",   i);
            $sformat(fname_expected, "vectors/vec%02d_expected.hex", i);
            $readmemh(fname_scores,   s_mem);
            $readmemh(fname_expected, e_mem);

            scores_mem[i*4 + 0] = s_mem[0];
            scores_mem[i*4 + 1] = s_mem[1];
            scores_mem[i*4 + 2] = s_mem[2];
            scores_mem[i*4 + 3] = s_mem[3];
            expected_class[i]   = e_mem[0];
        end

        $display("============================================");
        $display("  Argmax (partner's version) — Vector Testbench");
        $display("  N_CLASSES = 4, SCORE_WIDTH = 32");
        $display("============================================");

        for (int i = 0; i < N_VECTORS; i++) begin
            run_vector(
                i,
                scores_mem[i*4 + 0],
                scores_mem[i*4 + 1],
                scores_mem[i*4 + 2],
                scores_mem[i*4 + 3],
                expected_class[i][1:0]
            );
        end

        $display("============================================");
        $display("  Results: %0d / %0d passed", tests_passed, tests_run);
        if (tests_passed == tests_run)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");
        $display("============================================");

        $finish;
    end

    // --------------------------------------------------------
    // Watchdog
    // --------------------------------------------------------
    initial begin
        #10000000;
        $display("[TIMEOUT] Aborting");
        $finish;
    end

    // --------------------------------------------------------
    // VCD
    // --------------------------------------------------------
    initial begin
        $dumpfile("argmax_waves.vcd");
        $dumpvars(0, tb_argmax);
    end

endmodule

`default_nettype wire