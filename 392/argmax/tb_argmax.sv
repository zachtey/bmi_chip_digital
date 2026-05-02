`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_argmax.sv
//
// Loads expected scores from vectors/vecNN_scores.hex (4 lines,
// each a signed int64 stored as 16-hex-digit unsigned),
// loads expected class from vecNN_expected.hex,
// runs argmax, checks output.
//
// Note: scores are stored as 64-bit but argmax uses 40-bit.
// We sign-extend / truncate by taking the bottom 40 bits.
// All test vectors have scores well within 40-bit range
// (golden model max ~400K, 40-bit max ~5e11).
//
// Compile + run:
//   iverilog -g2012 -o argmax_sim tb_argmax.sv argmax.sv
//   vvp argmax_sim
// ============================================================

module tb_argmax;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int CLK_PERIOD = 10;
    localparam int N_VECTORS  = 40;

    // --------------------------------------------------------
    // DUT signals
    // --------------------------------------------------------
    logic                clk;
    logic                rst_n;
    logic                start_i;
    logic signed [39:0]  score0_i, score1_i, score2_i, score3_i;
    logic        [1:0]   class_o;
    logic                valid_o;

    // --------------------------------------------------------
    // DUT instantiation
    // --------------------------------------------------------
    argmax dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .start_i  (start_i),
        .score0_i (score0_i),
        .score1_i (score1_i),
        .score2_i (score2_i),
        .score3_i (score3_i),
        .class_o  (class_o),
        .valid_o  (valid_o)
    );

    // --------------------------------------------------------
    // Clock
    // --------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // --------------------------------------------------------
    // Storage for scores and expected class per vector
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
        input int           vec_idx,
        input logic [63:0]  s0_64,
        input logic [63:0]  s1_64,
        input logic [63:0]  s2_64,
        input logic [63:0]  s3_64,
        input logic [1:0]   exp_class
    );
        logic [1:0] got_class;
        int         timeout;

        tests_run++;

        // Truncate to 40 bits (sign-preserving since values are in range)
        @(posedge clk); #1;
        score0_i = s0_64[39:0];
        score1_i = s1_64[39:0];
        score2_i = s2_64[39:0];
        score3_i = s3_64[39:0];
        start_i  = 1'b1;
        @(posedge clk); #1;
        start_i  = 1'b0;

        // Wait for valid_o (max 10 cycles)
        timeout = 0;
        while (!valid_o && timeout < 10) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!valid_o) begin
            $display("[FAIL] Vec %02d | valid_o never asserted", vec_idx);
            return;
        end

        got_class = class_o;

        // Check
        if (got_class === exp_class) begin
            $display("[PASS] Vec %02d | scores=[%0d, %0d, %0d, %0d]  class=%0d",
                     vec_idx,
                     $signed(s0_64), $signed(s1_64),
                     $signed(s2_64), $signed(s3_64),
                     got_class);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | expected class=%0d  got=%0d",
                     vec_idx, exp_class, got_class);
        end

        // Wait for return to IDLE
        repeat(3) @(posedge clk);
    endtask

    // --------------------------------------------------------
    // Main
    // --------------------------------------------------------
    initial begin
        // Initialise
        rst_n    = 1'b0;
        start_i  = 1'b0;
        score0_i = 40'd0;
        score1_i = 40'd0;
        score2_i = 40'd0;
        score3_i = 40'd0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Load all scores and expected classes
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
        $display("  Argmax — Vector Testbench (%0d vectors)", N_VECTORS);
        $display("============================================");

        // Run all vectors
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

        // Summary
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
    // VCD dump
    // --------------------------------------------------------
    initial begin
        $dumpfile("argmax_waves.vcd");
        $dumpvars(0, tb_argmax);
    end

endmodule

`default_nettype wire
