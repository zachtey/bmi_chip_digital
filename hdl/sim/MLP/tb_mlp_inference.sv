`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_mlp_inference.sv 
//
// 1. Read weights.hex (108 bytes)
// 2. Shift them into the scan chain LSB-first (864 bits)
// 3. For each of 40 vectors:
//    - Load 8 SBP values into sbp_features
//    - Pulse start
//    - Wait for done
//    - Compare class_scores[0..3] against vec*_scores.hex
//
// Bias scales overridden to match weights.hex header:
//   HIDDEN_BIAS_SCALE = 85
//   OUTPUT_BIAS_SCALE = 109
//
// Compile + run from hdl/sim/:
//   iverilog -g2012 -o mlp_sim tb_mlp_inference.sv ../sv/mlp_inference.sv
//   vvp mlp_sim
//   surfer mlp_waves.vcd
// ============================================================

module tb_mlp_inference;

    // Parameters
    localparam int CLK_PERIOD  = 10;
    localparam int N_VECTORS   = 40;
    localparam int N_IN        = 8;
    localparam int N_HIDDEN    = 8;
    localparam int N_OUT       = 4;
    localparam int IN_WIDTH    = 8;
    localparam int W_WIDTH     = 8;
    localparam int SCORE_WIDTH = 32;
    localparam int N_WEIGHTS   = N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_IN + N_OUT; // 108
    localparam int TOTAL_BITS  = N_WEIGHTS * W_WIDTH;                          // 864

    // DUT signals
    logic                          clk;
    logic                          rst_n;
    logic                          start;
    logic                          done;
    logic [IN_WIDTH-1:0]           sbp_features [0:N_IN-1];
    logic signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1];
    logic                          scan_en;
    logic                          scan_clk;
    logic                          scan_in;

    // DUT
    mlp_inference #(
        .N_IN              (N_IN),
        .N_HIDDEN          (N_HIDDEN),
        .N_OUT             (N_OUT),
        .IN_WIDTH           (IN_WIDTH),
        .W_WIDTH            (W_WIDTH),
        .ACC_WIDTH          (32),
        .SCORE_WIDTH        (SCORE_WIDTH),
        .HIDDEN_BIAS_SCALE  (85),
        .OUTPUT_BIAS_SCALE  (109)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .done         (done),
        .sbp_features (sbp_features),
        .class_scores (class_scores),
        .scan_en      (scan_en),
        .scan_clk     (scan_clk),
        .scan_in      (scan_in)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Storage
    logic [7:0]  weights_mem [0:N_WEIGHTS-1];
    logic [7:0]  sbp_mem     [0:N_VECTORS*N_IN-1];
    logic [63:0] scores_mem  [0:N_VECTORS*N_OUT-1];

    // Tracking
    int tests_run    = 0;
    int tests_passed = 0;
    int score_mismatches;

    // ------------------------------------------------------
    // Shift all 108 weight bytes into scan chain LSB-first
    // ------------------------------------------------------
    task automatic load_weights();
        $display("Loading %0d weight bytes (%0d bits) via scan chain...",
                 N_WEIGHTS, TOTAL_BITS);
        scan_en  = 1'b1;
        scan_clk = 1'b0;

        for (int b = 0; b < N_WEIGHTS; b++) begin
            for (int bit_idx = 0; bit_idx < 8; bit_idx++) begin
                #2;
                scan_in  = weights_mem[b][bit_idx];
                #1;
                scan_clk = 1'b1;
                #2;
                scan_clk = 1'b0;
            end
        end

        scan_en = 1'b0;
        scan_in = 1'b0;

        @(posedge clk); #1;
    endtask

    // ------------------------------------------------------
    // Run one vector
    // ------------------------------------------------------
    task automatic run_vector(input int vec_idx);
        logic signed [SCORE_WIDTH-1:0] exp_s [0:N_OUT-1];
        logic signed [SCORE_WIDTH-1:0] got_s [0:N_OUT-1];
        bit                            ok;
        int                            timeout;

        tests_run++;
        score_mismatches = 0;
        ok = 1'b1;

        // Load SBP features
        @(posedge clk); #1;
        for (int i = 0; i < N_IN; i++)
            sbp_features[i] = sbp_mem[vec_idx*N_IN + i];

        // Expected scores (truncate 64-bit to 32-bit)
        for (int k = 0; k < N_OUT; k++)
            exp_s[k] = scores_mem[vec_idx*N_OUT + k][31:0];

        // Pulse start
        @(posedge clk); #1;
        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;

        // Wait for done
        timeout = 0;
        while (!done && timeout < 200) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!done) begin
            $display("[FAIL] Vec %02d | done never asserted", vec_idx);
            return;
        end

        for (int k = 0; k < N_OUT; k++)
            got_s[k] = class_scores[k];

        // Compare
        for (int k = 0; k < N_OUT; k++) begin
            if (got_s[k] !== exp_s[k]) begin
                if (score_mismatches < 4)
                    $display("       score[%0d]: expected=%0d  got=%0d",
                             k, $signed(exp_s[k]), $signed(got_s[k]));
                score_mismatches++;
                ok = 1'b0;
            end
        end

        if (ok) begin
            $display("[PASS] Vec %02d | scores=[%0d, %0d, %0d, %0d]",
                     vec_idx,
                     $signed(got_s[0]), $signed(got_s[1]),
                     $signed(got_s[2]), $signed(got_s[3]));
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | %0d score mismatches", vec_idx, score_mismatches);
        end

        repeat(3) @(posedge clk);
    endtask

    // ------------------------------------------------------
    // Main
    // ------------------------------------------------------
    initial begin
        rst_n    = 1'b0;
        start    = 1'b0;
        scan_en  = 1'b0;
        scan_clk = 1'b0;
        scan_in  = 1'b0;
        for (int i = 0; i < N_IN; i++) sbp_features[i] = '0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        // Load weights from weights.hex
        $readmemh("weights.hex", weights_mem);

        // Load all vectors
        for (int i = 0; i < N_VECTORS; i++) begin
            string fn_sbp;
            string fn_scores;
            logic [7:0]  s_buf  [0:N_IN-1];
            logic [63:0] sc_buf [0:N_OUT-1];

            $sformat(fn_sbp,    "vectors/vec%02d_sbp.hex",    i);
            $sformat(fn_scores, "vectors/vec%02d_scores.hex", i);
            $readmemh(fn_sbp,    s_buf);
            $readmemh(fn_scores, sc_buf);

            for (int n = 0; n < N_IN; n++)  sbp_mem[i*N_IN + n]    = s_buf[n];
            for (int n = 0; n < N_OUT; n++) scores_mem[i*N_OUT + n] = sc_buf[n];
        end

        $display("============================================");
        $display("  MLP Inference — Vector Testbench");
        $display("  Architecture: %0d -> %0d -> %0d", N_IN, N_HIDDEN, N_OUT);
        $display("  Bias scales: hidden=85, output=109");
        $display("============================================");

        load_weights();

        $display("Weights loaded. Running %0d vectors...", N_VECTORS);
        $display("");

        for (int i = 0; i < N_VECTORS; i++) begin
            run_vector(i);
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

    // Watchdog
    initial begin
        #50000000;
        $display("[TIMEOUT] Aborting");
        $finish;
    end

    // VCD
    initial begin
        $dumpfile("mlp_waves.vcd");
        $dumpvars(0, tb_mlp_inference);
    end

endmodule

`default_nettype wire