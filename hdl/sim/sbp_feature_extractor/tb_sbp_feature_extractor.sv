`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_sbp_feature_extractor.sv  (for partner's sbp_feature_extractor.sv)
//
// Module name in the file is `sbp_feature_extraction` (note: not
// _extractor) — file naming inconsistency on partner's side.
//
// For each of 40 vectors:
//   1. Load 2000 ADC bytes from vectors/vecNN_adc.hex
//      Layout: ch0[0..249], ch1[0..249], ..., ch7[0..249]
//   2. Pre-load sample_window[ch][s] directly
//   3. Pulse start
//   4. Wait for done
//   5. Compare sbp_features[0..7] to vectors/vecNN_sbp.hex
//
// Compile + run from hdl/sim/:
//   iverilog -g2012 -o sbp_sim tb_sbp_feature_extractor.sv \
//                              ../sv/sbp_feature_extractor.sv
//   vvp sbp_sim
//   surfer sbp_waves.vcd
// ============================================================

module tb_sbp_feature_extractor;

    // Parameters
    localparam int CLK_PERIOD = 10;
    localparam int N_VECTORS  = 40;
    localparam int N_CH       = 8;
    localparam int N_SAMPLES  = 250;
    localparam int ADC_WIDTH  = 8;
    localparam int SBP_WIDTH  = 8;
    localparam int N_TOTAL    = N_CH * N_SAMPLES;

    // DUT signals
    logic                       clk;
    logic                       rst_n;
    logic                       start;
    logic                       done;
    logic [ADC_WIDTH-1:0]       sample_window [0:N_CH-1][0:N_SAMPLES-1];
    logic [SBP_WIDTH-1:0]       sbp_features  [0:N_CH-1];

    // DUT (note: module name is sbp_feature_extraction, not _extractor)
    sbp_feature_extraction #(
        .N_CH      (N_CH),
        .N_SAMPLES (N_SAMPLES),
        .ADC_WIDTH (ADC_WIDTH),
        .SBP_WIDTH (SBP_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .done          (done),
        .sample_window (sample_window),
        .sbp_features  (sbp_features)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Storage
    logic [7:0] adc_mem      [0:N_VECTORS*N_TOTAL-1];
    logic [7:0] expected_sbp [0:N_VECTORS*N_CH-1];

    // Tracking
    int tests_run    = 0;
    int tests_passed = 0;

    // ------------------------------------------------------
    // Pre-load sample_window from adc_mem for given vector
    // ------------------------------------------------------
    task automatic load_window(input int vec_idx);
        for (int ch = 0; ch < N_CH; ch++) begin
            for (int s = 0; s < N_SAMPLES; s++) begin
                sample_window[ch][s] = adc_mem[vec_idx*N_TOTAL + ch*N_SAMPLES + s];
            end
        end
    endtask

    // ------------------------------------------------------
    // Run one vector
    // ------------------------------------------------------
    task automatic run_vector(input int vec_idx);
        bit  ok;
        int  timeout;
        int  mismatches;

        tests_run++;
        mismatches = 0;
        ok = 1'b1;

        // Load window
        @(posedge clk); #1;
        load_window(vec_idx);

        // Pulse start
        @(posedge clk); #1;
        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;

        // Wait for done (max 2500 cycles — should take ~2000)
        timeout = 0;
        while (!done && timeout < 2500) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!done) begin
            $display("[FAIL] Vec %02d | done never asserted", vec_idx);
            return;
        end

        // Verify all 8 SBP values
        for (int ch = 0; ch < N_CH; ch++) begin
            logic [7:0] expected;
            expected = expected_sbp[vec_idx*N_CH + ch];
            if (sbp_features[ch] !== expected) begin
                if (mismatches < 5)
                    $display("       ch%0d: expected=%02h (%0d)  got=%02h (%0d)",
                             ch, expected, expected,
                             sbp_features[ch], sbp_features[ch]);
                mismatches++;
                ok = 1'b0;
            end
        end

        if (ok) begin
            $display("[PASS] Vec %02d | sbp=[%0d %0d %0d %0d %0d %0d %0d %0d]",
                     vec_idx,
                     sbp_features[0], sbp_features[1],
                     sbp_features[2], sbp_features[3],
                     sbp_features[4], sbp_features[5],
                     sbp_features[6], sbp_features[7]);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | %0d ch mismatches", vec_idx, mismatches);
        end

        repeat(5) @(posedge clk);
    endtask

    // ------------------------------------------------------
    // Main
    // ------------------------------------------------------
    initial begin
        // Init
        rst_n = 1'b0;
        start = 1'b0;
        for (int ch = 0; ch < N_CH; ch++)
            for (int s = 0; s < N_SAMPLES; s++)
                sample_window[ch][s] = '0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        // Load all vectors
        for (int i = 0; i < N_VECTORS; i++) begin
            string fn_adc;
            string fn_sbp;
            logic [7:0] adc_buf [0:N_TOTAL-1];
            logic [7:0] sbp_buf [0:N_CH-1];

            $sformat(fn_adc, "vectors/vec%02d_adc.hex", i);
            $sformat(fn_sbp, "vectors/vec%02d_sbp.hex", i);
            $readmemh(fn_adc, adc_buf);
            $readmemh(fn_sbp, sbp_buf);

            for (int n = 0; n < N_TOTAL; n++)
                adc_mem[i*N_TOTAL + n] = adc_buf[n];
            for (int ch = 0; ch < N_CH; ch++)
                expected_sbp[i*N_CH + ch] = sbp_buf[ch];
        end

        $display("============================================");
        $display("  SBP Feature Extractor — Vector Testbench");
        $display("  Formula: SBP[ch] = sum(|adc - 128|) >> 8");
        $display("  %0d vectors, %0d channels each", N_VECTORS, N_CH);
        $display("============================================");

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
        #5000000000;
        $display("[TIMEOUT] Aborting");
        $finish;
    end

    // VCD
    initial begin
        $dumpfile("sbp_waves.vcd");
        $dumpvars(0, tb_sbp_feature_extractor);
    end

endmodule

`default_nettype wire