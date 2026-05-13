`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_sample_collection.sv  (for partner's sample_collection.sv)
//
// For each of 40 vectors:
//   1. Load 2000 ADC bytes from vectors/vecNN_adc.hex
//      Layout: ch0[0..249], ch1[0..249], ..., ch7[0..249]
//   2. Stream samples one per cycle into adc_sample/adc_channel
//      with adc_valid asserted
//   3. Wait for window_ready pulse
//   4. Verify sample_window[ch][s] matches every byte we sent
//
// Compile + run from hdl/sim/:
//   iverilog -g2012 -o sc_sim tb_sample_collection.sv \
//                              ../sv/sample_collection.sv
//   vvp sc_sim
//   surfer sc_waves.vcd
// ============================================================

module tb_sample_collection;

    // Parameters
    localparam int CLK_PERIOD = 10;
    localparam int N_VECTORS  = 40;
    localparam int N_CH       = 8;
    localparam int N_SAMPLES  = 250;
    localparam int ADC_WIDTH  = 8;
    localparam int N_TOTAL    = N_CH * N_SAMPLES;   // 2000

    // DUT signals
    logic                        clk;
    logic                        rst_n;
    logic [ADC_WIDTH-1:0]        adc_sample;
    logic                        adc_valid;
    logic [$clog2(N_CH)-1:0]     adc_channel;
    logic                        window_ready;
    logic [ADC_WIDTH-1:0]        sample_window [0:N_CH-1][0:N_SAMPLES-1];

    // DUT
    sample_collection #(
        .N_CH      (N_CH),
        .N_SAMPLES (N_SAMPLES),
        .ADC_WIDTH (ADC_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .adc_sample    (adc_sample),
        .adc_valid     (adc_valid),
        .adc_channel   (adc_channel),
        .window_ready  (window_ready),
        .sample_window (sample_window)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Storage: 40 vectors × 2000 samples each
    logic [7:0] adc_mem [0:N_VECTORS*N_TOTAL-1];

    // Tracking
    int tests_run    = 0;
    int tests_passed = 0;
    int mismatches;     // updated by verify_window

    // ------------------------------------------------------
    // Stream one full window's worth of ADC samples
    // ------------------------------------------------------
    task automatic stream_window(input int vec_idx);
        for (int ch = 0; ch < N_CH; ch++) begin
            for (int s = 0; s < N_SAMPLES; s++) begin
                @(posedge clk); #1;
                adc_sample  = adc_mem[vec_idx*N_TOTAL + ch*N_SAMPLES + s];
                adc_channel = ch[$clog2(N_CH)-1:0];
                adc_valid   = 1'b1;
            end
        end
        @(posedge clk); #1;
        adc_valid = 1'b0;
    endtask

    // ------------------------------------------------------
    // Verify window contents match what we streamed
    // Updates module-level `mismatches` count
    // ------------------------------------------------------
    function automatic bit verify_window(input int vec_idx);
        bit ok;
        ok         = 1'b1;
        mismatches = 0;

        for (int ch = 0; ch < N_CH; ch++) begin
            for (int s = 0; s < N_SAMPLES; s++) begin
                logic [7:0] expected;
                expected = adc_mem[vec_idx*N_TOTAL + ch*N_SAMPLES + s];
                if (sample_window[ch][s] !== expected) begin
                    if (mismatches < 5)
                        $display("       ch%0d[%0d]: expected=%02h  got=%02h",
                                 ch, s, expected, sample_window[ch][s]);
                    mismatches++;
                    ok = 1'b0;
                end
            end
        end
        return ok;
    endfunction

    // ------------------------------------------------------
    // Run one vector
    // ------------------------------------------------------
    task automatic run_vector(input int vec_idx);
        bit  ok;
        int  timeout;

        tests_run++;

        // Stream samples
        stream_window(vec_idx);

        // Wait for window_ready (max 50 cycles after last sample)
        timeout = 0;
        while (!window_ready && timeout < 50) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!window_ready) begin
            $display("[FAIL] Vec %02d | window_ready never asserted", vec_idx);
            return;
        end

        // Verify (updates module-level `mismatches`)
        ok = verify_window(vec_idx);

        if (ok) begin
            $display("[PASS] Vec %02d | all %0d samples match",
                     vec_idx, N_TOTAL);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | %0d mismatches (showed first 5)",
                     vec_idx, mismatches);
        end

        // Wait a few cycles before next vector
        repeat(5) @(posedge clk);
    endtask

    // ------------------------------------------------------
    // Main
    // ------------------------------------------------------
    initial begin
        // Init
        rst_n       = 1'b0;
        adc_sample  = '0;
        adc_valid   = 1'b0;
        adc_channel = '0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        // Load all vectors
        for (int i = 0; i < N_VECTORS; i++) begin
            string fn;
            logic [7:0] tmp [0:N_TOTAL-1];
            $sformat(fn, "vectors/vec%02d_adc.hex", i);
            $readmemh(fn, tmp);
            for (int n = 0; n < N_TOTAL; n++)
                adc_mem[i*N_TOTAL + n] = tmp[n];
        end

        $display("============================================");
        $display("  Sample Collection — Vector Testbench");
        $display("  %0d vectors, %0d samples each (%0d ch x %0d)",
                 N_VECTORS, N_TOTAL, N_CH, N_SAMPLES);
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
        $dumpfile("sc_waves.vcd");
        $dumpvars(0, tb_sample_collection);
    end

endmodule

`default_nettype wire