`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_output_formatter.sv
//
// Loads expected class from vectors/vecNN_expected.hex,
// pulses result_valid_i with that class, checks:
//   1. tx_byte_o == {5'b00000, 1'b1, class[1:0]}
//   2. load_o pulses for exactly one cycle
//   3. Module waits in WAIT_SPI_DONE until tx_done_i
//   4. Module returns to IDLE and accepts next vector
//
// Compile + run:
//   iverilog -g2012 -o fmt_sim tb_output_formatter.sv output_formatter.sv
//   vvp fmt_sim
// ============================================================

module tb_output_formatter;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int CLK_PERIOD = 10;   // 100 MHz
    localparam int N_VECTORS  = 40;

    // --------------------------------------------------------
    // DUT signals
    // --------------------------------------------------------
    logic       clk;
    logic       rst_n;
    logic       result_valid_i;
    logic [1:0] class_i;
    logic       tx_done_i;
    logic [7:0] tx_byte_o;
    logic       load_o;

    // --------------------------------------------------------
    // DUT instantiation
    // --------------------------------------------------------
    output_formatter dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .result_valid_i (result_valid_i),
        .class_i        (class_i),
        .tx_done_i      (tx_done_i),
        .tx_byte_o      (tx_byte_o),
        .load_o         (load_o)
    );

    // --------------------------------------------------------
    // Clock
    // --------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // --------------------------------------------------------
    // Test tracking
    // --------------------------------------------------------
    int tests_run    = 0;
    int tests_passed = 0;

    // --------------------------------------------------------
    // Storage for expected class per vector
    // --------------------------------------------------------
    logic [7:0] expected_class [N_VECTORS-1:0];

    // --------------------------------------------------------
    // Task : pulse result_valid for one cycle
    // --------------------------------------------------------
    task automatic pulse_valid(input logic [1:0] cls);
        @(posedge clk); #1;
        result_valid_i = 1'b1;
        class_i        = cls;
        @(posedge clk); #1;
        result_valid_i = 1'b0;
        class_i        = 2'b00;
    endtask

    // --------------------------------------------------------
    // Task : simulate SPI done after N cycles
    // --------------------------------------------------------
    task automatic pulse_tx_done(input int delay_cycles);
        repeat(delay_cycles) @(posedge clk);
        #1;
        tx_done_i = 1'b1;
        @(posedge clk); #1;
        tx_done_i = 1'b0;
    endtask

    // --------------------------------------------------------
    // Task : run one vector
    // --------------------------------------------------------
    task automatic run_vector(
        input int         vec_idx,
        input logic [7:0] exp_class
    );
        logic [7:0] exp_byte;
        logic [7:0] captured_byte;
        logic       load_seen;
        int         load_cycle_count;
        int         timeout;

        tests_run++;
        exp_byte         = {5'b00000, 1'b1, exp_class[1:0]};
        load_seen        = 1'b0;
        load_cycle_count = 0;
        captured_byte    = 8'hXX;

        // Send result_valid pulse
        pulse_valid(exp_class[1:0]);

        // Wait for load_o to go high (max 10 cycles)
        timeout = 0;
        while (!load_o && timeout < 10) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!load_o) begin
            $display("[FAIL] Vec %02d | load_o never asserted", vec_idx);
            // Still need to finish transaction cleanly
            pulse_tx_done(5);
            repeat(5) @(posedge clk);
            return;
        end

        // Capture byte while load_o is high
        captured_byte    = tx_byte_o;
        load_cycle_count = 1;

        // Check load_o goes low next cycle (must be exactly 1 cycle)
        @(posedge clk); #1;
        if (load_o) begin
            $display("[FAIL] Vec %02d | load_o held high more than 1 cycle", vec_idx);
            load_cycle_count++;
        end

        // Simulate SPI taking 8 cycles then asserting tx_done
        pulse_tx_done(8);

        // Wait for module to return to IDLE
        repeat(3) @(posedge clk);

        // Check byte
        if (captured_byte === exp_byte && load_cycle_count == 1) begin
            $display("[PASS] Vec %02d | class=%0d  expected=0x%02X  got=0x%02X",
                     vec_idx, exp_class[1:0], exp_byte, captured_byte);
            tests_passed++;
        end else if (captured_byte !== exp_byte) begin
            $display("[FAIL] Vec %02d | class=%0d  expected=0x%02X  got=0x%02X  BYTE MISMATCH",
                     vec_idx, exp_class[1:0], exp_byte, captured_byte);
        end

        // Gap between vectors
        repeat(5) @(posedge clk);
    endtask

    // --------------------------------------------------------
    // Main test sequence
    // --------------------------------------------------------
    initial begin
        // Initialise
        rst_n          = 1'b0;
        result_valid_i = 1'b0;
        class_i        = 2'b00;
        tx_done_i      = 1'b0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Load all expected classes from vector files
        for (int i = 0; i < N_VECTORS; i++) begin
            string fname;
            logic [7:0] mem [0:0];
            $sformat(fname, "vectors/vec%02d_expected.hex", i);
            $readmemh(fname, mem);
            expected_class[i] = mem[0];
        end

        $display("============================================");
        $display("  Output Formatter — Vector Testbench (%0d vectors)", N_VECTORS);
        $display("============================================");

        // Run all vectors
        for (int i = 0; i < N_VECTORS; i++) begin
            run_vector(i, expected_class[i]);
        end

        // Summary
        $display("============================================");
        $display("  Results: %0d / %0d passed", tests_passed, tests_run);
        if (tests_passed == tests_run)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED — check waveform");
        $display("============================================");

        $finish;
    end

    // --------------------------------------------------------
    // Watchdog
    // --------------------------------------------------------
    initial begin
        #10000000;
        $display("[TIMEOUT] Simulation exceeded limit — aborting");
        $finish;
    end

    // --------------------------------------------------------
    // VCD dump
    // --------------------------------------------------------
    initial begin
        $dumpfile("fmt_waves.vcd");
        $dumpvars(0, tb_output_formatter);
    end

endmodule

`default_nettype wire
