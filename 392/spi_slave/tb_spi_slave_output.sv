`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_spi_slave_output.sv
//
// Loads expected class from vectors/vecNN_expected.hex,
// packs into SPI byte {5'b00000, 1'b1, class[1:0]},
// runs SPI transaction, checks MISO output.
//
// Compile + run:
//   iverilog -g2012 -o spi_sim tb_spi_slave_output.sv spi_slave_output.sv
//   vvp spi_sim
// ============================================================

module tb_spi_slave_output;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int CLK_PERIOD = 10;    // 100 MHz system clock (ns)
    localparam int SPI_HALF   = 50;    // 10 MHz SPI clock half-period (ns)
    localparam int N_VECTORS  = 40;

    // --------------------------------------------------------
    // DUT signals
    // --------------------------------------------------------
    logic       clk;
    logic       rst_n;
    logic       load_i;
    logic [7:0] tx_byte_i;
    logic       spi_sclk_i;
    logic       spi_cs_n_i;
    logic       spi_miso_o;
    logic       tx_done_o;

    // --------------------------------------------------------
    // DUT instantiation
    // --------------------------------------------------------
    spi_slave_output dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .load_i     (load_i),
        .tx_byte_i  (tx_byte_i),
        .spi_sclk_i (spi_sclk_i),
        .spi_cs_n_i (spi_cs_n_i),
        .spi_miso_o (spi_miso_o),
        .tx_done_o  (tx_done_o)
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
    // Task : load a byte into the DUT (one sys-clk pulse)
    // --------------------------------------------------------
    task automatic do_load(input logic [7:0] byte_in);
        @(posedge clk); #1;
        load_i    = 1'b1;
        tx_byte_i = byte_in;
        @(posedge clk); #1;
        load_i    = 1'b0;
    endtask

    // --------------------------------------------------------
    // Task : run one SPI transaction, return captured byte
    // --------------------------------------------------------
    task automatic spi_transfer(output logic [7:0] rx_byte);
        logic [7:0] captured;
        captured = 8'h00;

        // Assert CS
        spi_cs_n_i = 1'b0;
        #(SPI_HALF);

        for (int b = 7; b >= 0; b--) begin
            // Rising edge — master samples MISO
            spi_sclk_i = 1'b1;
            #(SPI_HALF/2);
            captured[b] = spi_miso_o;
            #(SPI_HALF/2);

            // Falling edge — DUT shifts next bit
            spi_sclk_i = 1'b0;
            #(SPI_HALF);
        end

        // Deassert CS
        #(SPI_HALF);
        spi_cs_n_i = 1'b1;
        #(SPI_HALF);

        rx_byte = captured;
    endtask

    // --------------------------------------------------------
    // Task : run one vector
    // --------------------------------------------------------
    task automatic run_vector(
        input int          vec_idx,
        input logic [7:0]  exp_class
    );
        logic [7:0] tx_byte;
        logic [7:0] rx_byte;
        logic [7:0] exp_byte;

        tests_run++;

        // Pack byte: {5'b00000, 1'b1, class[1:0]}
        tx_byte  = {5'b00000, 1'b1, exp_class[1:0]};
        exp_byte = tx_byte;

        // Load into DUT
        do_load(tx_byte);

        // Wait a few cycles before CS
        repeat(5) @(posedge clk);

        // Run SPI transaction
        spi_transfer(rx_byte);

        // Wait for tx_done
        repeat(10) @(posedge clk);

        // Check
        if (rx_byte === exp_byte) begin
            $display("[PASS] Vec %02d | class=%0d  byte=0x%02X  received=0x%02X",
                     vec_idx, exp_class[1:0], exp_byte, rx_byte);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | class=%0d  byte=0x%02X  received=0x%02X  MISMATCH",
                     vec_idx, exp_class[1:0], exp_byte, rx_byte);
        end

        // Gap between transactions
        repeat(20) @(posedge clk);
    endtask

    // --------------------------------------------------------
    // Main test sequence
    // --------------------------------------------------------
    initial begin
        // Initialise signals
        rst_n      = 1'b0;
        load_i     = 1'b0;
        tx_byte_i  = 8'h00;
        spi_sclk_i = 1'b0;
        spi_cs_n_i = 1'b1;

        // Reset
        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Load all expected classes from vector files
        for (int i = 0; i < N_VECTORS; i++) begin
            string fname;
            logic [7:0] mem [0:1];
            $sformat(fname, "vectors/vec%02d_expected.hex", i);
            $readmemh(fname, mem);
            expected_class[i] = mem[0];
        end

        $display("============================================");
        $display("  SPI Slave — Vector Testbench (%0d vectors)", N_VECTORS);
        $display("  Sys clk : 100 MHz | SPI clk : 10 MHz");
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
        #50000000;
        $display("[TIMEOUT] Simulation exceeded limit — aborting");
        $finish;
    end

    // --------------------------------------------------------
    // VCD dump
    // --------------------------------------------------------
    initial begin
        $dumpfile("spi_waves.vcd");
        $dumpvars(0, tb_spi_slave_output);
    end

endmodule

`default_nettype wire