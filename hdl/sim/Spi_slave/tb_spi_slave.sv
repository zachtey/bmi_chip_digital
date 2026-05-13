`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_spi_slave.sv  
//
// Clock-synchronous SPI master — every spi_sclk and spi_cs_n
// edge is driven on a posedge of the internal clk. This makes
// the synchronizer behavior deterministic across all vectors
// (no phase drift between transactions).
//
// SPI clock period = 2 * SPI_HALF_CYC internal clock cycles.
//   Slave needs internal clk >= 2x SPI clk (CDC requirement),
//   so SPI_HALF_CYC must be >= 2. Use 50 cycles for plenty of
//   margin and easy waveform reading.
//
// Compile + run from hdl/sim/:
//   iverilog -g2012 -o spi_sim tb_spi_slave.sv ../sv/spi_slave.sv
//   vvp spi_sim
// ============================================================

module tb_spi_slave;

    // Parameters
    localparam int CLK_PERIOD   = 10;     // 100 MHz internal
    localparam int SPI_HALF_CYC = 50;     // 50 internal cycles per SPI half-period
                                          // → SPI ≈ 1 MHz
    localparam int N_VECTORS    = 40;
    localparam int PKT_BYTES    = 10;
    localparam int PKT_BITS     = PKT_BYTES * 8;

    // DUT signals
    logic                clk;
    logic                rst_n;
    logic                spi_sclk;
    logic                spi_cs_n;
    logic                spi_miso;
    logic [PKT_BITS-1:0] packet_data;
    logic                packet_valid;
    logic                packet_ready;

    // DUT
    spi_slave #(
        .PKT_BYTES (PKT_BYTES)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .spi_sclk     (spi_sclk),
        .spi_cs_n     (spi_cs_n),
        .spi_miso     (spi_miso),
        .packet_data  (packet_data),
        .packet_valid (packet_valid),
        .packet_ready (packet_ready)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Storage
    logic [63:0] scores_mem     [0:N_VECTORS*4-1];
    logic [7:0]  expected_class [0:N_VECTORS-1];

    // Tracking
    int tests_run    = 0;
    int tests_passed = 0;

    // Helper: build expected packet
    function automatic logic [PKT_BITS-1:0] build_packet(
        input logic [1:0]  cls,
        input logic [31:0] s0,
        input logic [31:0] s1,
        input logic [31:0] s2,
        input logic [31:0] s3
    );
        return {
            8'hAA,
            6'b000000, cls,
            s0[31:16],
            s1[31:16],
            s2[31:16],
            s3[31:16]
        };
    endfunction

    // ============================================================
    // Clock-synchronous SPI master
    //
    // All transitions happen on @(posedge clk). The SPI clock is
    // a divided version of the internal clock — toggles every
    // SPI_HALF_CYC internal cycles.
    //
    // Sequence per bit:
    //   half-period 1 (SCLK low):  slave updates miso_reg on the
    //                              detected falling edge
    //   half-period 2 (SCLK high): master samples MISO once it
    //                              has been stable for several
    //                              cycles (synchronizer settled)
    // ============================================================
    task automatic spi_transfer(output logic [PKT_BITS-1:0] rx_packet);
        logic [PKT_BITS-1:0] captured;
        int                  cyc;
        captured = '0;

        // Pre-condition: sclk = 1, cs_n = 1
        // Drop CS while sclk is high
        @(posedge clk); spi_cs_n = 1'b0;

        // Hold for synchronizer to detect cs_active
        // (need >= 2 cycles, use 10 for margin)
        repeat(10) @(posedge clk);

        // Drop sclk to LOW — this is the first FALLING edge
        // The slave will detect it after 2 cycles and load packet
        @(posedge clk); spi_sclk = 1'b0;

        // Wait one full half-period for slave to:
        //  - synchronize the falling edge (2 cycles)
        //  - load packet_data into shift_reg, set transmitting=1
        //  - register miso_reg <= shift_reg[79] on the SAME falling
        //    detection cycle... actually no, miso_reg only updates
        //    when transmitting=1, which is set by the load. So the
        //    first miso_reg update happens on the SECOND detected
        //    falling edge. Until then, the combinational override
        //    (cs_active && !transmitting && packet_valid) drives
        //    spi_miso = packet_data[79].
        repeat(SPI_HALF_CYC) @(posedge clk);

        // Now run 80 SPI bit cycles
        for (int b = PKT_BITS - 1; b >= 0; b--) begin
            // Rising edge — sample MISO
            @(posedge clk); spi_sclk = 1'b1;

            // Wait late in high phase so synchronizer has
            // propagated any miso_reg change from the previous
            // falling edge
            repeat(SPI_HALF_CYC - 2) @(posedge clk);

            // Sample MISO
            @(posedge clk); captured[b] = spi_miso;

            // Falling edge
            @(posedge clk); spi_sclk = 1'b0;

            // Wait full half-period for slave to update miso_reg
            repeat(SPI_HALF_CYC - 1) @(posedge clk);
        end

        // Deassert CS while sclk is low
        @(posedge clk); spi_cs_n = 1'b1;
        @(posedge clk); spi_sclk = 1'b1;   // return idle high

        // Hold for synchronizer to register CS deassertion
        repeat(10) @(posedge clk);

        rx_packet = captured;
    endtask

    // Run one vector
    task automatic run_vector(
        input int          vec_idx,
        input logic [63:0] s0_64,
        input logic [63:0] s1_64,
        input logic [63:0] s2_64,
        input logic [63:0] s3_64,
        input logic [1:0]  cls
    );
        logic [PKT_BITS-1:0] expected_packet;
        logic [PKT_BITS-1:0] rx_packet;

        tests_run++;

        expected_packet = build_packet(cls,
            s0_64[31:0], s1_64[31:0], s2_64[31:0], s3_64[31:0]);

        // Drive packet
        @(posedge clk);
        packet_data  = expected_packet;
        packet_valid = 1'b1;

        // Long settle before SPI starts
        repeat(100) @(posedge clk);

        // SPI transaction
        spi_transfer(rx_packet);

        // Wait for packet_ready
        repeat(50) @(posedge clk);

        // Deassert packet_valid
        @(posedge clk);
        packet_valid = 1'b0;

        if (rx_packet === expected_packet) begin
            $display("[PASS] Vec %02d | class=%0d  rx=%020h",
                     vec_idx, cls, rx_packet);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | class=%0d", vec_idx, cls);
            $display("       expected = %020h", expected_packet);
            $display("       got      = %020h", rx_packet);
        end

        // Long gap between vectors
        repeat(200) @(posedge clk);
    endtask

    // Main
    initial begin
        rst_n        = 1'b0;
        spi_sclk     = 1'b1;     // idle HIGH so first transition is FALLING
        spi_cs_n     = 1'b1;     // idle inactive
        packet_data  = '0;
        packet_valid = 1'b0;

        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(50) @(posedge clk);

        // Load vectors
        for (int i = 0; i < N_VECTORS; i++) begin
            string fn_scores;
            string fn_expected;
            logic [63:0] s_buf [0:3];
            logic [7:0]  e_buf [0:0];

            $sformat(fn_scores,   "vectors/vec%02d_scores.hex",   i);
            $sformat(fn_expected, "vectors/vec%02d_expected.hex", i);
            $readmemh(fn_scores,   s_buf);
            $readmemh(fn_expected, e_buf);

            scores_mem[i*4 + 0] = s_buf[0];
            scores_mem[i*4 + 1] = s_buf[1];
            scores_mem[i*4 + 2] = s_buf[2];
            scores_mem[i*4 + 3] = s_buf[3];
            expected_class[i]   = e_buf[0];
        end

        $display("============================================");
        $display("  SPI Slave (partner's) — Clock-Sync Testbench");
        $display("  Internal clk : 100 MHz");
        $display("  SPI clk      : %0d MHz", 1000 / (CLK_PERIOD * 2 * SPI_HALF_CYC));
        $display("  Packet       : 80 bits");
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

    // Watchdog
    initial begin
        #5000000000;
        $display("[TIMEOUT] Aborting");
        $finish;
    end

    // VCD
    initial begin
        $dumpfile("spi_waves.vcd");
        $dumpvars(0, tb_spi_slave);
    end

endmodule

`default_nettype wire