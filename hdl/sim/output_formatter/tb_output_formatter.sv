`default_nettype none
`timescale 1ns/1ps

// ============================================================
// tb_output_formatter.sv  
//
// Drives class + 4 scores into the formatter, pulses
// decision_valid, then checks the 80-bit packet matches:
//
//   Byte 0:    8'hAA              (sync)
//   Byte 1:    {6'b0, class[1:0]} (class)
//   Byte 2-3:  score[0][31:16]    (upper half of score 0)
//   Byte 4-5:  score[1][31:16]
//   Byte 6-7:  score[2][31:16]
//   Byte 8-9:  score[3][31:16]
//
// Then pulses packet_ready and verifies packet_valid drops.
//
// Compile + run from hdl/sim/:
//   iverilog -g2012 -o fmt_sim tb_output_formatter.sv ../sv/output_formatter.sv
//   vvp fmt_sim
//   surfer fmt_waves.vcd
// ============================================================

module tb_output_formatter;

    // Parameters
    localparam int CLK_PERIOD  = 10;
    localparam int N_VECTORS   = 40;
    localparam int N_CLASSES   = 4;
    localparam int SCORE_WIDTH = 32;
    localparam int PKT_BYTES   = 10;
    localparam int PKT_BITS    = PKT_BYTES * 8;

    // DUT signals
    logic                          clk;
    logic                          rst_n;
    logic [1:0]                    predicted_class;
    logic signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1];
    logic                          decision_valid;
    logic [PKT_BITS-1:0]           packet_data;
    logic                          packet_valid;
    logic                          packet_ready;

    // DUT
    output_formatter #(
        .N_CLASSES   (N_CLASSES),
        .SCORE_WIDTH (SCORE_WIDTH),
        .PKT_BYTES   (PKT_BYTES)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .predicted_class (predicted_class),
        .class_scores    (class_scores),
        .decision_valid  (decision_valid),
        .packet_data     (packet_data),
        .packet_valid    (packet_valid),
        .packet_ready    (packet_ready)
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

    // Task : run one vector
    task automatic run_vector(
        input int          vec_idx,
        input logic [63:0] s0_64,
        input logic [63:0] s1_64,
        input logic [63:0] s2_64,
        input logic [63:0] s3_64,
        input logic [1:0]  cls
    );
        logic [PKT_BITS-1:0] expected_packet;
        logic [PKT_BITS-1:0] got_packet;
        int                  timeout;

        tests_run++;

        expected_packet = build_packet(cls,
            s0_64[31:0], s1_64[31:0], s2_64[31:0], s3_64[31:0]);

        // Drive scores + class
        @(posedge clk); #1;
        predicted_class = cls;
        class_scores[0] = s0_64[31:0];
        class_scores[1] = s1_64[31:0];
        class_scores[2] = s2_64[31:0];
        class_scores[3] = s3_64[31:0];
        decision_valid  = 1'b1;
        @(posedge clk); #1;
        decision_valid  = 1'b0;

        // Wait for packet_valid
        timeout = 0;
        while (!packet_valid && timeout < 10) begin
            @(posedge clk); #1;
            timeout++;
        end

        if (!packet_valid) begin
            $display("[FAIL] Vec %02d | packet_valid never asserted", vec_idx);
            return;
        end

        got_packet = packet_data;

        if (got_packet === expected_packet) begin
            $display("[PASS] Vec %02d | class=%0d  packet=%020h",
                     vec_idx, cls, got_packet);
            tests_passed++;
        end else begin
            $display("[FAIL] Vec %02d | class=%0d", vec_idx, cls);
            $display("       expected = %020h", expected_packet);
            $display("       got      = %020h", got_packet);
        end

        // Simulate SPI completing — pulse packet_ready
        @(posedge clk); #1;
        packet_ready = 1'b1;
        @(posedge clk); #1;
        packet_ready = 1'b0;

        repeat(3) @(posedge clk);
    endtask

    // Main
    initial begin
        // Init
        rst_n           = 1'b0;
        predicted_class = 2'b00;
        decision_valid  = 1'b0;
        packet_ready    = 1'b0;
        for (int i = 0; i < N_CLASSES; i++) class_scores[i] = '0;

        repeat(5) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

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
        $display("  Output Formatter (partner's version) — Vector Testbench");
        $display("  Packet: 80 bits = sync + class + 4 score upper-halves");
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
        #10000000;
        $display("[TIMEOUT] Aborting");
        $finish;
    end

    // VCD
    initial begin
        $dumpfile("fmt_waves.vcd");
        $dumpvars(0, tb_output_formatter);
    end

endmodule

`default_nettype wire