`timescale 1ns/1ps

module tb_output_formatter;

    //tb + dut params
    localparam integer CLK_PERIOD = 10; //tb runs on 100 MHz clock
    localparam integer N_CLASSES   = 4;
    localparam integer SCORE_WIDTH = 32;
    localparam integer PKT_BYTES   = 10;
    localparam integer PKT_BITS    = PKT_BYTES * 8;

    //dut inputs 
    logic clk, rst_n;
    logic [1:0] predicted_class;
    logic signed [SCORE_WIDTH-1:0] class_scores [0:3];
    logic decision_valid;
    logic packet_ready;
    //dut outputs
    logic [PKT_BYTES*8-1:0] packet_data;
    logic packet_valid;

    // testbench result counters
    integer pass_count;
    integer fail_count;

    //dut instantiation
    output_formatter #(
    .N_CLASSES (N_CLASSES),
    .SCORE_WIDTH (SCORE_WIDTH),
    .PKT_BYTES(PKT_BYTES)
) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .predicted_class (predicted_class),
    .class_scores   (class_scores),
    .decision_valid   (decision_valid),
    .packet_data (packet_data),
    .packet_valid (packet_valid),
    .packet_ready (packet_ready)
);

    //process 1: clock generation (100 MHz)
    initial clk = 1'b0;
    always # (CLK_PERIOD/2) clk = ~clk;

    //process 2: apply reset and test sequence, prints results
    initial begin
        rst_n        = 1'b0;
        predicted_class = 2'd0;
        decision_valid = 1'b0;
        packet_ready   = 1'b0;
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
        if (packet_data !== '0 || packet_valid !== 1'b0) begin
            $fatal(
                1,
                "Reset values incorrect: class=0x%020X valid=%b",
                packet_data,
                packet_valid
            );
        end

        // release reset away from the rising clock edge
        @(negedge clk);
        rst_n = 1'b1;

        // FMT baseline: without decision_valid or packet_ready, the reset
        // values must hold.
        check_case(
            1,
            32'sd1, 32'sd2, 32'sd3, 32'sd4,
            1'b0, 2'd0, 1'b0
        );

        // FMT-001/002/003/004: capture class 0 and verify the complete packet.
        // Each score has a recognizable upper halfword so field ordering is
        // visible in the expected 80-bit value.
        check_case(
            2,
            32'sh1234_5678,
            32'shFEDC_BA98,
            32'sh7FFF_0001,
            32'sh8000_FFFF,
            1'b0, 2'd0, 1'b1
        );

        // FMT-005: delay consumption for three clocks. Change every formatter
        // input while decision_valid is low; the pending packet must remain
        // valid and bit-for-bit stable.
        check_case(
            3,
            32'sd100, 32'sd200, 32'sd300, 32'sd400,
            1'b0, 2'd1, 1'b0
        );
        check_case(
            4,
            -32'sd1, -32'sd2, -32'sd3, -32'sd4,
            1'b0, 2'd2, 1'b0
        );
        check_case(
            5,
            32'shAAAA_0001, 32'shBBBB_0002,
            32'shCCCC_0003, 32'shDDDD_0004,
            1'b0, 2'd3, 1'b0
        );

        // FMT-006: acknowledge the pending class-0 packet. packet_valid must
        // clear while packet_data retains the most recently formatted value.
        check_case(
            6,
            32'sd10, 32'sd20, 32'sd30, 32'sd40,
            1'b1, 2'd3, 1'b0
        );

        // FMT-003: exercise the three remaining class encodings. Each valid
        // packet is followed by a ready acknowledgement.
        check_case(
            7,
            32'sh1111_0001, 32'sh2222_0002,
            32'sh3333_0003, 32'sh4444_0004,
            1'b0, 2'd1, 1'b1
        );
        check_case(
            8,
            32'sd0, 32'sd0, 32'sd0, 32'sd0,
            1'b1, 2'd0, 1'b0
        );
        check_case(
            9,
            32'sh5555_0001, 32'sh6666_0002,
            32'sh7777_0003, 32'sh8888_0004,
            1'b0, 2'd2, 1'b1
        );
        check_case(
            10,
            32'sd0, 32'sd0, 32'sd0, 32'sd0,
            1'b1, 2'd0, 1'b0
        );
        check_case(
            11,
            32'sh9999_0001, 32'shAAAA_0002,
            32'shBBBB_0003, 32'shCCCC_0004,
            1'b0, 2'd3, 1'b1
        );
        check_case(
            12,
            32'sd0, 32'sd0, 32'sd0, 32'sd0,
            1'b1, 2'd0, 1'b0
        );

        // Priority corner case beyond FMT-001..006: simultaneous decision and
        // ready stores the new packet_data but leaves packet_valid cleared.
        check_case(
            13,
            32'sh1111_2222,
            32'sh3333_4444,
            32'sh5555_6666,
            32'sh7777_8888,
            1'b1, 2'd3, 1'b1
        );

        //End of tests, print results
        $display("----------------------------------------");
        $display(
            "OUTPUT FORMATTER TEST: %0d passed, %0d failed",
            pass_count,
            fail_count
        );
        $display("----------------------------------------");

        if (fail_count != 0)
            $fatal(1, "Output Formatter regression failed");

        $display("ALL OUTPUT FORMATTER TESTS PASS");
        $finish;
    
    end

    //process 3: watch for timeout
    initial begin
        #(1000 * CLK_PERIOD);
        $fatal(1, "Output Formatter testbench timeout");
    end


    // tasks!!
    //packet_ready, predicted_class, class_scores, decision_valid applied as inputs from driver in tb
    //behavior: if decision_valid is pulsed, pack the next output to be sent on the 
    //          next clock cycle only if packet_ready is low. If we send the next
    //          packet, then we also pulse packet_valid to be high
    task automatic check_case;
    input integer test_id;

    input logic signed [SCORE_WIDTH-1:0] score0;
    input logic signed [SCORE_WIDTH-1:0] score1;
    input logic signed [SCORE_WIDTH-1:0] score2;
    input logic signed [SCORE_WIDTH-1:0] score3;

    input logic       packet_ready_in;
    input logic [1:0] predicted_class_in;
    input logic       decision_valid_in;

    logic [PKT_BITS-1:0] new_packet;
    logic [PKT_BITS-1:0] expected_packet_data;
    logic                expected_packet_valid;
    logic                case_pass;

    begin
        case_pass = 1'b1;

        // Begin with the formatter's current registered state.
        expected_packet_data  = packet_data;
        expected_packet_valid = packet_valid;

        // Construct the packet that decision_valid would capture.
        new_packet = {
            8'hAA,
            6'b0, predicted_class_in,
            score0[SCORE_WIDTH-1 -: 16],
            score1[SCORE_WIDTH-1 -: 16],
            score2[SCORE_WIDTH-1 -: 16],
            score3[SCORE_WIDTH-1 -: 16]
        };

        // Apply the formatter's specified priority.
        if (decision_valid_in) begin
            expected_packet_data  = new_packet;
            expected_packet_valid = 1'b1;
        end

        if (packet_ready_in)
            expected_packet_valid = 1'b0;

        // Drive the DUT away from its active edge.
        @(negedge clk);
        class_scores[0] = score0;
        class_scores[1] = score1;
        class_scores[2] = score2;
        class_scores[3] = score3;
        decision_valid  = decision_valid_in;
        packet_ready    = packet_ready_in;
        predicted_class = predicted_class_in;

        // DUT registers the result.
        @(posedge clk);
        #1;

        if (packet_valid !== expected_packet_valid) begin
            $display(
                "FAIL test %0d: packet_valid got=%b expected=%b",
                test_id,
                packet_valid,
                expected_packet_valid
            );
            case_pass = 1'b0;
        end

        if (packet_data !== expected_packet_data) begin
            $display(
                "FAIL test %0d: packet_data got=0x%020X expected=0x%020X",
                test_id,
                packet_data,
                expected_packet_data
            );
            case_pass = 1'b0;
        end

        if (case_pass) begin
            $display(
                "PASS test %0d: valid=%b packet=0x%020X",
                test_id,
                packet_valid,
                packet_data
            );
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
        end
    end
endtask
endmodule


/*
output_formatter.sv
Packs predicted_class and class_scores into a 10-byte SPI packet.

Packet layout (MSB-first, byte 0 sent first):
  Byte 0:    0xAA               sync header
  Byte 1:    {6'b0, class[1:0]} predicted class index
  Bytes 2-3: score[0][31:16]    PG-LF
  Bytes 4-5: score[1][31:16]    PG-HF
  Bytes 6-7: score[2][31:16]    SG-LF
  Bytes 8-9: score[3][31:16]    SG-HF

Handshake: decision_valid (1-cycle pulse) sets packet_valid (level);
           packet_ready   (1-cycle pulse from spi_slave) clears it.

module output_formatter #(
    parameter N_CLASSES   = 4,
    parameter SCORE_WIDTH = 32,
    parameter PKT_BYTES   = 10
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [1:0]                    predicted_class,
    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1],
    input  wire                          decision_valid,

    output logic [PKT_BYTES*8-1:0]       packet_data,
    output logic                         packet_valid, //to spi slave; packet_data has complete packet that has not been consumed
    input  wire                          packet_ready //from spi slave; the spi slave finished consuming that packet
);
*/
