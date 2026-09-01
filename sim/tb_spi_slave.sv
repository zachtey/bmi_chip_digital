`timescale 1ns/1ps

// Black-box protocol testbench for spi_slave.
//
// The stimulus acts as an external SPI Mode-0 master:
//   * SCLK idles low.
//   * The master samples MISO on each rising SCLK edge.
//   * The DUT changes MISO in response to falling SCLK edges.
//
// spi_sclk and spi_cs_n are asynchronous inputs to the DUT and are sampled by
// the system clock. Tests therefore include different phase offsets and only
// use SCLK periods that satisfy the documented f_clk >= 8*f_spi requirement.
module tb_spi_slave;

    localparam integer CLK_PERIOD = 10;
    localparam integer PKT_BYTES  = 10;
    localparam integer PKT_BITS   = PKT_BYTES * 8;

    logic clk;
    logic rst_n;
    logic spi_sclk;
    logic spi_cs_n;
    logic spi_miso;
    logic [PKT_BITS-1:0] packet_data;
    logic packet_valid;
    logic packet_ready;

    integer pass_count;
    integer fail_count;
    integer ready_pulse_count;

    spi_slave #(
        .PKT_BYTES(PKT_BYTES)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n),
        .spi_miso(spi_miso),
        .packet_data(packet_data), .packet_valid(packet_valid),
        .packet_ready(packet_ready)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Count completion pulses independently of task timing.
    always @(posedge clk) begin
        #1;
        if (packet_ready === 1'b1)
            ready_pulse_count = ready_pulse_count + 1;
    end

    task automatic record_result;
        input integer test_id;
        input logic case_pass;
        input [8*80-1:0] description;
        begin
            if (case_pass) begin
                $display("PASS test %0d: %0s", test_id, description);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL test %0d: %0s", test_id, description);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Wait long enough for asynchronous CS to pass through the synchronizer
    // and for the packet to be preloaded before the first sample edge.
    task automatic select_slave;
        input integer phase_delay;
        begin
            #(phase_delay);
            spi_sclk = 1'b0;
            spi_cs_n = 1'b0;
            repeat (4) @(posedge clk);
            #1;
        end
    endtask

    task automatic deselect_slave;
        begin
            spi_sclk = 1'b0;
            spi_cs_n = 1'b1;
            repeat (4) @(posedge clk);
            #1;
        end
    endtask

    // Receive one complete packet and check the first-bit setup, full bit
    // ordering, lack of early completion, and exactly one ready pulse.
    task automatic check_transfer;
        input integer test_id;
        input logic [PKT_BITS-1:0] stimulus_packet;
        input integer spi_period;
        input integer phase_delay;
        integer bit_index;
        integer ready_before;
        integer half_period;
        logic [PKT_BITS-1:0] received_packet;
        logic case_pass;
        begin
            case_pass       = 1'b1;
            half_period     = spi_period/2;
            received_packet = '0;
            packet_data     = stimulus_packet;
            packet_valid    = 1'b1;
            ready_before    = ready_pulse_count;

            select_slave(phase_delay);

            // CPHA=0 requires the first bit to be valid before rising SCLK.
            if (spi_miso !== stimulus_packet[PKT_BITS-1]) begin
                $display("  first MISO bit got=%b expected=%b",
                         spi_miso, stimulus_packet[PKT_BITS-1]);
                case_pass = 1'b0;
            end

            for (bit_index = 0; bit_index < PKT_BITS;
                 bit_index = bit_index + 1) begin
                #(half_period);
                spi_sclk = 1'b1;
                #1;
                received_packet = {
                    received_packet[PKT_BITS-2:0], spi_miso
                };

                if ((bit_index < PKT_BITS-1) &&
                    (ready_pulse_count != ready_before)) begin
                    $display("  packet_ready asserted before bit 80");
                    case_pass = 1'b0;
                end

                #(half_period-1);
                spi_sclk = 1'b0;
            end

            // Allow the synchronized final edge to reach the FSM.
            repeat (5) @(posedge clk);
            #1;

            if (received_packet !== stimulus_packet) begin
                $display("  received=0x%020h expected=0x%020h",
                         received_packet, stimulus_packet);
                case_pass = 1'b0;
            end
            if (ready_pulse_count != ready_before + 1) begin
                $display("  ready pulses got=%0d expected=1",
                         ready_pulse_count-ready_before);
                case_pass = 1'b0;
            end
            if (packet_ready !== 1'b0) begin
                $display("  packet_ready was not a one-cycle pulse");
                case_pass = 1'b0;
            end

            deselect_slave();
            packet_valid = 1'b0;
            record_result(test_id, case_pass,
                          "80-bit Mode-0 transfer and completion");
        end
    endtask

    // Clock only part of a packet, release CS, and prove no completion pulse.
    task automatic check_abort;
        input integer test_id;
        input integer bits_to_send;
        integer bit_index;
        integer ready_before;
        logic case_pass;
        begin
            case_pass    = 1'b1;
            packet_data  = 80'hAA_01_1234_5678_9ABC_DEF0;
            packet_valid = 1'b1;
            ready_before = ready_pulse_count;
            select_slave(0);

            for (bit_index = 0; bit_index < bits_to_send;
                 bit_index = bit_index + 1) begin
                #40 spi_sclk = 1'b1;
                #40 spi_sclk = 1'b0;
            end

            deselect_slave();
            if (ready_pulse_count != ready_before) begin
                $display("  abort after %0d bits incorrectly completed",
                         bits_to_send);
                case_pass = 1'b0;
            end
            if (spi_miso !== 1'b0) begin
                $display("  MISO did not return low after abort");
                case_pass = 1'b0;
            end
            packet_valid = 1'b0;
            record_result(test_id, case_pass,
                          "early CS deassert aborts without ready");
        end
    endtask

    task automatic check_inactive_clocks;
        input integer test_id;
        integer edge_index;
        integer ready_before;
        logic case_pass;
        begin
            case_pass    = 1'b1;
            ready_before = ready_pulse_count;
            packet_data  = 80'hAA_02_FFFF_0000_5555_AAAA;
            packet_valid = 1'b1;
            spi_cs_n     = 1'b1;

            for (edge_index = 0; edge_index < 100; edge_index = edge_index + 1)
                #20 spi_sclk = ~spi_sclk;
            spi_sclk = 1'b0;
            repeat (4) @(posedge clk);
            #1;

            if (ready_pulse_count != ready_before || spi_miso !== 1'b0)
                case_pass = 1'b0;
            packet_valid = 1'b0;
            record_result(test_id, case_pass,
                          "SCLK ignored while CS is inactive");
        end
    endtask

    task automatic check_reset_during_transfer;
        input integer test_id;
        integer bit_index;
        integer ready_before;
        logic case_pass;
        begin
            case_pass    = 1'b1;
            ready_before = ready_pulse_count;
            packet_data  = 80'hAA_03_1111_2222_3333_4444;
            packet_valid = 1'b1;
            select_slave(0);

            for (bit_index = 0; bit_index < 23; bit_index = bit_index + 1) begin
                #40 spi_sclk = 1'b1;
                #40 spi_sclk = 1'b0;
            end

            #7 rst_n = 1'b0;
            #1;
            if (packet_ready !== 1'b0 || spi_miso !== 1'b0)
                case_pass = 1'b0;
            #12 rst_n = 1'b1;
            deselect_slave();

            if (ready_pulse_count != ready_before)
                case_pass = 1'b0;
            packet_valid = 1'b0;
            record_result(test_id, case_pass,
                          "asynchronous reset aborts active transfer");
        end
    endtask

    task automatic check_valid_low;
        input integer test_id;
        integer bit_index;
        integer ready_before;
        logic case_pass;
        begin
            case_pass    = 1'b1;
            ready_before = ready_pulse_count;
            packet_valid = 1'b0;
            packet_data  = 80'hAA_00_1234_5678_9ABC_DEF0;
            select_slave(0);

            for (bit_index = 0; bit_index < PKT_BITS; bit_index = bit_index + 1) begin
                #40 spi_sclk = 1'b1;
                #40 spi_sclk = 1'b0;
            end
            deselect_slave();

            if (ready_pulse_count != ready_before || spi_miso !== 1'b0)
                case_pass = 1'b0;
            record_result(test_id, case_pass,
                          "transfer cannot start while packet_valid is low");
        end
    endtask

    initial begin
        rst_n            = 1'b0;
        spi_sclk         = 1'b0;
        spi_cs_n         = 1'b1;
        packet_data      = '0;
        packet_valid     = 1'b0;
        pass_count       = 0;
        fail_count       = 0;
        ready_pulse_count = 0;

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (3) @(posedge clk);

        check_transfer(1, 80'hAA_00_1234_FEDC_7FFF_8000, 80, 0);
        check_transfer(2, 80'hD5_A3_0123_4567_89AB_CDEF, 100, 3);
        check_transfer(3, 80'h80_01_FF00_55AA_0F0F_F0F0, 120, 7);

        check_abort(4, 1);
        check_abort(5, 17);
        check_abort(6, 79);
        check_inactive_clocks(7);
        check_reset_during_transfer(8);
        check_valid_low(9);

        // A successful transfer after abort/reset cases also proves recovery
        // and supplies the second half of back-to-back packet coverage.
        check_transfer(10, 80'hAA_03_1111_2222_3333_4444, 80, 5);
        check_transfer(11, 80'hAA_02_ABCD_0000_8000_7FFF, 80, 1);

        $display("------------------------------------------");
        $display("SPI SLAVE TEST: %0d passed, %0d failed",
                 pass_count, fail_count);
        $display("------------------------------------------");
        if (fail_count != 0)
            $fatal(1, "SPI slave regression failed");
        $display("ALL SPI SLAVE TESTS PASS");
        $finish;
    end

    initial begin
        #(2000000);
        $fatal(1, "SPI slave testbench timeout");
    end

endmodule
