// tb_gls.sv  -  Gate-level simulation, 40 test vectors
`timescale 1ns/1ps

module tb_gls;

    // -- Parameters -------------------------------------------
    parameter N_CH            = 8;
    parameter N_SAMPLES       = 250;
    parameter ADC_WIDTH       = 8;
    parameter PKT_BYTES       = 10;
    parameter PKT_BITS        = PKT_BYTES * 8;
    parameter N_VECTORS       = 40;
    parameter TOTAL_SCAN_BITS = 864;

    parameter CLK_PERIOD  = 10;
    parameter SCAN_PERIOD = 40;
    parameter SPI_PERIOD  = 100;

    // -- DUT signals ------------------------------------------
    reg                  clk, rst_n;
    reg  [ADC_WIDTH-1:0] adc_sample;
    wire [2:0]           adc_channel;

    reg                  spi_sclk, spi_cs_n;
    wire                 spi_miso;

    reg                  scan_en, scan_clk, scan_in;

    // -- DUT (gate-level netlist) ------------------------------
    bmi_chip_top dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .adc_sample (adc_sample),
        .adc_channel(adc_channel),
        .spi_sclk   (spi_sclk),
        .spi_cs_n   (spi_cs_n),
        .spi_miso   (spi_miso),
        .scan_en    (scan_en),
        .scan_clk   (scan_clk),
        .scan_in    (scan_in)
    );

    // -- Clock ------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -- Test data --------------------------------------------
    reg [7:0] adc_mem   [0:N_CH*N_SAMPLES-1];
    reg [7:0] exp_class [0:N_VECTORS-1];
    reg [7:0] weights   [0:TOTAL_SCAN_BITS/8-1];
    reg [PKT_BITS-1:0] rx_packet;

    integer pass_count = 0;
    integer fail_count = 0;

    // -- Background ADC driver state --------------------------
    integer drv_cnt [0:N_CH-1];

    // -- Background ADC driver --------------------------------
    // Uses dut.packet_ready (a preserved top-level wire in the
    // synthesized netlist) to detect collection restart instead
    // of the RTL-internal sample_cnt registers.
    initial begin
        integer ch;
        for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
        wait (rst_n === 1'b1);
        forever begin
            @(negedge clk);

            // packet_ready pulses for one cycle when the SPI slave
            // finishes — this is the same event that resets sample_cnt
            // inside sample_collection, so resetting drv_cnt here keeps
            // the TB in sync with the RTL without needing internal access.
            if (dut.packet_ready === 1'b1)
                for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;

            ch = dut.adc_channel;
            adc_sample = adc_mem[ch * N_SAMPLES +
                                 (drv_cnt[ch] < N_SAMPLES ? drv_cnt[ch] : N_SAMPLES - 1)];
            if (drv_cnt[ch] < N_SAMPLES)
                drv_cnt[ch] = drv_cnt[ch] + 1;
        end
    end

    // -- Scan chain task --------------------------------------
    task load_weights;
        integer i, b;
        reg [TOTAL_SCAN_BITS-1:0] scan_data;
        begin
            for (i = 0; i < TOTAL_SCAN_BITS/8; i++)
                scan_data[i*8 +: 8] = weights[i];
            scan_en = 1;
            for (b = 0; b < TOTAL_SCAN_BITS; b++) begin
                scan_in = scan_data[b];
                #(SCAN_PERIOD/2) scan_clk = 1;
                #(SCAN_PERIOD/2) scan_clk = 0;
            end
            scan_en = 0;
            scan_in = 0;
        end
    endtask

    // -- SPI master -------------------------------------------
    task spi_receive;
        integer b;
        begin
            wait (dut.packet_valid);
            @(posedge clk);
            spi_cs_n = 0;
            #(SPI_PERIOD);
            rx_packet = 0;
            for (b = 0; b < PKT_BITS; b++) begin
                spi_sclk = 0;
                #(SPI_PERIOD/2);
                spi_sclk = 1;
                rx_packet = {rx_packet[PKT_BITS-2:0], spi_miso};
                #(SPI_PERIOD/2);
            end
            spi_sclk = 0;
            #(SPI_PERIOD);
            spi_cs_n = 1;
            #(SPI_PERIOD);
        end
    endtask

    // -- Result checker ---------------------------------------
    task check_result;
        input integer vec;
        reg [7:0] sync_byte, rx_class, exp;
        begin
            sync_byte = rx_packet[PKT_BITS-1 -: 8];
            if (sync_byte !== 8'hAA)
                $display("  WARNING vec%02d: bad sync byte 0x%02X", vec, sync_byte);

            rx_class = rx_packet[PKT_BITS-9 -: 8] & 8'h03;
            exp      = exp_class[vec] & 8'h03;

            if (rx_class === exp) begin
                $display("PASS vec%02d  got=%0d", vec, rx_class);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL vec%02d  got=%0d  expected=%0d", vec, rx_class, exp);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // -- Main sequence ----------------------------------------
    integer vec;

    initial begin
        rst_n    = 0;
        adc_sample = 0;
        spi_sclk = 0;
        spi_cs_n = 1;
        scan_en  = 0;
        scan_clk = 0;
        scan_in  = 0;

        repeat(4) @(posedge clk);

        $readmemh("weights.hex", weights);
        load_weights();
        $display("Weights loaded via scan chain.");
        repeat(2) @(posedge clk);

        $readmemh("vectors/all_expected.hex", exp_class);
        $readmemh("vectors/vec00_adc.hex", adc_mem);

        rst_n = 1;

        for (vec = 0; vec < N_VECTORS; vec++) begin
            @(posedge dut.window_ready);

            if (vec + 1 < N_VECTORS)
                $readmemh($sformatf("vectors/vec%02d_adc.hex", vec + 1), adc_mem);

            spi_receive();
            check_result(vec);
        end

        $display("----------------------------------------------");
        $display("GLS      %0d / %0d  PASS", pass_count, N_VECTORS);
        $display("----------------------------------------------");
        if (fail_count == 0)
            $display("ALL PASS");
        else
            $display("%0d FAILURES", fail_count);

        $finish;
    end

endmodule
