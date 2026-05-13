`timescale 1ns/1ps

module tb_bmi_chip_top;

    parameter N_CH      = 8;
    parameter N_SAMPLES = 250;
    parameter ADC_WIDTH = 8;
    parameter N_OUT     = 4;
    parameter PKT_BYTES = 10;
    parameter PKT_BITS  = PKT_BYTES * 8;
    parameter N_VECTORS = 40;
    parameter TOTAL_SCAN_BITS = 864;

    parameter CLK_PERIOD  = 10;
    parameter SCAN_PERIOD = 40;
    parameter SPI_PERIOD  = 100;

    reg                    clk, rst_n;
    reg  [ADC_WIDTH-1:0]   adc_sample;
    reg                    adc_valid;
    reg  [2:0]             adc_channel;
    reg                    spi_sclk, spi_cs_n;
    wire                   spi_miso;
    reg                    scan_en, scan_clk, scan_in;

    bmi_chip_top dut (
        .clk(clk), .rst_n(rst_n),
        .adc_sample(adc_sample), .adc_valid(adc_valid),
        .adc_channel(adc_channel),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n), .spi_miso(spi_miso),
        .scan_en(scan_en), .scan_clk(scan_clk), .scan_in(scan_in)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [7:0] adc_mem   [0:N_CH*N_SAMPLES-1];
    reg [7:0] exp_class [0:N_VECTORS-1];
    reg [7:0] weights   [0:TOTAL_SCAN_BITS/8-1];
    reg [PKT_BITS-1:0] rx_packet;

    integer pass_count = 0;
    integer fail_count = 0;

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

    task drive_adc_window;
        integer ch, s;
        begin
            for (ch = 0; ch < N_CH; ch++) begin
                for (s = 0; s < N_SAMPLES; s++) begin
                    @(posedge clk); #1;
                    adc_sample  = adc_mem[ch*N_SAMPLES + s];
                    adc_channel = ch[2:0];
                    adc_valid   = 1;
                    @(posedge clk); #1;
                    adc_valid   = 0;
                end
            end
        end
    endtask

    task spi_receive;
        integer b;
        begin
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

    integer vec;
    reg [7:0] rx_class;
    reg [7:0] exp;
    reg [7:0] sync_byte;

    initial begin
        $dumpfile("top/top_gl_waves.vcd");
        $dumpvars(0, tb_bmi_chip_top);

        rst_n      = 0;
        adc_valid  = 0;
        adc_sample = 0;
        adc_channel= 0;
        spi_sclk   = 0;
        spi_cs_n   = 1;
        scan_en    = 0;
        scan_clk   = 0;
        scan_in    = 0;

        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $readmemh("weights.hex", weights);
        load_weights();
        $display("Weights loaded via scan chain.");
        repeat(4) @(posedge clk);

        $readmemh("vectors/all_expected.hex", exp_class);

        for (vec = 0; vec < N_VECTORS; vec++) begin

            $readmemh($sformatf("vectors/vec%02d_adc.hex", vec), adc_mem);

            drive_adc_window();

            // Each sample takes 2 cycles (valid + deassert)
            // sample_collection: ~4000 cycles
            // SBP: ~2000 cycles
            // MLP: ~110 cycles
            // argmax + formatter: ~5 cycles
            repeat(6500) @(posedge clk);

            spi_receive();

            sync_byte = rx_packet[PKT_BITS-1 -: 8];
            if (sync_byte !== 8'hAA)
                $display("  WARNING vec%02d: bad sync byte 0x%02X", vec, sync_byte);

            rx_class = rx_packet[PKT_BITS-9 -: 8] & 8'h03;
            exp      = exp_class[vec] & 8'h03;

            if (rx_class === exp) begin
                $display("PASS vec%02d: class=%0d", vec, rx_class);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL vec%02d: got class=%0d, expected=%0d",
                         vec, rx_class, exp);
                fail_count = fail_count + 1;
            end

            repeat(10) @(posedge clk);
        end

        $display("----------------------------------------");
        $display("GATE-LEVEL RESULTS: %0d/%0d pass", pass_count, N_VECTORS);
        $display("----------------------------------------");

        if (fail_count == 0)
            $display("ALL PASS");
        else
            $display("%0d FAILURES", fail_count);

        $finish;
    end

endmodule