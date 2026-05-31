// ============================================================
// tb_bmi_chip_top.sv  -  RTL simulation, 40 test vectors
//
// ADC model: a background process continuously presents
// adc_mem[adc_channel * N_SAMPLES + drv_cnt[ch]] at every
// negedge, exactly as a real SAR ADC behind an 8:1 MUX would.
// When packet_ready fires (collection restart), it auto-resets
// drv_cnt to 0 to stay in sync with sample_collection's
// sample_cnt reset, eliminating stale captures between vectors.
// ============================================================
`timescale 1ns/1ps

module tb_bmi_chip_top;

    // -- Parameters -------------------------------------------
    parameter N_CH            = 8;
    parameter N_SAMPLES       = 250;
    parameter ADC_WIDTH       = 8;
    parameter PKT_BYTES       = 10;
    parameter PKT_BITS        = PKT_BYTES * 8;   // 80
    parameter N_VECTORS       = 40;
    parameter TOTAL_SCAN_BITS = 864;

    parameter CLK_PERIOD  = 10;    // 10 ns -> 100 MHz
    parameter SCAN_PERIOD = 40;    // 40 ns scan clock
    parameter SPI_PERIOD  = 100;   // 100 ns SPI clock (10 MHz)

    // -- DUT signals ------------------------------------------
    reg                  clk, rst_n;
    reg  [ADC_WIDTH-1:0] adc_sample;
    wire [2:0]           adc_channel;   // output from DUT -> drives analog MUX

    reg                  spi_sclk, spi_cs_n;
    wire                 spi_miso;

    reg                  scan_en, scan_clk, scan_in;

    // -- DUT --------------------------------------------------
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
    reg [7:0] adc_mem   [0:N_CH*N_SAMPLES-1];   // current/next vector ADC samples
    reg [7:0] exp_class [0:N_VECTORS-1];          // expected class per vector
    reg [7:0] weights   [0:TOTAL_SCAN_BITS/8-1];  // MLP weights

    reg [PKT_BITS-1:0] rx_packet;

    integer pass_count = 0;
    integer fail_count = 0;

    // -- Background ADC driver state --------------------------
    // Per-channel sample index, mirrors sample_collection.sample_cnt.
    // Declared at module scope so both initial blocks can access it.
    integer drv_cnt [0:N_CH-1];

    // -- Background ADC driver --------------------------------
    // Runs continuously after reset. Presents adc_mem[ch*N_SAMPLES+drv_cnt[ch]]
    // at every negedge. On collection restart (sample_cnt all reset to 0),
    // resets drv_cnt to 0 so it stays in sync with the RTL.
    initial begin
        integer ch;
        for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
        wait (rst_n === 1'b1);
        forever begin
            @(negedge clk);

            // Detect collection restart: packet_ready fired at the previous
            // posedge and reset all sample_cnt to 0. Re-sync drv_cnt to 0
            // so the first real sample lands at index 0 in sample_window.
            begin : restart_check
                integer all_zero;
                all_zero = 1;
                for (ch = 0; ch < N_CH; ch++)
                    if (dut.u_sc.sample_cnt[ch] !== 0) all_zero = 0;
                if (all_zero && (dut.u_sc.collecting === 1'b1))
                    for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
            end

            // Present the correct sample for whatever channel the MUX selects.
            // Clamp at N_SAMPLES-1 to avoid out-of-bounds reads when the
            // background driver outruns collection during the paused window.
            ch = dut.adc_channel;
            adc_sample = adc_mem[ch * N_SAMPLES +
                                 (drv_cnt[ch] < N_SAMPLES ? drv_cnt[ch] : N_SAMPLES - 1)];
            if (drv_cnt[ch] < N_SAMPLES)
                drv_cnt[ch] = drv_cnt[ch] + 1;
        end
    end

    // -- Scan chain: load weights before releasing reset -------
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
    // Blocks until the DUT's output_formatter raises packet_valid,
    // then clocks the 80-bit packet out over SPI Mode 0.
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
            // packet_ready has now fired; sample_collection.resume restarted
            // collection. The background ADC driver will detect the restart
            // at the next negedge and reset drv_cnt to 0.
        end
    endtask

    // -- Result checker ----------------------------------------
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
                $display("  scores(32b): PG-LF=%0d  PG-HF=%0d  SG-LF=%0d  SG-HF=%0d",
                         $signed(dut.class_scores[0]),
                         $signed(dut.class_scores[1]),
                         $signed(dut.class_scores[2]),
                         $signed(dut.class_scores[3]));
                $display("  sbp_features(hex): %02X %02X %02X %02X %02X %02X %02X %02X",
                         dut.u_sbp.sbp_features[0], dut.u_sbp.sbp_features[1],
                         dut.u_sbp.sbp_features[2], dut.u_sbp.sbp_features[3],
                         dut.u_sbp.sbp_features[4], dut.u_sbp.sbp_features[5],
                         dut.u_sbp.sbp_features[6], dut.u_sbp.sbp_features[7]);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // -- Main sequence -----------------------------------------
    integer vec;

    initial begin
        rst_n      = 0;
        adc_sample = 0;
        spi_sclk   = 0;
        spi_cs_n   = 1;
        scan_en    = 0;
        scan_clk   = 0;
        scan_in    = 0;

        repeat(4) @(posedge clk);

        // Load MLP weights through the scan chain BEFORE releasing reset.
        $readmemh("weights.hex", weights);
        load_weights();
        $display("Weights loaded via scan chain.");
        repeat(2) @(posedge clk);

        $readmemh("vectors/all_expected.hex", exp_class);

        // Pre-load the first vector so the background driver has valid data
        // from the very first posedge after reset.
        $readmemh("vectors/vec00_adc.hex", adc_mem);

        // Release reset. adc_channel immediately begins cycling 0->7->0.
        // The background driver wakes up at the next negedge and starts
        // presenting vec00 samples.
        rst_n = 1;

        for (vec = 0; vec < N_VECTORS; vec++) begin
            // Wait for the RTL to collect all N_CH*N_SAMPLES samples.
            // The background driver drove them continuously, one per negedge.
            @(posedge dut.window_ready);

            // Preload the NEXT vector immediately, while the SBP->MLP->SPI
            // pipeline is still running (collecting is paused, so the RTL
            // ignores adc_sample until packet_ready fires).
            // When collection restarts, the background driver's drv_cnt will
            // be reset to 0 and it will present vec+1 samples from index 0.
            if (vec + 1 < N_VECTORS)
                $readmemh($sformatf("vectors/vec%02d_adc.hex", vec + 1), adc_mem);

            // Receive the SPI packet (internally waits for packet_valid).
            // packet_ready fires during this call; collection restarts.
            spi_receive();
            check_result(vec);
        end

        $display("----------------------------------------------");
        $display("RTL SIM  %0d / %0d  PASS", pass_count, N_VECTORS);
        $display("----------------------------------------------");
        if (fail_count == 0)
            $display("ALL PASS");
        else
            $display("%0d FAILURES", fail_count);

        $finish;
    end

endmodule
