// tb_bmi_chip_top.sv  -  RTL simulation, 40 test vectors
//
// ADC model: a background process continuously presents
// adc_mem[adc_channel * N_SAMPLES + drv_cnt[ch]] at every
// negedge, exactly as a real SAR ADC behind an 8:1 MUX would.
// When packet_ready fires (collection restart), it auto-resets
// drv_cnt to 0 to stay in sync with sample_collection's
// sample_cnt reset, eliminating stale captures between vectors.

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

    // Expected intermediate results for the vector currently being checked.
    reg [7:0]         exp_sbp    [0:N_CH-1];
    reg signed [63:0] exp_scores [0:3];

    reg [PKT_BITS-1:0] rx_packet;

    integer pass_count = 0;
    integer fail_count = 0;

    // -- Top-level pipeline monitor ---------------------------
    localparam integer FRONTEND_TO_MLP_DONE = 122;
    localparam integer MLP_TO_DECISION       = 1;
    localparam integer DECISION_TO_PACKET    = 1;

    integer pipeline_error_count;
    integer cycle_count;
    integer frontend_done_cycle;
    integer mlp_done_cycle;
    integer decision_cycle;
    integer frontend_event_count;
    integer mlp_event_count;
    integer decision_event_count;
    integer packet_event_count;
    integer ready_event_count;

    reg pipeline_checks_active;
    reg previous_frontend_done;
    reg previous_mlp_done;
    reg previous_decision_valid;
    reg previous_packet_ready;
    reg previous_packet_valid;
    reg awaiting_mlp;
    reg awaiting_decision;
    reg awaiting_packet;
    reg result_pending;
    reg ready_seen;
    reg packet_hold_active;
    reg [PKT_BITS-1:0] held_packet;

    // Observe signals after the DUT's nonblocking assignments settle. These
    // checks verify the connections between blocks, complementing the unit
    // tests that verify each block's internal behavior.
    always @(posedge clk) begin : pipeline_monitor
        integer monitor_idx;
        #1;
        cycle_count = cycle_count + 1;

        if (!rst_n) begin
            previous_frontend_done = 1'b0;
            previous_mlp_done      = 1'b0;
            previous_decision_valid = 1'b0;
            previous_packet_ready  = 1'b0;
            previous_packet_valid  = 1'b0;
            awaiting_mlp           = 1'b0;
            awaiting_decision      = 1'b0;
            awaiting_packet        = 1'b0;
            result_pending         = 1'b0;
            ready_seen             = 1'b0;
            packet_hold_active     = 1'b0;
        end else if (pipeline_checks_active) begin
            // All event outputs except packet_valid are specified as pulses.
            if (dut.sbp_done && previous_frontend_done) begin
                $display("PIPELINE FAIL: features_done wider than one cycle");
                pipeline_error_count = pipeline_error_count + 1;
            end
            if (dut.mlp_done && previous_mlp_done) begin
                $display("PIPELINE FAIL: mlp_done wider than one cycle");
                pipeline_error_count = pipeline_error_count + 1;
            end
            if (dut.decision_valid && previous_decision_valid) begin
                $display("PIPELINE FAIL: decision_valid wider than one cycle");
                pipeline_error_count = pipeline_error_count + 1;
            end
            if (dut.packet_ready && previous_packet_ready) begin
                $display("PIPELINE FAIL: packet_ready wider than one cycle");
                pipeline_error_count = pipeline_error_count + 1;
            end

            if (dut.sbp_done) begin
                frontend_event_count = frontend_event_count + 1;
                if (result_pending || awaiting_mlp) begin
                    $display("PIPELINE FAIL: new frontend result overwrote pending work");
                    pipeline_error_count = pipeline_error_count + 1;
                end
                if (!dut.u_mlp.weights_loaded) begin
                    $display("PIPELINE FAIL: frontend completed before weights_loaded");
                    pipeline_error_count = pipeline_error_count + 1;
                end
                for (monitor_idx = 0; monitor_idx < N_CH;
                     monitor_idx = monitor_idx + 1)
                    if (^dut.sbp_features[monitor_idx] === 1'bx) begin
                        $display("PIPELINE FAIL: X/Z in valid SBP feature %0d",
                                 monitor_idx);
                        pipeline_error_count = pipeline_error_count + 1;
                    end
                frontend_done_cycle = cycle_count;
                awaiting_mlp = 1'b1;
                result_pending = 1'b1;
                ready_seen = 1'b0;
            end

            if (dut.mlp_done) begin
                mlp_event_count = mlp_event_count + 1;
                if (!awaiting_mlp) begin
                    $display("PIPELINE FAIL: mlp_done without frontend result");
                    pipeline_error_count = pipeline_error_count + 1;
                end else if ((cycle_count-frontend_done_cycle) !=
                             FRONTEND_TO_MLP_DONE) begin
                    $display("PIPELINE FAIL: frontend->MLP latency got=%0d expected=%0d",
                             cycle_count-frontend_done_cycle,
                             FRONTEND_TO_MLP_DONE);
                    pipeline_error_count = pipeline_error_count + 1;
                end
                for (monitor_idx = 0; monitor_idx < 4;
                     monitor_idx = monitor_idx + 1)
                    if (^dut.class_scores[monitor_idx] === 1'bx) begin
                        $display("PIPELINE FAIL: X/Z in valid score %0d", monitor_idx);
                        pipeline_error_count = pipeline_error_count + 1;
                    end
                awaiting_mlp = 1'b0;
                awaiting_decision = 1'b1;
                mlp_done_cycle = cycle_count;
            end

            if (dut.decision_valid) begin
                decision_event_count = decision_event_count + 1;
                if (!awaiting_decision) begin
                    $display("PIPELINE FAIL: decision without MLP completion");
                    pipeline_error_count = pipeline_error_count + 1;
                end else if ((cycle_count-mlp_done_cycle) != MLP_TO_DECISION) begin
                    $display("PIPELINE FAIL: MLP->decision latency got=%0d expected=%0d",
                             cycle_count-mlp_done_cycle, MLP_TO_DECISION);
                    pipeline_error_count = pipeline_error_count + 1;
                end
                if (^dut.predicted_class === 1'bx) begin
                    $display("PIPELINE FAIL: X/Z in valid predicted class");
                    pipeline_error_count = pipeline_error_count + 1;
                end
                awaiting_decision = 1'b0;
                awaiting_packet = 1'b1;
                decision_cycle = cycle_count;
            end

            // packet_valid is held until consumption, so count only its rise.
            if (dut.packet_valid && !previous_packet_valid) begin
                packet_event_count = packet_event_count + 1;
                if (!awaiting_packet) begin
                    $display("PIPELINE FAIL: packet without decision");
                    pipeline_error_count = pipeline_error_count + 1;
                end else if ((cycle_count-decision_cycle) != DECISION_TO_PACKET) begin
                    $display("PIPELINE FAIL: decision->packet latency got=%0d expected=%0d",
                             cycle_count-decision_cycle, DECISION_TO_PACKET);
                    pipeline_error_count = pipeline_error_count + 1;
                end
                if (^dut.packet_data === 1'bx) begin
                    $display("PIPELINE FAIL: X/Z in valid packet");
                    pipeline_error_count = pipeline_error_count + 1;
                end
                held_packet = dut.packet_data;
                packet_hold_active = 1'b1;
                awaiting_packet = 1'b0;
            end

            if (packet_hold_active && dut.packet_valid &&
                (dut.packet_data !== held_packet)) begin
                $display("PIPELINE FAIL: packet changed while valid and pending");
                pipeline_error_count = pipeline_error_count + 1;
            end

            if (dut.packet_ready) begin
                ready_event_count = ready_event_count + 1;
                if (!result_pending || !dut.packet_valid) begin
                    $display("PIPELINE FAIL: packet_ready without pending valid packet");
                    pipeline_error_count = pipeline_error_count + 1;
                end
                ready_seen = 1'b1;
            end

            if (packet_hold_active && !dut.packet_valid)
                packet_hold_active = 1'b0;

            // The frontend must not collect another sample until a completed
            // SPI transaction supplies packet_ready.
            if (result_pending &&
                (dut.u_frontend.state === dut.u_frontend.COLLECT)) begin
                if (!ready_seen) begin
                    $display("PIPELINE FAIL: frontend resumed before packet_ready");
                    pipeline_error_count = pipeline_error_count + 1;
                end else begin
                    if (dut.adc_channel !== '0) begin
                        $display("PIPELINE FAIL: frontend did not resume at channel 0");
                        pipeline_error_count = pipeline_error_count + 1;
                    end
                    result_pending = 1'b0;
                    ready_seen = 1'b0;
                end
            end

            previous_frontend_done = dut.sbp_done;
            previous_mlp_done = dut.mlp_done;
            previous_decision_valid = dut.decision_valid;
            previous_packet_ready = dut.packet_ready;
            previous_packet_valid = dut.packet_valid;
        end
    end

    // -- Background ADC driver state --------------------------
    // Per-channel sample index, mirrors the streaming frontend counters.
    // Declared at module scope so both initial blocks can access it.
    integer drv_cnt [0:N_CH-1];
    integer frontend_was_paused;

    // -- Background ADC driver --------------------------------
    // Runs continuously after reset. Presents adc_mem[ch*N_SAMPLES+drv_cnt[ch]]
    // at every negedge. On collection restart (sample_cnt all reset to 0),
    // resets drv_cnt to 0 so it stays in sync with the RTL.
    initial begin
        integer ch;
        for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
        frontend_was_paused = 0;
        wait (rst_n === 1'b1);
        forever begin
            @(negedge clk);

            // Detect the frontend's actual PAUSE -> COLLECT transition rather
            // than relying on the one-cycle packet_ready pulse. This keeps the
            // behavioral ADC aligned even if handshake timing later changes.
            begin : restart_check
                if (frontend_was_paused &&
                    (dut.u_frontend.state === dut.u_frontend.COLLECT))
                    for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;

                frontend_was_paused =
                    (dut.u_frontend.state === dut.u_frontend.PAUSE);
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

    // Scan chain: load weights before releasing reset 
    task load_weights;
        integer i, b;
        reg [TOTAL_SCAN_BITS-1:0] scan_data;
        begin
            // Explicitly clear the scan-length tracker before shifting a new
            // model image.
            scan_en = 0;
            scan_in = 0;
            #(SCAN_PERIOD/2) scan_clk = 1;
            #(SCAN_PERIOD/2) scan_clk = 0;

            for (i = 0; i < TOTAL_SCAN_BITS/8; i++)
                scan_data[i*8 +: 8] = weights[i]; //packing scan_data[i:i+8] into weight[i]
            scan_en = 1; //scan_en = 1 → each rising scan_clk edge shifts in a bit
            for (b = 0; b < TOTAL_SCAN_BITS; b++) begin //scan weight, already stored in scan_data, bit by bit from LSB onwards 
                scan_in = scan_data[b];
                #(SCAN_PERIOD/2) scan_clk = 1;
                #(SCAN_PERIOD/2) scan_clk = 0;
            end
            scan_en = 0; //turn off scan enables
            scan_in = 0;
        end
    endtask

    // SPI master, acts like a recieving MCU or device that reads the SPI interface with the DUT chip
    // Blocks until the DUT's output_formatter raises packet_valid,
    // then clocks the 80-bit packet out over SPI Mode 0.
    // drives spi_cs_n, spi_sclk
    // samples spi_miso
    task spi_receive;
        integer b;
        begin
            wait (dut.packet_valid);
            // Deliberately leave the formatted result pending so the monitor
            // proves packet stability and frontend backpressure behavior.
            repeat (5) @(posedge clk);
            @(posedge clk);

            spi_cs_n = 0; // pull down chip select line
            #(SPI_PERIOD);
            rx_packet = 0; //clear RX packet
            for (b = 0; b < PKT_BITS; b++) begin
                spi_sclk = 0;
                #(SPI_PERIOD/2);
                spi_sclk = 1;
                rx_packet = {rx_packet[PKT_BITS-2:0], spi_miso}; //shifts into LSB 
                #(SPI_PERIOD/2);
            end
            spi_sclk = 0;
            #(SPI_PERIOD);
            spi_cs_n = 1; // disavke chip select
            #(SPI_PERIOD);
            // packet_ready has now fired; sample_collection.resume restarted
            // collection. The background ADC driver will detect the restart
            // at the next negedge and reset drv_cnt to 0.
        end
    endtask

    // Result checker 
    // Check one packet received for ADC vector vec.
    task check_result;
        input integer vec;

        integer   i;
        reg [7:0] sync_byte;
        reg [7:0] class_byte;
        reg [1:0] rx_class;
        reg [1:0] expected_class;
        reg       vector_pass;
        reg [PKT_BITS-1:0] expected_packet;

        begin
            vector_pass = 1'b1;

            // Decode packet fields.
            sync_byte     = rx_packet[79:72];
            class_byte    = rx_packet[71:64];
            rx_class      = class_byte[1:0];
            expected_class = exp_class[vec][1:0];
            expected_packet = {
                8'hAA,
                6'b0, expected_class,
                exp_scores[0][31:16],
                exp_scores[1][31:16],
                exp_scores[2][31:16],
                exp_scores[3][31:16]
            };

            // Byte 0 must contain the framing value.
            if (sync_byte !== 8'hAA) begin
                $display(
                    "FAIL vec%02d: sync got=0x%02X expected=0xAA",
                    vec,
                    sync_byte
                );
                vector_pass = 1'b0;
            end

            // The upper six class-byte bits are reserved and must be zero.
            if (class_byte[7:2] !== 6'b0) begin
                $display(
                    "FAIL vec%02d: reserved class bits are 0x%02X",
                    vec,
                    class_byte[7:2]
                );
                vector_pass = 1'b0;
            end

            // The low two bits contain the predicted class.
            if (rx_class !== expected_class) begin
                $display(
                    "FAIL vec%02d: class got=%0d expected=%0d",
                    vec,
                    rx_class,
                    expected_class
                );
                vector_pass = 1'b0;
            end

            // Check all feature-extractor outputs against the Python golden model.
            for (i = 0; i < N_CH; i = i + 1) begin
                if (dut.sbp_features[i] !== exp_sbp[i]) begin
                    $display(
                        "FAIL vec%02d: SBP[%0d] got=%0d expected=%0d",
                        vec, i, dut.sbp_features[i], exp_sbp[i]
                    );
                    vector_pass = 1'b0;
                end
            end

            // The golden files store signed 64-bit values, while this RTL stores
            // 32-bit scores. First prove that each reference value fits in 32 bits,
            // then compare the complete RTL score bit-for-bit.
            for (i = 0; i < 4; i = i + 1) begin
                if (exp_scores[i][63:32] !== {32{exp_scores[i][31]}}) begin
                    $display(
                        "FAIL vec%02d: golden score[%0d]=%0d exceeds signed 32-bit range",
                        vec, i, $signed(exp_scores[i])
                    );
                    vector_pass = 1'b0;
                end

                if (dut.class_scores[i] !== exp_scores[i][31:0]) begin
                    $display(
                        "FAIL vec%02d: score[%0d] got=%0d expected=%0d",
                        vec, i, $signed(dut.class_scores[i]), $signed(exp_scores[i])
                    );
                    vector_pass = 1'b0;
                end
            end

            // The formatter transmits bits [31:16] of each full score.
            if (rx_packet[63:48] !== exp_scores[0][31:16]) begin
                $display("FAIL vec%02d: SPI score[0] got=0x%04X expected=0x%04X",
                         vec, rx_packet[63:48], exp_scores[0][31:16]);
                vector_pass = 1'b0;
            end
            if (rx_packet[47:32] !== exp_scores[1][31:16]) begin
                $display("FAIL vec%02d: SPI score[1] got=0x%04X expected=0x%04X",
                         vec, rx_packet[47:32], exp_scores[1][31:16]);
                vector_pass = 1'b0;
            end
            if (rx_packet[31:16] !== exp_scores[2][31:16]) begin
                $display("FAIL vec%02d: SPI score[2] got=0x%04X expected=0x%04X",
                         vec, rx_packet[31:16], exp_scores[2][31:16]);
                vector_pass = 1'b0;
            end
            if (rx_packet[15:0] !== exp_scores[3][31:16]) begin
                $display("FAIL vec%02d: SPI score[3] got=0x%04X expected=0x%04X",
                         vec, rx_packet[15:0], exp_scores[3][31:16]);
                vector_pass = 1'b0;
            end

            if (vector_pass) begin
                $display(
                    "PASS vec%02d: sync=0x%02X class=%0d",
                    vec,
                    sync_byte,
                    rx_class
                );
                pass_count = pass_count + 1;
            end else begin
                $display(
                    "  SPI packet actual  : 0x%020X",
                    rx_packet
                );
                $display(
                    "  SPI packet expected: 0x%020X",
                    expected_packet
                );
                $display(
                    "  SBP actual  : %0d %0d %0d %0d %0d %0d %0d %0d",
                    dut.sbp_features[0],
                    dut.sbp_features[1],
                    dut.sbp_features[2],
                    dut.sbp_features[3],
                    dut.sbp_features[4],
                    dut.sbp_features[5],
                    dut.sbp_features[6],
                    dut.sbp_features[7]
                );
                $display(
                    "  SBP expected: %0d %0d %0d %0d %0d %0d %0d %0d",
                    exp_sbp[0], exp_sbp[1], exp_sbp[2], exp_sbp[3],
                    exp_sbp[4], exp_sbp[5], exp_sbp[6], exp_sbp[7]
                );
                $display(
                    "  Scores actual  : PG-LF=%0d PG-HF=%0d SG-LF=%0d SG-HF=%0d",
                    $signed(dut.class_scores[0]),
                    $signed(dut.class_scores[1]),
                    $signed(dut.class_scores[2]),
                    $signed(dut.class_scores[3])
                );
                $display(
                    "  Scores expected: PG-LF=%0d PG-HF=%0d SG-LF=%0d SG-HF=%0d",
                    $signed(exp_scores[0]),
                    $signed(exp_scores[1]),
                    $signed(exp_scores[2]),
                    $signed(exp_scores[3])
                );

                fail_count = fail_count + 1;
            end
        end
    endtask

    // Main sequence
    integer vec;

    initial begin
        rst_n      = 0;
        adc_sample = 0;
        spi_sclk   = 0;
        spi_cs_n   = 1;
        scan_en    = 0;
        scan_clk   = 0;
        scan_in    = 0;
        pipeline_error_count = 0;
        cycle_count = 0;
        frontend_event_count = 0;
        mlp_event_count = 0;
        decision_event_count = 0;
        packet_event_count = 0;
        ready_event_count = 0;
        pipeline_checks_active = 1'b0;

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

        // Prime channel 0 before releasing reset so its first sample is stable
        // before the collector's first active rising edge. The background ADC
        // driver takes over at the following falling edge.
        adc_sample = adc_mem[0];
        drv_cnt[0] = 1;
        @(negedge clk);

        // Release reset between capture edges. adc_channel starts at channel 0.
        rst_n = 1;
        wait (dut.u_mlp.weights_loaded === 1'b1);
        pipeline_checks_active = 1'b1;

        for (vec = 0; vec < N_VECTORS; vec++) begin
            // Wait for the RTL to collect all N_CH*N_SAMPLES samples.
            // The background driver drove them continuously, one per negedge.
            @(posedge dut.sbp_done);

            // Load the Python golden-model outputs for the current vector.
            $readmemh($sformatf("vectors/vec%02d_sbp.hex", vec), exp_sbp);
            $readmemh($sformatf("vectors/vec%02d_scores.hex", vec), exp_scores);

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

        if (frontend_event_count != N_VECTORS ||
            mlp_event_count      != N_VECTORS ||
            decision_event_count != N_VECTORS ||
            packet_event_count   != N_VECTORS ||
            ready_event_count    != N_VECTORS) begin
            $display("PIPELINE FAIL: event totals frontend=%0d mlp=%0d decision=%0d packet=%0d ready=%0d",
                     frontend_event_count, mlp_event_count,
                     decision_event_count, packet_event_count,
                     ready_event_count);
            pipeline_error_count = pipeline_error_count + 1;
        end

        $display("PIPELINE CHECKS: %0d errors", pipeline_error_count);

        if (fail_count == 0 && pipeline_error_count == 0) begin
            $display("ALL PASS");
            $finish;
        end else begin
            $fatal(1, "%0d vector failures, %0d pipeline failures",
                   fail_count, pipeline_error_count);
        end

        
    end

    // Independent watchdog: any pipeline deadlock fails instead of hanging.
    initial begin
        #(5000000);
        $fatal(1, "Top-level pipeline regression timeout");
    end

endmodule
