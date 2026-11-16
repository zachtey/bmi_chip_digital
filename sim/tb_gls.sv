// tb_gls.sv - gate-level (or RTL) simulation through the chip's ports only.
//
// The host loads and commits the weights over APB, starts the pipeline, and
// for each of the 40 vectors waits for irq, reads RESULT and SCORE0..3,
// compares them with the Python golden files and writes RESULT_ACK. No
// signal inside the DUT is referenced, so the same bench runs on a netlist.
//
// ADC model: it follows adc_channel alone. A channel's sample index advances
// when the MUX moves off that channel; when the last channel has supplied
// N_SAMPLES samples the window is complete and the next vector is loaded.

`timescale 1ns/1ps

module tb_gls;

    parameter N_CH              = 8;
    parameter N_SAMPLES         = 250;
    parameter ADC_WIDTH         = 8;
    parameter N_VECTORS         = 40;
    parameter TOTAL_WEIGHT_BITS = 864;
    parameter N_WEIGHT_WORDS    = TOTAL_WEIGHT_BITS / 32;

    parameter CLK_PERIOD     = 10;       // bmi_clk period, ns
    parameter PCLK_PERIOD_PS = 7000;     // pclk period, ps
    localparam real PCLK_PERIOD = PCLK_PERIOD_PS / 1000.0;

    localparam [11:0] ADDR_COMMAND = 12'h000, ADDR_STATUS = 12'h004,
                      ADDR_IRQ_EN  = 12'h008, ADDR_ACK    = 12'h00C,
                      ADDR_RESULT  = 12'h010, ADDR_SCORE0 = 12'h014,
                      ADDR_COMMIT  = 12'h024, ADDR_ERROR  = 12'h028,
                      ADDR_WEIGHT0 = 12'h100;
    localparam [31:0] ST_RESULT_VALID = 32'h01, ST_CMD_BUSY = 32'h02,
                      ST_WT_BUSY = 32'h04, ST_WT_VALID = 32'h08, ST_RUNNING = 32'h10;

    reg                  bmi_clk, pclk, presetn;
    reg  [ADC_WIDTH-1:0] adc_sample;
    wire [2:0]           adc_channel;
    reg                  psel, penable, pwrite;
    reg  [11:0]          paddr;
    reg  [31:0]          pwdata;
    wire [31:0]          prdata;
    wire                 pready, pslverr, irq;

    // Default parameters only, so a netlist can be dropped in. The design
    // default is 10 MHz / 2.5 MS/s, i.e. one capture every 4 clocks.
    bmi_chip_top dut (
        .bmi_clk(bmi_clk), .pclk(pclk), .presetn(presetn),
        .adc_sample(adc_sample), .adc_channel(adc_channel),
        .psel(psel), .penable(penable), .pwrite(pwrite),
        .paddr(paddr), .pwdata(pwdata),
        .prdata(prdata), .pready(pready), .pslverr(pslverr),
        .irq(irq)
    );

    initial bmi_clk = 0;
    always #(CLK_PERIOD/2.0) bmi_clk = ~bmi_clk;
    initial pclk = 0;
    always #(PCLK_PERIOD/2.0) pclk = ~pclk;

    reg [7:0] adc_mem   [0:N_CH*N_SAMPLES-1];
    reg [7:0] exp_class [0:N_VECTORS-1];
    reg [7:0] weights   [0:TOTAL_WEIGHT_BITS/8-1];
    reg signed [63:0] exp_scores [0:3];

    integer pass_count = 0, fail_count = 0, apb_error_count = 0;

    // ---- ADC model (ports only) ----
    integer drv_cnt [0:N_CH-1];
    integer prev_channel, adc_vec;

    initial begin : adc_model
        integer ch;
        for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
        prev_channel = 0;
        adc_vec = 0;
        wait (presetn === 1'b1);
        forever begin
            @(negedge bmi_clk);
            ch = adc_channel;
            if (ch != prev_channel) begin
                drv_cnt[prev_channel] = drv_cnt[prev_channel] + 1;
                // Last channel has given its last sample: window complete.
                if (prev_channel == N_CH - 1 && drv_cnt[N_CH-1] == N_SAMPLES) begin
                    for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
                    adc_vec = adc_vec + 1;
                    if (adc_vec < N_VECTORS)
                        $readmemh($sformatf("vectors/vec%02d_adc.hex", adc_vec), adc_mem);
                end
            end
            prev_channel = adc_channel;
            ch = adc_channel;
            adc_sample = adc_mem[ch * N_SAMPLES +
                                 (drv_cnt[ch] < N_SAMPLES ? drv_cnt[ch] : N_SAMPLES - 1)];
        end
    end

    // ---- APB master ----
    task apb_bus_idle;
        begin psel = 0; penable = 0; pwrite = 0; paddr = 0; pwdata = 0; end
    endtask

    task apb_transfer;
        input         write;
        input  [11:0] addr;
        input  [31:0] wdata;
        input         expect_error;
        output [31:0] rdata;
        begin
            @(posedge pclk); #1;
            psel = 1; penable = 0; pwrite = write; paddr = addr; pwdata = wdata;
            @(posedge pclk); #1;
            penable = 1;
            @(negedge pclk);
            rdata = prdata;
            if (pslverr !== expect_error || pready !== 1'b1) begin
                $display("APB FAIL: addr=0x%03X write=%b pslverr=%b (expected %b) pready=%b",
                         addr, write, pslverr, expect_error, pready);
                apb_error_count = apb_error_count + 1;
            end
            @(posedge pclk); #1;
            apb_bus_idle();
        end
    endtask

    reg [31:0] scratch;
    task apb_read;  input [11:0] addr; output [31:0] rdata;
        begin apb_transfer(1'b0, addr, 32'd0, 1'b0, rdata); end
    endtask
    task apb_write; input [11:0] addr; input [31:0] wdata;
        begin apb_transfer(1'b1, addr, wdata, 1'b0, scratch); end
    endtask

    task wait_not_busy;
        output [31:0] status;
        begin
            apb_read(ADDR_STATUS, status);
            while (status & (ST_CMD_BUSY | ST_WT_BUSY)) apb_read(ADDR_STATUS, status);
        end
    endtask

    task expect_status;
        input [31:0] expected;
        reg   [31:0] status;
        begin
            apb_read(ADDR_STATUS, status);
            if (status !== expected) begin
                $display("APB FAIL: STATUS=0x%08X expected 0x%08X", status, expected);
                apb_error_count = apb_error_count + 1;
            end
        end
    endtask

    integer vec, n, i;
    reg [31:0] word, readback, status, rx_result;
    reg [31:0] rx_scores [0:3];
    reg        vector_pass;

    initial begin
        presetn = 0;
        adc_sample = 0;
        apb_bus_idle();
        $readmemh("weights.hex", weights);
        $readmemh("vectors/all_expected.hex", exp_class);
        $readmemh("vectors/vec00_adc.hex", adc_mem);

        repeat (4) @(posedge pclk);
        repeat (4) @(posedge bmi_clk);
        @(negedge pclk);
        presetn = 1;
        repeat (3) @(posedge pclk);

        expect_status(32'd0);
        apb_transfer(1'b1, ADDR_COMMAND, 32'd1, 1'b1, scratch);   // START before commit: refused

        // Load, read back and commit the weights.
        for (n = 0; n < N_WEIGHT_WORDS; n = n + 1)
            apb_write(ADDR_WEIGHT0 + 4*n,
                      {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]});
        for (n = 0; n < N_WEIGHT_WORDS; n = n + 1) begin
            word = {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]};
            apb_read(ADDR_WEIGHT0 + 4*n, readback);
            if (readback !== word) begin
                $display("WEIGHT FAIL: word %0d read 0x%08X expected 0x%08X", n, readback, word);
                apb_error_count = apb_error_count + 1;
            end
        end
        apb_write(ADDR_COMMIT, 32'd1);
        wait_not_busy(status);
        expect_status(ST_WT_VALID);

        apb_write(ADDR_IRQ_EN, 32'd1);
        apb_write(ADDR_COMMAND, 32'd1);          // START
        wait_not_busy(status);
        if (!(status & ST_RUNNING)) begin
            $display("APB FAIL: RUNNING not set after START");
            apb_error_count = apb_error_count + 1;
        end

        for (vec = 0; vec < N_VECTORS; vec = vec + 1) begin
            wait (irq === 1'b1);
            $readmemh($sformatf("vectors/vec%02d_scores.hex", vec), exp_scores);
            apb_read(ADDR_RESULT, rx_result);
            for (i = 0; i < 4; i = i + 1) apb_read(ADDR_SCORE0 + 4*i, rx_scores[i]);
            apb_write(ADDR_ACK, 32'd1);

            vector_pass = 1'b1;
            if (rx_result !== {30'd0, exp_class[vec][1:0]}) begin
                $display("FAIL vec%02d: RESULT=0x%08X expected class %0d",
                         vec, rx_result, exp_class[vec][1:0]);
                vector_pass = 1'b0;
            end
            for (i = 0; i < 4; i = i + 1)
                if (rx_scores[i] !== exp_scores[i][31:0]) begin
                    $display("FAIL vec%02d: SCORE%0d got=0x%08X expected=0x%08X",
                             vec, i, rx_scores[i], exp_scores[i][31:0]);
                    vector_pass = 1'b0;
                end
            if (vector_pass) begin
                $display("PASS vec%02d: class=%0d scores=%0d %0d %0d %0d", vec, rx_result[1:0],
                         $signed(rx_scores[0]), $signed(rx_scores[1]),
                         $signed(rx_scores[2]), $signed(rx_scores[3]));
                pass_count = pass_count + 1;
            end else
                fail_count = fail_count + 1;
        end

        apb_write(ADDR_COMMAND, 32'd2);          // STOP
        wait_not_busy(status);
        if (status & ST_RUNNING) begin
            $display("APB FAIL: RUNNING still set after STOP");
            apb_error_count = apb_error_count + 1;
        end
        apb_read(ADDR_ERROR, status);
        if (status !== 32'd0) begin
            $display("APB FAIL: ERROR_STATUS=0x%08X", status);
            apb_error_count = apb_error_count + 1;
        end

        $display("----------------------------------------------");
        $display("GLS  %0d / %0d  PASS, %0d APB errors", pass_count, N_VECTORS, apb_error_count);
        $display("----------------------------------------------");
        if (fail_count == 0 && pass_count == N_VECTORS && apb_error_count == 0) begin
            $display("ALL PASS");
            $finish;
        end else
            $fatal(1, "%0d vector failures, %0d APB errors", fail_count, apb_error_count);
    end

    initial begin
        #(64'd50 * N_CH * N_SAMPLES * 4 * CLK_PERIOD + 4000.0 * PCLK_PERIOD);
        $fatal(1, "GLS timeout");
    end

endmodule
