`timescale 1ns/1ps
module tb_debug;
    parameter CLK_PERIOD = 10;
    parameter SPI_CLK_DIV = 5;
    parameter PKT_BITS = 80;

    reg clk, rst_n;
    reg [7:0] adc_sample;
    reg adc_valid;
    reg [2:0] adc_channel;
    reg spi_sclk, spi_cs_n;
    wire spi_miso;
    reg scan_en, scan_clk, scan_in;

    bmi_chip_top #(.HIDDEN_BIAS_SCALE(85),.OUTPUT_BIAS_SCALE(109)) dut (
        .clk(clk),.rst_n(rst_n),
        .adc_sample(adc_sample),.adc_valid(adc_valid),.adc_channel(adc_channel),
        .spi_sclk(spi_sclk),.spi_cs_n(spi_cs_n),.spi_miso(spi_miso),
        .scan_en(scan_en),.scan_clk(scan_clk),.scan_in(scan_in));

    initial clk=0;
    always #(CLK_PERIOD/2) clk=~clk;

    reg [7:0] adc_mem[0:8*250-1];
    reg [7:0] weights[0:107];
    reg [PKT_BITS-1:0] rx_packet;

    task load_weights;
        integer i,b;
        reg [863:0] scan_data;
        begin
            for(i=0;i<108;i++) scan_data[i*8+:8]=weights[i];
            scan_en=1;
            for(b=0;b<864;b++) begin
                scan_in=scan_data[b];
                #20 scan_clk=1; #20 scan_clk=0;
            end
            scan_en=0; scan_in=0;
        end
    endtask

    integer ch, s, b, t;
    initial begin
        $dumpfile("top/debug_waves.vcd");
        $dumpvars(0, tb_debug);

        rst_n=0; adc_valid=0; adc_sample=0; adc_channel=0;
        spi_sclk=0; spi_cs_n=1; scan_en=0; scan_clk=0; scan_in=0;
        repeat(4) @(posedge clk); rst_n=1;
        repeat(2) @(posedge clk);

        $readmemh("weights.hex", weights);
        load_weights();
        $display("t=%0t weights done", $time);
        repeat(4) @(posedge clk);

        $readmemh("vectors/vec00_adc.hex", adc_mem);

        // Drive ADC window
        for(ch=0; ch<8; ch++) begin
            for(s=0; s<250; s++) begin
                @(posedge clk); #1;
                adc_sample=adc_mem[ch*250+s]; adc_channel=ch[2:0]; adc_valid=1;
                @(posedge clk); #1;
                adc_valid=0;
            end
        end
        $display("t=%0t ADC window done", $time);

        // Monitor internal signals
        $display("t=%0t window_ready=%b sbp_done=%b mlp_done=%b decision_valid=%b packet_valid=%b",
            $time,
            dut.window_ready, dut.sbp_done, dut.mlp_done,
            dut.u_argmax.decision_valid, dut.u_fmt.packet_valid);

        // Wait up to 50000 clocks for packet_valid
        t=0;
        while(!dut.u_fmt.packet_valid && t<50000) begin
            @(posedge clk); t=t+1;
            if(t%1000==0)
                $display("t=%0t waiting... window_ready=%b sbp_done=%b mlp_done=%b decision_valid=%b packet_valid=%b",
                    $time,
                    dut.window_ready, dut.sbp_done, dut.mlp_done,
                    dut.u_argmax.decision_valid, dut.u_fmt.packet_valid);
        end

        if(t==50000) begin
            $display("TIMEOUT - packet_valid never went high");
            $display("Final state: window_ready=%b sbp_done=%b mlp_done=%b decision_valid=%b packet_valid=%b",
                dut.window_ready, dut.sbp_done, dut.mlp_done,
                dut.u_argmax.decision_valid, dut.u_fmt.packet_valid);
        end else begin
            $display("t=%0t packet_valid asserted! packet_data=0x%020X", $time, dut.u_fmt.packet_data);

            // SPI receive
            @(posedge clk); #1; spi_cs_n=0;
            repeat(SPI_CLK_DIV*2) @(posedge clk);
            rx_packet=0;
            for(b=0;b<PKT_BITS;b++) begin
                @(posedge clk); #1; spi_sclk=0;
                repeat(SPI_CLK_DIV) @(posedge clk);
                rx_packet={spi_miso, rx_packet[PKT_BITS-1:1]};
                @(posedge clk); #1; spi_sclk=1;
                repeat(SPI_CLK_DIV) @(posedge clk);
            end
            @(posedge clk); #1; spi_sclk=0;
            repeat(SPI_CLK_DIV*2) @(posedge clk);
            @(posedge clk); #1; spi_cs_n=1;

            $display("rx_packet=0x%020X", rx_packet);
            $display("sync byte=0x%02X (expect 0xAA)", rx_packet[79:72]);
            $display("class byte=0x%02X, class bits=%0d", rx_packet[71:64], rx_packet[65:64]);
        end

        $finish;
    end
endmodule
