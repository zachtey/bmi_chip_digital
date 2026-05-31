// sample_collection.sv
// Collects 8 channels x 250 samples from ADC into a window.
// When all 2000 samples are received, pulses window_ready for
// one clock cycle and presents the full window on sample_window.
module sample_collection #(
    parameter N_CH      = 8,    // number of electrode channels
    parameter N_SAMPLES = 250,  // samples per channel per 50ms bin
    parameter ADC_WIDTH = 8     // ADC resolution in bits
)(
    input  wire                                        clk,
    input  wire                                        rst_n,       // active-low sync reset

    // ADC interface - adc_channel drives the analog MUX select
    input  wire [ADC_WIDTH-1:0]                        adc_sample,
    output reg  [$clog2(N_CH)-1:0]                    adc_channel, // MUX select output, cycles 0..N_CH-1

    // Collection resume: pulse high to restart collecting after window_ready.
    // Wire to packet_ready from spi_slave so collection restarts only after
    // the downstream pipeline has fully consumed the previous window.
    input  wire                                        resume,

    // Window interface
    output reg                                         window_ready,  // 1-cycle pulse
    output reg  [ADC_WIDTH-1:0]                        sample_window [0:N_CH-1][0:N_SAMPLES-1]
);

    reg [$clog2(N_SAMPLES)-1:0] sample_cnt [0:N_CH-1];
    reg [$clog2(N_CH+1)-1:0]   done_count;
    reg collecting;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_ready <= 1'b0;
            done_count   <= '0;
            collecting   <= 1'b1;
            adc_channel  <= '0;
            for (i = 0; i < N_CH; i++) begin
                sample_cnt[i] <= '0;
                for (j = 0; j < N_SAMPLES; j++)
                    sample_window[i][j] <= '0;
            end

        end else begin
            window_ready <= 1'b0;

            // Advance MUX select every clock cycle, wrapping 0..N_CH-1
            adc_channel <= (adc_channel == N_CH - 1) ? '0 : adc_channel + 1;

            if (collecting) begin
                if (sample_cnt[adc_channel] < N_SAMPLES) begin
                    sample_window[adc_channel][sample_cnt[adc_channel]] <= adc_sample;
                    sample_cnt[adc_channel] <= sample_cnt[adc_channel] + 1;

                    if (sample_cnt[adc_channel] == N_SAMPLES - 1) begin
                        done_count <= done_count + 1;

                        if (done_count == N_CH - 1) begin
                            window_ready <= 1'b1;
                            collecting   <= 1'b0;
                        end
                    end
                end
            end

            // Restart collecting when the SPI slave signals it has finished
            // transmitting the previous window's packet (resume = packet_ready).
            if (resume) begin
                done_count <= '0;
                for (i = 0; i < N_CH; i++)
                    sample_cnt[i] <= '0;
                collecting <= 1'b1;
            end
        end
    end

endmodule
