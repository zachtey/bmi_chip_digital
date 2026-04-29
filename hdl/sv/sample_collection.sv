// sample_collection.v
// Collects 8 channels × 250 samples from ADC into a window.
// When all 2000 samples are received, pulses window_ready for
// one clock cycle and presents the full window on sample_window.
module sample_collection #(
    parameter N_CH      = 8,    // number of electrode channels
    parameter N_SAMPLES = 250,  // samples per channel per 50ms bin
    parameter ADC_WIDTH = 8     // ADC resolution in bits
)(
    input  wire                                        clk,
    input  wire                                        rst_n,       // active-low sync reset

    // ADC interface (one sample per valid pulse)
    input  wire [ADC_WIDTH-1:0]                        adc_sample,
    input  wire                                        adc_valid,   // 1-cycle strobe
    input  wire [$clog2(N_CH)-1:0]                    adc_channel, // which channel this sample belongs to

    // Window interface
    output reg                                         window_ready,  // 1-cycle pulse
    output reg  [ADC_WIDTH-1:0]                        sample_window [0:N_CH-1][0:N_SAMPLES-1]
);

    // Per-channel sample counter 
    // Tracks how many samples have arrived for each channel.
    // All 8 counters run independently — the ADC can deliver
    // channels in any order (though ours is round-robin).
    reg [$clog2(N_SAMPLES)-1:0] sample_cnt [0:N_CH-1];

    // Number of channels that have reached N_SAMPLES ───────
    // When done_count == N_CH, the window is complete.
    reg [$clog2(N_CH+1)-1:0]   done_count;

    // Collecting flag 
    // Goes low for one cycle after window_ready, then high again.
    // Prevents overwriting a window before the downstream block
    // has consumed it. If you want back-to-back windows without
    // any gap, tie start_next directly to window_ready.
    reg collecting;

    integer i, j;

    always @(posedge clk) begin
        if (!rst_n) begin
            // ── Reset everything ─────────────────────────────
            window_ready <= 1'b0;
            done_count   <= '0;
            collecting   <= 1'b1;   // start collecting immediately after reset
            for (i = 0; i < N_CH; i++) begin
                sample_cnt[i] <= '0;
                for (j = 0; j < N_SAMPLES; j++)
                    sample_window[i][j] <= '0;
            end

        end else begin
            // Default: deassert window_ready (it's a 1-cycle pulse)
            window_ready <= 1'b0;

            if (collecting && adc_valid) begin
                // Write sample into memory 
                // Only accept if this channel hasn't filled yet.
                // This protects against the ADC delivering extra
                // samples for a channel before the window resets.
                if (sample_cnt[adc_channel] < N_SAMPLES) begin
                    sample_window[adc_channel][sample_cnt[adc_channel]] <= adc_sample;
                    sample_cnt[adc_channel] <= sample_cnt[adc_channel] + 1;

                    // ── Detect when this channel just finished ─
                    if (sample_cnt[adc_channel] == N_SAMPLES - 1) begin
                        done_count <= done_count + 1;

                        // ── Detect when ALL channels finished ──
                        if (done_count == N_CH - 1) begin
                            window_ready <= 1'b1;   // pulse for exactly one cycle
                            collecting   <= 1'b0;   // pause collecting
                        end
                    end
                end
            end

            // Restart for next window 
            // Start a new window the cycle after window_ready.
            // Downstream (SBP block) reads sample_window combinationally
            // so one cycle of latency is fine.
            if (window_ready) begin
                done_count <= '0;
                for (i = 0; i < N_CH; i++)
                    sample_cnt[i] <= '0;
                collecting <= 1'b1;
            end
        end
    end

endmodule