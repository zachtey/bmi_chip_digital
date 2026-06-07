// sample_collection.sv
// Time-multiplexes a single ADC across N_CH electrode channels.
// Cycles adc_channel 0..N_CH-1 every clock; collects N_SAMPLES per channel
// into sample_window. Pulses window_ready for one cycle when the window is full,
// then stalls until resume (= packet_ready from spi_slave) restarts collection.

module sample_collection #(
    parameter N_CH      = 8,
    parameter N_SAMPLES = 250,
    parameter ADC_WIDTH = 8
)(
    input  wire                                       clk,
    input  wire                                       rst_n,

    input  wire [ADC_WIDTH-1:0]                       adc_sample,
    output logic [$clog2(N_CH)-1:0]                  adc_channel,

    // Pulse high to restart after window_ready. Wire to spi_slave.packet_ready.
    input  wire                                       resume,

    output logic                                      window_ready,
    output logic [ADC_WIDTH-1:0]                      sample_window [0:N_CH-1][0:N_SAMPLES-1]
);

    typedef enum logic { COLLECT, PAUSE } state_t;

    state_t                          state;
    logic [$clog2(N_SAMPLES)-1:0]    sample_cnt   [0:N_CH-1];
    logic [$clog2(N_CH+1)-1:0]      done_count;

    // Next-cycle values (driven by always_comb)
    state_t                          next_state;
    logic [$clog2(N_CH)-1:0]         next_adc_channel;
    logic [$clog2(N_SAMPLES)-1:0]    next_sample_cnt [0:N_CH-1];
    logic [$clog2(N_CH+1)-1:0]       next_done_count;
    logic                             next_window_ready;

    // Window write port (driven by always_comb, used in always_ff)
    logic                             win_we;
    logic [$clog2(N_CH)-1:0]          win_ch;
    logic [$clog2(N_SAMPLES)-1:0]     win_idx;

    // -------------------------------------------------------------------------
    // Combinational: next-state and datapath
    // -------------------------------------------------------------------------
    always_comb begin
        next_state        = state;
        next_adc_channel  = (adc_channel == N_CH - 1) ? '0 : adc_channel + 1;
        next_done_count   = done_count;
        next_window_ready = 1'b0;
        win_we            = 1'b0;
        win_ch            = adc_channel;
        win_idx           = sample_cnt[adc_channel];

        for (int i = 0; i < N_CH; i++)
            next_sample_cnt[i] = sample_cnt[i];

        unique case (state)
            COLLECT: begin
                if (sample_cnt[adc_channel] < N_SAMPLES) begin
                    win_we  = 1'b1;
                    win_ch  = adc_channel;
                    win_idx = sample_cnt[adc_channel];
                    next_sample_cnt[adc_channel] = sample_cnt[adc_channel] + 1;

                    if (sample_cnt[adc_channel] == N_SAMPLES - 1) begin
                        next_done_count = done_count + 1;
                        if (done_count == N_CH - 1) begin
                            next_window_ready = 1'b1;
                            next_state      = PAUSE;
                        end
                    end
                end
            end

            PAUSE: begin
                if (resume) begin
                    next_done_count = '0;
                    for (int i = 0; i < N_CH; i++)
                        next_sample_cnt[i] = '0;
                    next_state = COLLECT;
                end
            end

            default: next_state = COLLECT;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential: state and datapath registers
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= COLLECT;
            adc_channel  <= '0;
            window_ready <= 1'b0;
            done_count   <= '0;
            for (int i = 0; i < N_CH; i++) begin
                sample_cnt[i] <= '0;
                for (int j = 0; j < N_SAMPLES; j++)
                    sample_window[i][j] <= '0;
            end
        end else begin
            state        <= next_state;
            adc_channel  <= next_adc_channel;
            window_ready <= next_window_ready;
            done_count   <= next_done_count;
            for (int i = 0; i < N_CH; i++)
                sample_cnt[i] <= next_sample_cnt[i];
            if (win_we)
                sample_window[win_ch][win_idx] <= adc_sample;
        end
    end

endmodule
