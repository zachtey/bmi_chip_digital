// spi_slave.sv
// Shifts out a PKT_BYTES-byte packet over SPI Mode 0 (CPOL=0, CPHA=0).
//
// SCLK and CS_N are sampled through a 2-FF synchronizer before use.
// Packet is preloaded into the shift register as soon as CS is active
// so that bit[PKT_BITS-1] is already on MISO before the first rising SCLK
// edge — required for SPI Mode 0.
// Falling SCLK edges update MISO; rising edges shift and count.
// Because SCLK and CS_N are oversampled rather than used as clocks, require
// f_clk >= 8*f_spi. The directed SPI regression sweeps frequency and phase at
// and below that maximum supported SPI rate.

module spi_slave #(
    parameter PKT_BYTES = 10,
    parameter PKT_BITS  = PKT_BYTES * 8
)(
    input  wire                clk,
    input  wire                rst_n,

    input  wire                spi_sclk,
    input  wire                spi_cs_n,
    output wire                spi_miso,

    input  wire [PKT_BITS-1:0] packet_data,
    input  wire                packet_valid,
    output logic               packet_ready
);

    // 2-FF synchronizer for SCLK and CS_N
    logic sclk_sync1, sclk_sync2;
    logic cs_n_sync1, cs_n_sync2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync1 <= 1'b0; sclk_sync2 <= 1'b0;
            cs_n_sync1 <= 1'b1; cs_n_sync2 <= 1'b1;
        end else begin
            sclk_sync1 <= spi_sclk;  sclk_sync2 <= sclk_sync1;
            cs_n_sync1 <= spi_cs_n;  cs_n_sync2 <= cs_n_sync1;
        end
    end

    wire sclk_rising  = ( sclk_sync1 && !sclk_sync2);
    wire sclk_falling = (!sclk_sync1 &&  sclk_sync2);
    wire cs_active    = !cs_n_sync2;

    // FSM
    typedef enum logic [1:0] { S_IDLE, S_TRANSMIT, S_DONE } state_t;
    state_t                          state,      next_state;

    logic [PKT_BITS-1:0]             shift_reg;
    logic [$clog2(PKT_BITS)-1:0]     bit_cnt;
    logic                             miso_reg;

    // Next-cycle values
    logic [PKT_BITS-1:0]             next_shift_reg;
    logic [$clog2(PKT_BITS)-1:0]     next_bit_cnt;
    logic                             next_miso_reg;
    logic                             next_packet_ready;

    // Combinational: next-state and datapath
    always_comb begin
        // Defaults — hold state
        next_state        = state;
        next_shift_reg    = shift_reg;
        next_bit_cnt      = bit_cnt;
        next_miso_reg     = miso_reg;
        next_packet_ready = 1'b0;

        // CS deassert resets to idle from any state
        if (!cs_active) begin
            next_state     = S_IDLE;
            next_bit_cnt   = '0;
            next_miso_reg  = 1'b0;
            next_shift_reg = '0;
        end else begin
            unique case (state)
                S_IDLE: begin
                    // Preload as soon as CS is active so MISO is ready before
                    // the master's first rising SCLK edge.
                    if (packet_valid) begin
                        next_shift_reg = packet_data;
                        next_miso_reg  = packet_data[PKT_BITS-1];
                        next_bit_cnt   = '0;
                        next_state     = S_TRANSMIT;
                    end
                end

                S_TRANSMIT: begin
                    // Update MISO on falling edge for next master sample
                    if (sclk_falling)
                        next_miso_reg = shift_reg[PKT_BITS-1];

                    // Shift on rising edge; detect last bit
                    if (sclk_rising) begin
                        next_shift_reg = shift_reg << 1;
                        next_bit_cnt   = bit_cnt + 1;
                        if (bit_cnt == PKT_BITS - 1) begin
                            next_packet_ready = 1'b1;
                            next_state        = S_DONE;
                        end
                    end
                end

                // Wait for CS to deassert (handled by the !cs_active priority above)
                S_DONE: ;

                default: next_state = S_IDLE;
            endcase
        end
    end

    // Sequential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            shift_reg    <= '0;
            bit_cnt      <= '0;
            miso_reg     <= 1'b0;
            packet_ready <= 1'b0;
        end else begin
            state        <= next_state;
            shift_reg    <= next_shift_reg;
            bit_cnt      <= next_bit_cnt;
            miso_reg     <= next_miso_reg;
            packet_ready <= next_packet_ready;
        end
    end

    assign spi_miso = miso_reg;

endmodule
