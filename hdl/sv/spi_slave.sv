// ============================================================
// spi_slave.v
//
// Shifts out the 10-byte packet over SPI when the master
// drives SCLK while CS_N is low. SPI Mode 0 (CPOL=0, CPHA=0).
//
// ============================================================
// BUG FIXES (Chris, May 2026):
//
//   ROOT CAUSE: Loading the packet on the first detected SCLK
//   falling edge is not compatible with SPI Mode 0. In Mode 0,
//   SCLK idles low and the master samples on the first rising
//   edge, so the slave must already be driving bit 79 before
//   that first rising edge arrives.
//
//   FIX: Once CS is active and packet_valid is high, preload the
//   packet immediately in the system clock domain and drive
//   miso_reg with packet_data[PKT_BITS-1]. Falling SCLK edges then
//   update MISO for subsequent bits, and rising edges shift.
//
//   Other improvements:
//     - Clear all state (bit_cnt, done, miso_reg) on CS deassert
//       so each transaction starts fresh.
//     - Restructured spi_miso as a simple registered output
//       (clean, no multi-driver issues).
// ============================================================

module spi_slave #(
    parameter PKT_BYTES = 10,
    parameter PKT_BITS  = PKT_BYTES * 8   // 80
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // External SPI pins
    input  wire                  spi_sclk,
    input  wire                  spi_cs_n,
    output wire                  spi_miso,

    // From output_formatter
    input  wire [PKT_BITS-1:0]   packet_data,
    input  wire                  packet_valid,

    // To output_formatter
    output reg                   packet_ready
);

    // ══════════════════════════════════════════════════════
    // 2-FF SYNCHRONIZER
    // ══════════════════════════════════════════════════════
    reg sclk_sync1, sclk_sync2;
    reg cs_n_sync1, cs_n_sync2;

    always @(posedge clk or negedge rst_n) begin
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

    // ══════════════════════════════════════════════════════
    // SHIFT REGISTER + BIT COUNTER + STATE
    // ══════════════════════════════════════════════════════
    reg [PKT_BITS-1:0]         shift_reg;
    reg [$clog2(PKT_BITS)-1:0] bit_cnt;
    reg                        transmitting;
    reg                        done;
    reg                        miso_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg    <= '0;
            bit_cnt      <= '0;
            transmitting <= 1'b0;
            done         <= 1'b0;
            miso_reg     <= 1'b0;
            packet_ready <= 1'b0;

        end else begin
            packet_ready <= 1'b0;

            // ── Reset state when CS deasserts ─────────────
            if (!cs_active) begin
                transmitting <= 1'b0;
                bit_cnt      <= '0;
                done         <= 1'b0;
                miso_reg     <= 1'b0;

            end else begin
                // Preload as soon as CS is active so SPI Mode 0 masters
                // can sample bit 79 on the first SCLK rising edge.
                if (!transmitting && !done && packet_valid) begin
                    shift_reg    <= packet_data;
                    bit_cnt      <= '0;
                    transmitting <= 1'b1;
                    miso_reg     <= packet_data[PKT_BITS-1];
                end

                if (transmitting) begin
                    // Drive MISO on falling edge (for bits 78..0)
                    // Note: we use shift_reg AFTER it has been
                    // shifted by the previous rising edge, so
                    // shift_reg[79] always holds the next bit.
                    if (sclk_falling) begin
                        miso_reg <= shift_reg[PKT_BITS-1];
                    end

                    // Shift on rising edge
                    if (sclk_rising) begin
                        shift_reg <= shift_reg << 1;
                        bit_cnt   <= bit_cnt + 1;

                        if (bit_cnt == PKT_BITS - 1) begin
                            transmitting <= 1'b0;
                            done         <= 1'b1;
                            packet_ready <= 1'b1;
                        end
                    end
                end
            end
        end
    end

    // Simple registered output — no combinational override needed
    assign spi_miso = miso_reg;

endmodule
