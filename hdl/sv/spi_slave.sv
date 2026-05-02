// ============================================================
// spi_slave.v
//
// Shifts out the 10-byte packet over SPI when the master
// drives SCLK while CS_N is low. Chip is always SPI Mode 0
// (CPOL=0, CPHA=0): data valid on rising SCLK edge,
// captured by master on rising edge.
//
// CLOCK DOMAIN CROSSING STRATEGY:
//   sclk is external — never used as a clock in this design.
//   Instead, sclk is sampled by the internal clk (1 MHz) using
//   a 2-FF synchronizer. A rising edge on sclk (detected by
//   comparing the synchronizer's two stages) triggers a shift.
//   This means your internal clk must be at least 2x faster
//   than sclk. At 1 MHz internal / 100 kHz SPI that's 10x —
//   plenty of margin.
//
// Packet: 80 bits (10 bytes), MSB first.
// After all 80 bits are shifted out, packet_ready pulses once
// to tell output_formatter to deassert packet_valid.
// ============================================================
module spi_slave #(
    parameter PKT_BYTES = 10,
    parameter PKT_BITS  = PKT_BYTES * 8   // 80
)(
    input  wire                  clk,       // internal system clock
    input  wire                  rst_n,

    // External SPI pins (from master / off-chip)
    input  wire                  spi_sclk,  // SPI clock from master
    input  wire                  spi_cs_n,  // chip select, active low
    output reg                   spi_miso,  // master-in slave-out

    // From output_formatter
    input  wire [PKT_BITS-1:0]   packet_data,   // 80-bit packet, MSB first
    input  wire                  packet_valid,  // formatter has data ready

    // To output_formatter
    output reg                   packet_ready   // 1-cycle pulse: transmission done
);

    // ══════════════════════════════════════════════════════
    // 2-FF SYNCHRONIZER — sclk edge detection
    // ══════════════════════════════════════════════════════
    // Never clock your chip's flip-flops with sclk directly.
    // Instead, sample sclk with two back-to-back flip-flops
    // clocked by your safe internal clk. This eliminates
    // metastability. A rising edge on sclk appears as:
    //   sclk_sync1 goes 0→1 one cycle after the real edge
    //   sclk_sync2 follows one cycle later
    // Rising edge detected when: sync1=1 AND sync2=0

    reg sclk_sync1, sclk_sync2;   // synchronizer FFs
    reg cs_n_sync1, cs_n_sync2;   // same treatment for cs_n

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync1 <= 1'b0; sclk_sync2 <= 1'b0;
            cs_n_sync1 <= 1'b1; cs_n_sync2 <= 1'b1;
        end else begin
            sclk_sync1 <= spi_sclk;  sclk_sync2 <= sclk_sync1;
            cs_n_sync1 <= spi_cs_n;  cs_n_sync2 <= cs_n_sync1;
        end
    end

    wire sclk_rising  = ( sclk_sync1 && !sclk_sync2);   // detected rising edge
    wire sclk_falling = (!sclk_sync1 &&  sclk_sync2);   // detected falling edge
    wire cs_active    = !cs_n_sync2;                     // cs_n low = selected

    // ══════════════════════════════════════════════════════
    // SHIFT REGISTER + BIT COUNTER
    // ══════════════════════════════════════════════════════
    reg [PKT_BITS-1:0]         shift_reg;
    reg [$clog2(PKT_BITS)-1:0] bit_cnt;
    reg                        transmitting;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg    <= '0;
            bit_cnt      <= '0;
            transmitting <= 1'b0;
            spi_miso     <= 1'b0;
            packet_ready <= 1'b0;

        end else begin
            packet_ready <= 1'b0;   // default deassert

            // ── Load packet when CS goes active ──────────
            // On the falling edge of CS_N (master starts transaction),
            // load the shift register if the formatter has data.
            // MSB goes out first — load packet_data directly.
            if (sclk_falling && cs_active && !transmitting) begin
                if (packet_valid) begin
                    shift_reg    <= packet_data;
                    bit_cnt      <= '0;
                    transmitting <= 1'b1;
                end
            end

            if (transmitting && cs_active) begin

                // ── Drive MISO on falling edge ────────────
                // SPI Mode 0: master samples MISO on rising edge.
                // We update MISO on falling edge so it's stable
                // well before the next rising edge.
                // Output MSB first: shift_reg[PKT_BITS-1]
                if (sclk_falling) begin
                    spi_miso <= shift_reg[PKT_BITS-1];
                end

                // ── Shift on rising edge ──────────────────
                // Shift the register left and increment counter.
                if (sclk_rising) begin
                    shift_reg <= shift_reg << 1;
                    bit_cnt   <= bit_cnt + 1;

                    // ── Last bit just clocked out ─────────
                    if (bit_cnt == PKT_BITS - 1) begin
                        transmitting <= 1'b0;
                        packet_ready <= 1'b1;   // tell formatter we're done
                    end
                end

            end

            // ── Abort if CS deasserted early ──────────────
            if (!cs_active && transmitting) begin
                transmitting <= 1'b0;
                spi_miso     <= 1'b0;
                // Do NOT pulse packet_ready — packet was not fully sent
            end

        end
    end

    // ── Pre-drive MISO at CS assertion ────────────────────────
    // Drive the first bit onto MISO as soon as CS goes low,
    // before the first SCLK edge, so the master sees valid data
    // on the very first rising edge. Combinational override.
    // (Only when a fresh load just happened — bit_cnt == 0)
    // This handles SPI masters that sample on the first rising
    // edge immediately after CS assertion.
    always @(*) begin
        if (cs_active && !transmitting && packet_valid)
            spi_miso = packet_data[PKT_BITS-1];   // preview MSB
        // (registered path handles all other cases)
    end

endmodule