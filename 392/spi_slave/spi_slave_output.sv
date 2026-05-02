`default_nettype none
`timescale 1ns/1ps

// ============================================================
// spi_slave_output.sv
//
// SPI Mode 0 (CPOL=0, CPHA=0) slave transmitter.
// Packet : 8 bits = { 5'b00000, result_valid, CLASS[1:0] }
// Bit order : MSB first
// CS        : active-low (spi_cs_n_i)
// Clock     : external SPI master clock (spi_sclk_i)
//             — no internal divider
//
// Protocol timing:
//   Master drives CS low, then toggles SCLK.
//   On each RISING edge of SCLK the master samples MISO.
//   We therefore update MISO on the FALLING edge of SCLK
//   (or when CS first goes low) so data is stable at the
//   next rising edge.
//
// Load behaviour:
//   When load_i is pulsed (one sys-clk cycle), the 8-bit
//   tx_byte_i is latched into the shift register so it is
//   ready before the master begins clocking.
//
// Two-process FSM (sequential + combinational).
// ============================================================

module spi_slave_output (
    // System domain
    input  logic       clk,
    input  logic       rst_n,

    // Parallel load interface (driven by output_formatter)
    input  logic       load_i,          // pulse: latch tx_byte_i now
    input  logic [7:0] tx_byte_i,       // { 5'b0, result_valid, CLASS[1:0] }

    // SPI bus (external master clock domain)
    input  logic       spi_sclk_i,      // SPI clock from master
    input  logic       spi_cs_n_i,      // chip-select, active-low

    output logic       spi_miso_o,      // master-in slave-out

    // Status back to system
    output logic       tx_done_o        // pulses when all 8 bits shifted out
);

    // --------------------------------------------------------
    // SPI clock edge detection (sys-clk domain)
    // --------------------------------------------------------
    logic sclk_r, sclk_rr;
    logic cs_n_r, cs_n_rr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_r  <= 1'b0;
            sclk_rr <= 1'b0;
            cs_n_r  <= 1'b1;
            cs_n_rr <= 1'b1;
        end else begin
            sclk_r  <= spi_sclk_i;
            sclk_rr <= sclk_r;
            cs_n_r  <= spi_cs_n_i;
            cs_n_rr <= cs_n_r;
        end
    end

    logic sclk_rise;   // one sys-clk pulse on SPI rising  edge
    logic sclk_fall;   // one sys-clk pulse on SPI falling edge
    logic cs_falling;  // one sys-clk pulse when CS goes low

    assign sclk_rise  =  sclk_r & ~sclk_rr;
    assign sclk_fall  = ~sclk_r &  sclk_rr;
    assign cs_falling = ~cs_n_r &  cs_n_rr;

    // --------------------------------------------------------
    // FSM state definition
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE  = 2'b00,
        LOAD  = 2'b01,
        SHIFT = 2'b10,
        DONE  = 2'b11
    } state_t;

    state_t state, next_state;

    // --------------------------------------------------------
    // Datapath registers
    // --------------------------------------------------------
    logic [7:0] shift_reg;          // transmit shift register
    logic [2:0] bit_cnt;            // counts 0-7 (8 bits)
    logic [7:0] tx_latch;           // holds byte between load and CS

    // --------------------------------------------------------
    // Process 1 : sequential state + datapath
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            shift_reg <= 8'h00;
            tx_latch  <= 8'h00;
            bit_cnt   <= 3'd0;
        end else begin
            // Always capture incoming byte when load is pulsed
            if (load_i)
                tx_latch <= tx_byte_i;

            state <= next_state;

            case (state)
                // -------------------------------------------------
                IDLE: begin
                    bit_cnt <= 3'd0;
                    // When CS falls, pre-load shift register so
                    // MSB is ready before the first SCLK rise.
                    if (cs_falling)
                        shift_reg <= tx_latch;
                end

                // -------------------------------------------------
                // LOAD is a one-cycle pass-through; shift_reg was
                // already set in IDLE on cs_falling. Nothing extra
                // needed here except transition to SHIFT.
                LOAD: begin
                    bit_cnt <= 3'd0;
                end

                // -------------------------------------------------
                SHIFT: begin
                    if (sclk_fall && !cs_n_rr) begin
                        // Shift out next bit on falling edge
                        // so master sees stable data on next rise.
                        shift_reg <= {shift_reg[6:0], 1'b0};
                        bit_cnt   <= bit_cnt + 3'd1;
                    end
                end

                // -------------------------------------------------
                DONE: begin
                    bit_cnt <= 3'd0;
                end

                default: ;
            endcase
        end
    end

    // --------------------------------------------------------
    // Process 2 : combinational next-state
    // --------------------------------------------------------
    always_comb begin
        next_state = state;
        case (state)
            IDLE:  if (cs_falling)               next_state = LOAD;
            LOAD:                                 next_state = SHIFT;
            SHIFT: if (bit_cnt == 3'd7 &&
                       sclk_fall && !cs_n_rr)     next_state = DONE;
            DONE:  if (cs_n_rr)                   next_state = IDLE;
            default:                              next_state = IDLE;
        endcase
    end

    // --------------------------------------------------------
    // Output assignments
    // --------------------------------------------------------
    // MISO always drives MSB of shift register.
    // When CS is deasserted, drive 0 (hi-z in real silicon —
    // use a tristate buffer at the pad level).
    assign spi_miso_o = (!cs_n_rr) ? shift_reg[7] : 1'b0;
    assign tx_done_o  = (state == DONE);

endmodule

`default_nettype wire