`default_nettype none
`timescale 1ns/1ps

// ============================================================
// output_formatter.sv
//
// Waits for result_valid to pulse, latches class_i, packs
// the SPI byte, pulses load_o for one cycle, then waits for
// the SPI block to finish before returning to IDLE.
//
// Packet format:
//   tx_byte_o = { 5'b00000, 1'b1, class_i[1:0] }
//
// Examples:
//   class=0 → 0x04   (5'b00000, 1, 00)
//   class=1 → 0x05   (5'b00000, 1, 01)
//   class=2 → 0x06   (5'b00000, 1, 10)
//   class=3 → 0x07   (5'b00000, 1, 11)
//
// FSM:
//   IDLE → PACK → LOAD → WAIT_SPI_DONE → IDLE
//
// Two-process FSM (sequential + combinational).
// ============================================================

module output_formatter (
    input  logic       clk,
    input  logic       rst_n,

    // From argmax / MLP
    input  logic       result_valid_i,   // pulses high one cycle when result ready
    input  logic [1:0] class_i,          // predicted class 0-3

    // From SPI slave
    input  logic       tx_done_i,        // SPI finished shifting out byte

    // To SPI slave
    output logic [7:0] tx_byte_o,        // packed SPI byte
    output logic       load_o            // one-cycle pulse: latch tx_byte_o now
);

    // --------------------------------------------------------
    // FSM state definition
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE         = 2'b00,
        PACK         = 2'b01,
        LOAD         = 2'b10,
        WAIT_SPI_DONE = 2'b11
    } state_t;

    state_t state, next_state;

    // --------------------------------------------------------
    // Internal registers
    // --------------------------------------------------------
    logic [1:0] class_latch;    // holds class_i after result_valid pulse
    logic [7:0] byte_latch;     // holds packed byte

    // --------------------------------------------------------
    // Process 1 : sequential — state + datapath
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            class_latch <= 2'b00;
            byte_latch  <= 8'h00;
        end else begin
            state <= next_state;

            case (state)
                // -------------------------------------------------
                IDLE: begin
                    // Latch class the moment valid pulses
                    if (result_valid_i)
                        class_latch <= class_i;
                end

                // -------------------------------------------------
                // Pack: build the byte from the latched class
                PACK: begin
                    byte_latch <= {5'b00000, 1'b1, class_latch};
                end

                // -------------------------------------------------
                // LOAD: drive load_o for exactly this one cycle
                // (output logic below handles the pulse)
                LOAD: ;

                // -------------------------------------------------
                // WAIT_SPI_DONE: hold until SPI finishes
                WAIT_SPI_DONE: ;

                default: ;
            endcase
        end
    end

    // --------------------------------------------------------
    // Process 2 : combinational — next-state logic
    // --------------------------------------------------------
    always_comb begin
        next_state = state;
        case (state)
            IDLE:          if (result_valid_i)  next_state = PACK;
            PACK:                               next_state = LOAD;
            LOAD:                               next_state = WAIT_SPI_DONE;
            WAIT_SPI_DONE: if (tx_done_i)       next_state = IDLE;
            default:                            next_state = IDLE;
        endcase
    end

    // --------------------------------------------------------
    // Output assignments
    // --------------------------------------------------------
    assign tx_byte_o = byte_latch;
    assign load_o    = (state == LOAD);   // naturally one cycle wide

endmodule

`default_nettype wire