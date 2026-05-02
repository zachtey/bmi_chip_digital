// ============================================================
// output_formatter.v
//
// Packs predicted_class and class_scores into a 10-byte
// serial packet for the SPI slave to transmit.
//
// Packet format (10 bytes, MSB-first):
//   Byte 0:    8'hAA            sync header
//   Byte 1:    {6'b0, class[1:0]}  predicted class
//   Byte 2-3:  score[0][31:16]  PG-LF  (upper 2 bytes)
//   Byte 4-5:  score[1][31:16]  PG-HF
//   Byte 6-7:  score[2][31:16]  SG-LF
//   Byte 8-9:  score[3][31:16]  SG-HF
//
// Handshake:
//   decision_valid (1-cycle pulse in) → packet_valid (level out)
//   packet_ready   (1-cycle pulse in from SPI slave when done)
//     → deasserts packet_valid, ready for next inference
// ============================================================
module output_formatter #(
    parameter N_CLASSES   = 4,
    parameter SCORE_WIDTH = 32,
    parameter PKT_BYTES   = 10
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // From argmax
    input  wire [1:0]                    predicted_class,
    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1],
    input  wire                          decision_valid,   // 1-cycle pulse

    // To SPI slave
    output reg  [PKT_BYTES*8-1:0]        packet_data,     // 80-bit shift register
    output reg                           packet_valid,    // level: packet is ready to send

    // From SPI slave: asserted when last bit has been shifted out
    input  wire                          packet_ready     // 1-cycle pulse
);

    // ── Packet assembly ───────────────────────────────────────
    // Triggered on decision_valid. Assembles all fields in one
    // clock cycle — pure combinational packing into a register.
    //
    // The packet is stored MSB-first so the SPI slave can shift
    // out packet_data[79] first without any bit reversal.

    localparam [7:0] SYNC = 8'hAA;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            packet_data  <= '0;
            packet_valid <= 1'b0;

        end else begin

            if (decision_valid) begin
                // ── Pack fields into packet ───────────────────
                packet_data <= {
                    SYNC,                                    // byte 0: sync
                    6'b000000, predicted_class,              // byte 1: class
                    class_scores[0][SCORE_WIDTH-1 -: 16],   // bytes 2-3: score 0
                    class_scores[1][SCORE_WIDTH-1 -: 16],   // bytes 4-5: score 1
                    class_scores[2][SCORE_WIDTH-1 -: 16],   // bytes 6-7: score 2
                    class_scores[3][SCORE_WIDTH-1 -: 16]    // bytes 8-9: score 3
                };
                packet_valid <= 1'b1;
            end

            // ── Deassert when SPI slave finishes sending ──────
            if (packet_ready) begin
                packet_valid <= 1'b0;
            end

        end
    end

endmodule