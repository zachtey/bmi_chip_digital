// output_formatter.sv
// Packs predicted_class and class_scores into a 10-byte SPI packet.
//
// Packet layout (MSB-first, byte 0 sent first):
//   Byte 0:    0xAA               sync header
//   Byte 1:    {6'b0, class[1:0]} predicted class index
//   Bytes 2-3: score[0][31:16]    PG-LF
//   Bytes 4-5: score[1][31:16]    PG-HF
//   Bytes 6-7: score[2][31:16]    SG-LF
//   Bytes 8-9: score[3][31:16]    SG-HF
//
// Handshake: decision_valid (1-cycle pulse) sets packet_valid (level);
//            packet_ready   (1-cycle pulse from spi_slave) clears it.

module output_formatter #(
    parameter N_CLASSES   = 4,
    parameter SCORE_WIDTH = 32,
    parameter PKT_BYTES   = 10
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [1:0]                    predicted_class,
    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_CLASSES-1],
    input  wire                          decision_valid,

    output logic [PKT_BYTES*8-1:0]       packet_data,
    output logic                         packet_valid, //to spi slave; packet_data has complete packet that has not been consumed
    input  wire                          packet_ready //from spi slave; the spi slave finished consuming that packet
);
    //packet_ready, predicted_class, class_scores, decision_valid applied as inputs from driver in tb

    localparam [7:0] SYNC = 8'hAA;

    // Next-cycle values
    logic [PKT_BYTES*8-1:0] next_packet_data;
    logic                    next_packet_valid;

    // Combinational: compute next packet_data and packet_valid
    always_comb begin
        next_packet_data  = packet_data;
        next_packet_valid = packet_valid;

        if (decision_valid) begin
            next_packet_data = {
                SYNC,
                6'b000000, predicted_class,
                class_scores[0][SCORE_WIDTH-1 -: 16],
                class_scores[1][SCORE_WIDTH-1 -: 16],
                class_scores[2][SCORE_WIDTH-1 -: 16],
                class_scores[3][SCORE_WIDTH-1 -: 16]
            };
            next_packet_valid = 1'b1;
        end

        // packet_ready takes priority — clears even if decision_valid fires same cycle
        if (packet_ready) //spi slave finished consuming the packet
            next_packet_valid = 1'b0;
    end

    // Sequential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            packet_data  <= '0;
            packet_valid <= 1'b0;
        end else begin
            packet_data  <= next_packet_data;
            packet_valid <= next_packet_valid;
        end
    end

endmodule
