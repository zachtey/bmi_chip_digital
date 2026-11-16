// weight_scan_loader.sv
// Serial (scan-chain) loader for the MLP weight image, used by bmi_chip_top.
//
// Weights are loaded once at power-up. For the default dimensions the image
// is 864 bits (108 bytes), shifted LSB-first, in this order:
//   hw[0..7][0..7], hb[0..7], ow[0..3][0..7], ob[0..3]
// After the last bit, byte i of the image is weight_image[i*8 +: 8].
//
// Load protocol (scan_clk domain)
//   1. Clock scan_clk once with scan_en low. This clears the bit counter and
//      the loaded flag, i.e. it begins a new load.
//   2. Shift exactly TOTAL_WEIGHT_BITS bits with scan_en high.
//   3. Stop clocking. The loaded flag stays set until the next clear clock.
//
// Clock domains
//   scan_reg, the bit counter and weights_loaded_scan are clocked by scan_clk.
//   weights_loaded is weights_loaded_scan passed through a two-flop
//   synchronizer into the clk domain. weight_image is NOT synchronized: it is
//   only used after weights_loaded is high, by which time it has been stable
//   for at least two clk cycles (weights are never reloaded at run time).

module weight_scan_loader #(
    parameter TOTAL_WEIGHT_BITS = 864
)(
    // Functional clock domain (destination of the synchronizer)
    input  wire                          clk,
    input  wire                          rst_n,

    // Scan chain
    input  wire                          scan_en,
    input  wire                          scan_clk,
    input  wire                          scan_in,

    // To mlp_inference
    output wire [TOTAL_WEIGHT_BITS-1:0]  weight_image,
    output logic                         weights_loaded
);

    //Scan chain loading
    logic [TOTAL_WEIGHT_BITS-1:0] scan_reg;
    localparam integer SCAN_COUNT_WIDTH = $clog2(TOTAL_WEIGHT_BITS + 1);
    localparam logic [SCAN_COUNT_WIDTH-1:0] SCAN_LAST_COUNT =
        SCAN_COUNT_WIDTH'(TOTAL_WEIGHT_BITS - 1);
    localparam logic [SCAN_COUNT_WIDTH-1:0] SCAN_COMPLETE_COUNT =
        SCAN_COUNT_WIDTH'(TOTAL_WEIGHT_BITS);
    logic [SCAN_COUNT_WIDTH-1:0] scan_bit_count;
    logic weights_loaded_scan;

    // Protocol: clock scan_clk once with scan_en low to begin/clear a load,
    // then shift exactly TOTAL_WEIGHT_BITS with scan_en high. The completed
    // indication remains asserted until the next explicit clear clock.
    always_ff @(posedge scan_clk) begin
        if (!scan_en) begin
            scan_bit_count     <= '0;
            weights_loaded_scan <= 1'b0;
        end else begin
            scan_reg <= {scan_in, scan_reg[TOTAL_WEIGHT_BITS-1:1]};
            if (!weights_loaded_scan) begin
                if (scan_bit_count == SCAN_LAST_COUNT) begin
                    scan_bit_count      <= SCAN_COMPLETE_COUNT;
                    weights_loaded_scan <= 1'b1;
                end else begin
                    scan_bit_count <= scan_bit_count + 1'b1;
                end
            end
        end
    end

    // weights_loaded_scan is stable after loading and crosses into the
    // functional clock domain through a standard two-flop level synchronizer.
    logic weights_loaded_meta;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_loaded_meta <= 1'b0;
            weights_loaded      <= 1'b0;
        end else begin
            weights_loaded_meta <= weights_loaded_scan;
            weights_loaded      <= weights_loaded_meta;
        end
    end

    assign weight_image = scan_reg;

endmodule
