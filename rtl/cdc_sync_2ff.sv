// cdc_sync_2ff.sv
// Two-flop level synchronizer. Use only for single bits or Gray-coded buses
// (at most one bit changing per source clock). d must come straight from a
// flip-flop in the source domain, with no combinational logic in between.
//
// Constraints: the path into meta_q is asynchronous (exclude it from normal
// timing, but bound it with set_max_delay -datapath_only). Place meta_q and
// sync_q adjacent, keep both off scan-reordering and retiming, and use
// synchronizer-grade flops if the library has them.

module cdc_sync_2ff #(
    parameter int WIDTH = 1
)(
    input  wire             clk,     // destination clock
    input  wire             rst_n,   // destination-domain reset
    input  wire [WIDTH-1:0] d,       // from the other clock domain
    output wire [WIDTH-1:0] q
);

    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic [WIDTH-1:0] meta_q;
    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic [WIDTH-1:0] sync_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            meta_q <= '0;
            sync_q <= '0;
        end else begin
            meta_q <= d;
            sync_q <= meta_// cdc_sync_2ff.sv
// Two-flop level synchronizer. Use only for single bits or Gray-coded buses
// (at most one bit changing per source clock). d must come straight from a
// flip-flop in the source domain, with no combinational logic in between.
//
// Constraints: the path into meta_q is asynchronous (exclude it from normal
// timing, but bound it with set_max_delay -datapath_only). Place meta_q and
// sync_q adjacent, keep both off scan-reordering and retiming, and use
// synchronizer-grade flops if the library has them.

module cdc_sync_2ff #(
    parameter int WIDTH = 1
)(
    input  wire             clk,     // destination clock
    input  wire             rst_n,   // destination-domain reset
    input  wire [WIDTH-1:0] d,       // from the other clock domain
    output wire [WIDTH-1:0] q
);

    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic [WIDTH-1:0] meta_q;
    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic [WIDTH-1:0] sync_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            meta_q <= '0;
            sync_q <= '0;
        end else begin
            meta_q <= d;
            sync_q <= meta_q;
        end
    end

    assign q = sync_q;

endmodule
q;
        end
    end

    assign q = sync_q;

endmodule
