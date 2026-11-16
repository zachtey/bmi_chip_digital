// reset_synchronizer.sv
// Brings an asynchronous, active-low reset into one clock domain:
// asserted asynchronously, released on the second rising edge of clk.
//
// arst_n must be glitch free. A low pulse of any width is caught. If clk is
// stopped, rst_n_out stays low until two edges have been seen.

module reset_synchronizer (
    input  wire  clk,        // destination clock
    input  wire  arst_n,     // asynchronous reset in, active low
    output wire  rst_n_out   // reset for the clk domain
);

    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic meta_q;
    (* ASYNC_REG = "TRUE", dont_touch = "true" *) logic sync_q;

    always_ff @(posedge clk or negedge arst_n) begin
        if (!arst_n) begin
            meta_q <= 1'b0;
            sync_q <= 1'b0;
        end else begin
            meta_q <= 1'b1;
            sync_q <= meta_q;
        end
    end

    assign rst_n_out = sync_q;

endmodule
