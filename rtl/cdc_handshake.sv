// cdc_handshake.sv
// Acknowledged request/response mailbox between two unrelated clocks.
// Four-phase (level) protocol: req up, ack up, req down, ack down. Only the
// single-bit req and ack levels go through synchronizers; the request and
// response words cross as bundled data, sampled while known to be stable.
//
// Source side
//   src_send   One-cycle pulse. Accepted only while src_busy is low; ignored
//              otherwise.
//   src_data   Request word. NOT registered here: the caller must hold it
//              unchanged from the src_send cycle until src_busy falls.
//   src_busy   High from the cycle after src_send until the whole four-phase
//              cycle has finished. A new src_send is legal once it is low.
//   src_done   One-cycle pulse when the destination has executed the request.
//              src_rsp is valid from this cycle on. It comes before src_busy
//              falls, so busy low implies the response is already visible.
//
// Destination side
//   dst_valid  Level: one request is waiting. dst_data is this module's own
//              registered copy of the request word and stays stable.
//   dst_done   Pulse while dst_valid is high: the request has been executed.
//              dst_rsp is captured in the same cycle. dst_valid falls on the
//              next edge, so each request is presented exactly once.
//
// Reset: both sides reset to req = ack = 0 by their own domain's copy of the
// global reset. Do not reset one side alone. A request raised while the
// other side is still in reset, or while its clock is stopped, simply waits.
//
// CDC constraints (see syn/bmi_chip_top.sdc)
//   * src_data -> dst_data_q is sampled at least two dst_clk periods after
//     req_q and src_data were launched. Its delay, and its skew against the
//     req_q -> u_req_sync path, must stay under one dst_clk period.
//   * dst_rsp_q -> src_rsp_q likewise, against ack_q -> u_ack_sync, under one
//     src_clk period.
//   * req_q and ack_q are flip-flop outputs and feed only the synchronizers.

module cdc_handshake #(
    parameter int REQ_WIDTH = 1,
    parameter int RSP_WIDTH = 1
)(
    // ---- source domain ----
    input  wire                  src_clk,
    input  wire                  src_rst_n,
    input  wire                  src_send,
    input  wire [REQ_WIDTH-1:0]  src_data,
    output wire                  src_busy,
    output wire                  src_done,
    output wire [RSP_WIDTH-1:0]  src_rsp,

    // ---- destination domain ----
    input  wire                  dst_clk,
    input  wire                  dst_rst_n,
    output wire                  dst_valid,
    output wire [REQ_WIDTH-1:0]  dst_data,
    input  wire                  dst_done,
    input  wire [RSP_WIDTH-1:0]  dst_rsp
);

    logic                 req_q, req_d;            // src_clk
    logic                 done_q, done_d;          // src_clk
    logic [RSP_WIDTH-1:0] src_rsp_q, src_rsp_d;    // src_clk

    logic                 ack_q, ack_d;            // dst_clk
    logic                 valid_q, valid_d;        // dst_clk
    logic [REQ_WIDTH-1:0] dst_data_q, dst_data_d;  // dst_clk
    logic [RSP_WIDTH-1:0] dst_rsp_q, dst_rsp_d;    // dst_clk

    wire ack_sync;   // ack_q seen in src_clk
    wire req_sync;   // req_q seen in dst_clk

    cdc_sync_2ff u_ack_sync (
        .clk(src_clk), .rst_n(src_rst_n), .d(ack_q), .q(ack_sync)
    );

    cdc_sync_2ff u_req_sync (
        .clk(dst_clk), .rst_n(dst_rst_n), .d(req_q), .q(req_sync)
    );

    // ---- source side ----
    always_comb begin
        req_d     = req_q;
        done_d    = 1'b0;
        src_rsp_d = src_rsp_q;

        if (req_q && ack_sync) begin
            // dst_rsp_q has been stable since before ack_q rose.
            req_d     = 1'b0;
            done_d    = 1'b1;
            src_rsp_d = dst_rsp_q;
        end else if (!req_q && !ack_sync && src_send) begin
            req_d = 1'b1;
        end
    end

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            req_q     <= 1'b0;
            done_q    <= 1'b0;
            src_rsp_q <= '0;
        end else begin
            req_q     <= req_d;
            done_q    <= done_d;
            src_rsp_q <= src_rsp_d;
        end
    end

    assign src_busy = req_q || ack_sync;
    assign src_done = done_q;
    assign src_rsp  = src_rsp_q;

    // ---- destination side ----
    always_comb begin
        ack_d      = ack_q;
        valid_d    = valid_q;
        dst_data_d = dst_data_q;
        dst_rsp_d  = dst_rsp_q;

        if (ack_q) begin
            if (!req_sync)
                ack_d = 1'b0;
        end else if (valid_q) begin
            if (dst_done) begin
                valid_d   = 1'b0;
                ack_d     = 1'b1;
                dst_rsp_d = dst_rsp;
            end
        end else if (req_sync) begin
            // src_data has been stable since before req_q rose.
            valid_d    = 1'b1;
            dst_data_d = src_data;
        end
    end

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            ack_q      <= 1'b0;
            valid_q    <= 1'b0;
            dst_data_q <= '0;
            dst_rsp_q  <= '0;
        end else begin
            ack_q      <= ack_d;
            valid_q    <= valid_d;
            dst_data_q <= dst_data_d;
            dst_rsp_q  <= dst_rsp_d;
        end
    end

    assign dst_valid = valid_q;
    assign dst_data  = dst_data_q;

endmodule
