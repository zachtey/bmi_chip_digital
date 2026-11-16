// cdc_async_fifo.sv
// Dual-clock FIFO with Gray-coded pointers (the classic two-pointer design:
// binary counters for addressing, Gray copies for crossing).
//
//   Depth is 2**ADDR_WIDTH. Pointers are ADDR_WIDTH+1 bits wide so that full
//   and empty can be told apart.
//   wfull   registered in wclk. Conservative: it may stay high for a couple
//           of wclk cycles after a read. A write while full is ignored.
//   rempty  registered in rclk. Conservative in the other direction. A read
//           while empty is ignored.
//   rdata   combinational read of the entry at the read pointer (first word
//           falls through). Valid whenever rempty is low, and held until the
//           cycle after rd_en.
//
// Reset: each side is reset by its own domain's copy of the global reset and
// both must be reset together. Never reset just one pointer.
//
// CDC constraints (see syn/bmi_chip_top.sdc)
//   * wgray_q and rg// cdc_async_fifo.sv
// Dual-clock FIFO with Gray-coded pointers (the classic two-pointer design:
// binary counters for addressing, Gray copies for crossing).
//
//   Depth is 2**ADDR_WIDTH. Pointers are ADDR_WIDTH+1 bits wide so that full
//   and empty can be told apart.
//   wfull   registered in wclk. Conservative: it may stay high for a couple
//           of wclk cycles after a read. A write while full is ignored.
//   rempty  registered in rclk. Conservative in the other direction. A read
//           while empty is ignored.
//   rdata   combinational read of the entry at the read pointer (first word
//           falls through). Valid whenever rempty is low, and held until the
//           cycle after rd_en.
//
// Reset: each side is reset by its own domain's copy of the global reset and
// both must be reset together. Never reset just one pointer.
//
// CDC constraints (see syn/bmi_chip_top.sdc)
//   * wgray_q and rgray_q are flip-flop outputs feeding only synchronizers.
//     Skew between the bits of a Gray pointer must stay under one source
//     clock period so that at most one bit is in transition when sampled.
//   * mem -> rdata is bundled data: an entry is written at least two rclk
//     periods before rempty can fall for it. The mem -> rclk capture path
//     must stay under one rclk period.

module cdc_async_fifo #(
    parameter int WIDTH      = 8,
    parameter int ADDR_WIDTH = 2      // depth = 2**ADDR_WIDTH, must be >= 2
)(
    // ---- write domain ----
    input  wire              wclk,
    input  wire              wrst_n,
    input  wire              wr_en,
    input  wire [WIDTH-1:0]  wdata,
    output wire              wfull,

    // ---- read domain ----
    input  wire              rclk,
    input  wire              rrst_n,
    input  wire              rd_en,
    output wire [WIDTH-1:0]  rdata,
    output wire              rempty
);

    localparam int DEPTH = 1 << ADDR_WIDTH;

    // Storage: written in wclk, no reset (never read before written).
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    logic [ADDR_WIDTH:0] wbin_q, wgray_q;   // wclk
    logic [ADDR_WIDTH:0] rbin_q, rgray_q;   // rclk
    logic                wfull_q;           // wclk
    logic                rempty_q;          // rclk

    wire [ADDR_WIDTH:0] rgray_wclk;   // read pointer seen in wclk
    wire [ADDR_WIDTH:0] wgray_rclk;   // write pointer seen in rclk

    cdc_sync_2ff #(.WIDTH(ADDR_WIDTH+1)) u_rptr_sync (
        .clk(wclk), .rst_n(wrst_n), .d(rgray_q), .q(rgray_wclk)
    );

    cdc_sync_2ff #(.WIDTH(ADDR_WIDTH+1)) u_wptr_sync (
        .clk(rclk), .rst_n(rrst_n), .d(wgray_q), .q(wgray_rclk)
    );

    // ---- write side ----
    wire                wr_accept = wr_en && !wfull_q;
    wire [ADDR_WIDTH:0] wbin_d    = wbin_q + {{ADDR_WIDTH{1'b0}}, wr_accept};
    wire [ADDR_WIDTH:0] wgray_d   = (wbin_d >> 1) ^ wbin_d;
    // Full: the next write pointer is exactly one lap ahead of the read
    // pointer, which in Gray code means the top two bits are inverted.
    wire                wfull_d   =
        (wgray_d == {~rgray_wclk[ADDR_WIDTH:ADDR_WIDTH-1],
                      rgray_wclk[ADDR_WIDTH-2:0]});

    always_ff @(posedge wclk or negedge wrst_n) begin
        if (!wrst_n) begin
            wbin_q  <= '0;
            wgray_q <= '0;
            wfull_q <= 1'b0;
        end else begin
            wbin_q  <= wbin_d;
            wgray_q <= wgray_d;
            wfull_q <= wfull_d;
        end
    end

    always_ff @(posedge wclk) begin
        if (wr_accept)
            mem[wbin_q[ADDR_WIDTH-1:0]] <= wdata;
    end

    assign wfull = wfull_q;

    // ---- read side ----
    wire                rd_accept = rd_en && !rempty_q;
    wire [ADDR_WIDTH:0] rbin_d    = rbin_q + {{ADDR_WIDTH{1'b0}}, rd_accept};
    wire [ADDR_WIDTH:0] rgray_d   = (rbin_d >> 1) ^ rbin_d;
    wire                rempty_d  = (rgray_d == wgray_rclk);

    always_ff @(posedge rclk or negedge rrst_n) begin
        if (!rrst_n) begin
            rbin_q   <= '0;
            rgray_q  <= '0;
            rempty_q <= 1'b1;
        end else begin
            rbin_q   <= rbin_d;
            rgray_q  <= rgray_d;
            rempty_q <= rempty_d;
        end
    end

    assign rdata  = mem[rbin_q[ADDR_WIDTH-1:0]];
    assign rempty = rempty_q;

`ifndef SYNTHESIS
    initial begin
        if (ADDR_WIDTH < 2)
            $fatal(1, "cdc_async_fifo: ADDR_WIDTH must be at least 2");
    end
`endif

endmodule
ray_q are flip-flop outputs feeding only synchronizers.
//     Skew between the bits of a Gray pointer must stay under one source
//     clock period so that at most one bit is in transition when sampled.
//   * mem -> rdata is bundled data: an entry is written at least two rclk
//     periods before rempty can fall for it. The mem -> rclk capture path
//     must stay under one rclk period.

module cdc_async_fifo #(
    parameter int WIDTH      = 8,
    parameter int ADDR_WIDTH = 2      // depth = 2**ADDR_WIDTH, must be >= 2
)(
    // ---- write domain ----
    input  wire              wclk,
    input  wire              wrst_n,
    input  wire              wr_en,
    input  wire [WIDTH-1:0]  wdata,
    output wire              wfull,

    // ---- read domain ----
    input  wire              rclk,
    input  wire              rrst_n,
    input  wire              rd_en,
    output wire [WIDTH-1:0]  rdata,
    output wire              rempty
);

    localparam int DEPTH = 1 << ADDR_WIDTH;

    // Storage: written in wclk, no reset (never read before written).
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    logic [ADDR_WIDTH:0] wbin_q, wgray_q;   // wclk
    logic [ADDR_WIDTH:0] rbin_q, rgray_q;   // rclk
    logic                wfull_q;           // wclk
    logic                rempty_q;          // rclk

    wire [ADDR_WIDTH:0] rgray_wclk;   // read pointer seen in wclk
    wire [ADDR_WIDTH:0] wgray_rclk;   // write pointer seen in rclk

    cdc_sync_2ff #(.WIDTH(ADDR_WIDTH+1)) u_rptr_sync (
        .clk(wclk), .rst_n(wrst_n), .d(rgray_q), .q(rgray_wclk)
    );

    cdc_sync_2ff #(.WIDTH(ADDR_WIDTH+1)) u_wptr_sync (
        .clk(rclk), .rst_n(rrst_n), .d(wgray_q), .q(wgray_rclk)
    );

    // ---- write side ----
    wire                wr_accept = wr_en && !wfull_q;
    wire [ADDR_WIDTH:0] wbin_d    = wbin_q + {{ADDR_WIDTH{1'b0}}, wr_accept};
    wire [ADDR_WIDTH:0] wgray_d   = (wbin_d >> 1) ^ wbin_d;
    // Full: the next write pointer is exactly one lap ahead of the read
    // pointer, which in Gray code means the top two bits are inverted.
    wire                wfull_d   =
        (wgray_d == {~rgray_wclk[ADDR_WIDTH:ADDR_WIDTH-1],
                      rgray_wclk[ADDR_WIDTH-2:0]});

    always_ff @(posedge wclk or negedge wrst_n) begin
        if (!wrst_n) begin
            wbin_q  <= '0;
            wgray_q <= '0;
            wfull_q <= 1'b0;
        end else begin
            wbin_q  <= wbin_d;
            wgray_q <= wgray_d;
            wfull_q <= wfull_d;
        end
    end

    always_ff @(posedge wclk) begin
        if (wr_accept)
            mem[wbin_q[ADDR_WIDTH-1:0]] <= wdata;
    end

    assign wfull = wfull_q;

    // ---- read side ----
    wire                rd_accept = rd_en && !rempty_q;
    wire [ADDR_WIDTH:0] rbin_d    = rbin_q + {{ADDR_WIDTH{1'b0}}, rd_accept};
    wire [ADDR_WIDTH:0] rgray_d   = (rbin_d >> 1) ^ rbin_d;
    wire                rempty_d  = (rgray_d == wgray_rclk);

    always_ff @(posedge rclk or negedge rrst_n) begin
        if (!rrst_n) begin
            rbin_q   <= '0;
            rgray_q  <= '0;
            rempty_q <= 1'b1;
        end else begin
            rbin_q   <= rbin_d;
            rgray_q  <= rgray_d;
            rempty_q <= rempty_d;
        end
    end

    assign rdata  = mem[rbin_q[ADDR_WIDTH-1:0]];
    assign rempty = rempty_q;

`ifndef SYNTHESIS
    initial begin
        if (ADDR_WIDTH < 2)
            $fatal(1, "cdc_async_fifo: ADDR_WIDTH must be at least 2");
    end
`endif

endmodule
