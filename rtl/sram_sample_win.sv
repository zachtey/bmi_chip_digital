// sram_sample_win.sv
// Behavioral single-port SRAM for the 8-channel x 250-sample window.
// Depth = N_CH * N_SAMPLES = 2000 words, width = ADC_WIDTH = 8 bits.
// Address encoding: addr = channel * N_SAMPLES + sample_index
//
// Write: synchronous on posedge clk.
// Read:  asynchronous (combinational) — zero read latency for sbp_feature_extractor.
//
// syn_ramstyle / ram_style attributes ask the synthesis tool to infer a
// memory macro rather than a flip-flop array. Falls back to registers if
// no matching macro is available.

module sram_sample_win #(
    parameter N_CH      = 8,
    parameter N_SAMPLES = 250,
    parameter DATA_W    = 8
)(
    input  wire                               clk,

    // Write port (driven by sample_collection)
    input  wire                               wr_en,
    input  wire [$clog2(N_CH*N_SAMPLES)-1:0] wr_addr,
    input  wire [DATA_W-1:0]                  wr_data,

    // Read port (driven by sbp_feature_extractor)
    input  wire [$clog2(N_CH*N_SAMPLES)-1:0] rd_addr,
    output wire [DATA_W-1:0]                  rd_data
);
    localparam DEPTH = N_CH * N_SAMPLES;

    (* syn_ramstyle = "latch_array" *)
    (* ram_style    = "block"       *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    assign rd_data = mem[rd_addr];

endmodule
