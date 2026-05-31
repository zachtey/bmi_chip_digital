// sram_sample_win.sv
//
// Behavioral single-port SRAM for the 8-channel × 250-sample window.
// Depth = N_CH * N_SAMPLES = 2000 words, width = ADC_WIDTH = 8 bits.
// Address map: addr = channel * N_SAMPLES + sample_index
//
// Write: synchronous (posedge clk, wr_en gated)
// Read:  asynchronous (combinational output, zero latency)
//        → sbp_feature_extractor can use rd_data directly, no pipeline needed
//
// Synthesis hint: (* syn_ramstyle *) asks Genus to infer a memory structure
// rather than individual flip-flops.  Falls back to registers if no macro matches.

module sram_sample_win #(
    parameter N_CH      = 8,
    parameter N_SAMPLES = 250,
    parameter DATA_W    = 8
)(
    input  wire                              clk,

    // Write port  (driven by sample_collection)
    input  wire                              wr_en,
    input  wire [$clog2(N_CH*N_SAMPLES)-1:0] wr_addr,
    input  wire [DATA_W-1:0]                 wr_data,

    // Read port   (driven by sbp_feature_extractor)
    input  wire [$clog2(N_CH*N_SAMPLES)-1:0] rd_addr,
    output wire [DATA_W-1:0]                 rd_data
);
    localparam DEPTH = N_CH * N_SAMPLES;   // 2000

    (* syn_ramstyle = "latch_array" *)
    (* ram_style    = "block"       *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Synchronous write
    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    // Asynchronous (combinational) read — no latency penalty for sbp
    assign rd_data = mem[rd_addr];

endmodule
