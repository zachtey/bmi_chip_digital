// Virtuoso symbol import stub — do NOT use for synthesis or PNR
// Import into Virtuoso to create a symbol with VDD/VSS pins
module bmi_chip_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  adc_sample,
    output wire [2:0]  adc_channel,
    input  wire        spi_sclk,
    input  wire        spi_cs_n,
    output wire        spi_miso,
    input  wire        scan_en,
    input  wire        scan_clk,
    input  wire        scan_in,
    inout  wire        VDD,
    inout  wire        VSS
);
endmodule
