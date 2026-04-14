module test(
    input wire clk,
    input wire rst_n,
    output wire output_signal
);

always @(posedge clk) begin
    if (!rst_n) begin
        output_signal = 0;
    end else begin
        output_signal = 1;
    end
end
    
endmodule