`timescale 1ns/1ps

// Permanent unit test for the streaming ADC/SBP frontend. Each case drives a
// complete round-robin ADC window and calculates the expected feature directly
// from the samples presented to the DUT.
module tb_streaming_sbp_frontend;

    localparam integer N_CH          = 8;
    localparam integer N_SAMPLES     = 250;
    localparam integer ADC_WIDTH     = 8;
    localparam integer SBP_WIDTH     = 8;
    localparam integer FEATURE_SHIFT = 8;
    localparam integer CLK_PERIOD    = 10;

    logic clk;
    logic rst_n;
    logic [ADC_WIDTH-1:0] adc_sample;
    logic [$clog2(N_CH)-1:0] adc_channel;
    logic resume;
    logic features_done;
    logic [SBP_WIDTH-1:0] sbp_features [0:N_CH-1];

    integer pass_count;
    integer fail_count;

    streaming_sbp_frontend #(
        .N_CH(N_CH), .N_SAMPLES(N_SAMPLES),
        .ADC_WIDTH(ADC_WIDTH), .SBP_WIDTH(SBP_WIDTH),
        .FEATURE_SHIFT(FEATURE_SHIFT)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .adc_sample(adc_sample), .adc_channel(adc_channel),
        .resume(resume),
        .features_done(features_done), .sbp_features(sbp_features)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Produce several useful windows without storing a second sample memory.
    function automatic [ADC_WIDTH-1:0] sample_for_case;
        input integer pattern;
        input integer channel;
        input integer sample_index;
        begin
            case (pattern)
                0: sample_for_case = 8'd128; // zero deviation
                1: begin                     // both ADC rails
                    if (channel == 0)      sample_for_case = 8'd0;
                    else if (channel == 1) sample_for_case = 8'd255;
                    else                   sample_for_case = 8'd128;
                end
                2: begin // samples 0 and 249 sum to exactly 256 deviation
                    if (sample_index == 0 || sample_index == N_SAMPLES-1)
                        sample_for_case = 8'd0;
                    else
                        sample_for_case = 8'd128;
                end
                default: sample_for_case =
                    (channel*31 + sample_index*7) & 8'hff;
            endcase
        end
    endfunction

    function automatic integer deviation;
        input integer sample_value;
        begin
            if (sample_value >= 128)
                deviation = sample_value - 128;
            else
                deviation = 128 - sample_value;
        end
    endfunction

    task automatic restart_collection;
        begin
            @(negedge clk);
            resume = 1'b1;
            @(posedge clk);
            #1;
            // Collection has restarted, but the next capture is still half a
            // cycle away. Deassert here so check_window can drive sample 0 at
            // that intervening falling edge.
            resume = 1'b0;
        end
    endtask

    task automatic check_window;
        input integer test_id;
        input integer pattern;
        integer capture;
        integer channel;
        integer sample_index [0:N_CH-1];
        integer expected_sum [0:N_CH-1];
        integer expected_feature;
        logic [ADC_WIDTH-1:0] driven_sample;
        logic case_pass;
        logic [$clog2(N_CH)-1:0] paused_channel;
        begin
            case_pass = 1'b1;
            for (channel = 0; channel < N_CH; channel = channel + 1) begin
                sample_index[channel] = 0;
                expected_sum[channel] = 0;
            end

            for (capture = 0; capture < N_CH*N_SAMPLES;
                 capture = capture + 1) begin
                @(negedge clk);
                channel = adc_channel;

                if (channel !== (capture % N_CH)) begin
                    $display("FAIL test %0d: capture %0d selected channel %0d, expected %0d",
                             test_id, capture, channel, capture % N_CH);
                    case_pass = 1'b0;
                end

                driven_sample = sample_for_case(
                    pattern, channel, sample_index[channel]);
                adc_sample = driven_sample;
                expected_sum[channel] = expected_sum[channel] +
                                        deviation(driven_sample);
                sample_index[channel] = sample_index[channel] + 1;

                @(posedge clk);
                #1;
            end

            if (features_done !== 1'b1) begin
                $display("FAIL test %0d: features_done missing on final capture",
                         test_id);
                case_pass = 1'b0;
            end

            for (channel = 0; channel < N_CH; channel = channel + 1) begin
                expected_feature = expected_sum[channel] >> FEATURE_SHIFT;
                if (sbp_features[channel] !== expected_feature[SBP_WIDTH-1:0]) begin
                    $display("FAIL test %0d: feature[%0d] got=%0d expected=%0d sum=%0d",
                             test_id, channel, sbp_features[channel],
                             expected_feature, expected_sum[channel]);
                    case_pass = 1'b0;
                end
            end

            // Completion is a pulse, and the ADC selector/features must remain
            // stable while the frontend waits for downstream packet completion.
            paused_channel = adc_channel;
            adc_sample = 8'd0;
            repeat (3) begin
                @(posedge clk);
                #1;
                if (features_done !== 1'b0 || adc_channel !== paused_channel) begin
                    $display("FAIL test %0d: frontend did not remain paused",
                             test_id);
                    case_pass = 1'b0;
                end
            end

            if (case_pass) begin
                $display("PASS test %0d: pattern=%0d features=[%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
                         test_id, pattern,
                         sbp_features[0], sbp_features[1],
                         sbp_features[2], sbp_features[3],
                         sbp_features[4], sbp_features[5],
                         sbp_features[6], sbp_features[7]);
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task automatic check_midwindow_reset;
        integer capture;
        begin
            restart_collection();
            for (capture = 0; capture < 37; capture = capture + 1) begin
                @(negedge clk);
                adc_sample = 8'd0;
                @(posedge clk);
            end

            @(negedge clk);
            rst_n = 1'b0;
            #1;
            if (features_done !== 1'b0 || adc_channel !== '0 ||
                sbp_features[0] !== '0) begin
                $display("FAIL test 5: asynchronous reset did not clear frontend");
                fail_count = fail_count + 1;
            end else begin
                $display("PASS test 5: asynchronous mid-window reset");
                pass_count = pass_count + 1;
            end
            @(negedge clk);
            rst_n = 1'b1;
        end
    endtask

    initial begin
        rst_n       = 1'b0;
        adc_sample  = 8'd128;
        resume      = 1'b0;
        pass_count  = 0;
        fail_count  = 0;

        repeat (3) @(posedge clk);
        #1;
        rst_n = 1'b1;

        check_window(1, 0); // midpoint
        restart_collection();
        check_window(2, 1); // both rails
        restart_collection();
        check_window(3, 2); // explicitly includes final sample
        restart_collection();
        check_window(4, 3); // different data on every channel
        check_midwindow_reset();

        $display("------------------------------------------------");
        $display("STREAMING FRONTEND TEST: %0d passed, %0d failed",
                 pass_count, fail_count);
        $display("------------------------------------------------");
        if (fail_count != 0)
            $fatal(1, "Streaming frontend regression failed");
        $display("ALL STREAMING FRONTEND TESTS PASS");
        $finish;
    end

    initial begin
        #(500000);
        $fatal(1, "Streaming frontend testbench timeout");
    end

endmodule
