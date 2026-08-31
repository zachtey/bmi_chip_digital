`timescale 1ns/1ps

// Unit testbench for mlp_inference.
//
// The production model is already exercised by tb_bmi_chip_top. This bench
// instead loads small synthetic models whose arithmetic is easy to understand.
// That isolates scan loading, signed MAC operations, bias scaling, ReLU,
// control latency, and the done pulse from the rest of the chip.
module tb_mlp_inference;

    // Testbench and DUT configuration
    localparam integer CLK_PERIOD         = 10;
    localparam integer SCAN_PERIOD        = 20;
    localparam integer N_IN               = 8;
    localparam integer N_HIDDEN           = 8;
    localparam integer N_OUT              = 4;
    localparam integer IN_WIDTH           = 8;
    localparam integer W_WIDTH            = 8;
    localparam integer ACC_WIDTH          = 32;
    localparam integer SCORE_WIDTH        = 32;

    // Small scales keep the directed arithmetic easy to calculate by hand.
    localparam integer HIDDEN_BIAS_SCALE  = 3;
    localparam integer OUTPUT_BIAS_SCALE  = 5;

    localparam integer TOTAL_WEIGHT_BYTES =
        N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_IN + N_OUT;
    localparam integer TOTAL_WEIGHT_BITS  = TOTAL_WEIGHT_BYTES * W_WIDTH;

    // From the edge that accepts start to the edge that asserts done:
    //   8 hidden neurons * (8 MAC + bias + ReLU) = 80 cycles
    //   4 output neurons * (8 MAC + bias + store) = 40 cycles
    //   S_DONE                                      =  1 cycle
    localparam integer EXPECTED_LATENCY = 121;
    localparam integer MAX_WAIT_CYCLES  = EXPECTED_LATENCY + 10;

    // DUT interface
    logic clk;
    logic rst_n;

    logic start;
    logic done;
    logic [IN_WIDTH-1:0] sbp_features [0:N_IN-1];
    logic signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1];

    logic scan_en;
    logic scan_clk;
    logic scan_in;

    integer pass_count;
    integer fail_count;

    // Testbench copy of the programmed model
    // These arrays describe the weights that load_model shifts into the DUT.
    // They are also consumed by the independent procedural reference model.
    logic signed [W_WIDTH-1:0] model_hw [0:N_HIDDEN-1][0:N_IN-1];
    logic signed [W_WIDTH-1:0] model_hb [0:N_HIDDEN-1];
    logic signed [W_WIDTH-1:0] model_ow [0:N_OUT-1][0:N_HIDDEN-1];
    logic signed [W_WIDTH-1:0] model_ob [0:N_OUT-1];

    logic signed [ACC_WIDTH-1:0] expected_hidden [0:N_HIDDEN-1];
    logic signed [SCORE_WIDTH-1:0] expected_scores [0:N_OUT-1];

    // DUT
    mlp_inference #(
        .N_IN              (N_IN),
        .N_HIDDEN          (N_HIDDEN),
        .N_OUT             (N_OUT),
        .IN_WIDTH          (IN_WIDTH),
        .W_WIDTH           (W_WIDTH),
        .ACC_WIDTH         (ACC_WIDTH),
        .SCORE_WIDTH       (SCORE_WIDTH),
        .HIDDEN_BIAS_SCALE (HIDDEN_BIAS_SCALE),
        .OUTPUT_BIAS_SCALE (OUTPUT_BIAS_SCALE)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .done         (done),
        .sbp_features (sbp_features),
        .class_scores (class_scores),
        .scan_en      (scan_en),
        .scan_clk     (scan_clk),
        .scan_in      (scan_in)
    );

    // System clock. The scan clock is driven manually by load_model.
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Model construction helpers

    // Clear every software-model parameter before constructing a new case.
    task automatic clear_model;
        integer hidden_idx;
        integer input_idx;
        integer output_idx;
        begin
            for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1) begin
                model_hb[hidden_idx] = '0;
                for (input_idx = 0; input_idx < N_IN; input_idx = input_idx + 1)
                    model_hw[hidden_idx][input_idx] = '0;
            end

            for (output_idx = 0; output_idx < N_OUT; output_idx = output_idx + 1) begin
                model_ob[output_idx] = '0;
                for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                    model_ow[output_idx][hidden_idx] = '0;
            end
        end
    endtask

    // Model 1: hidden[j] equals input[j]. The output layer selects, negates,
    // or adds known hidden values and also exercises output bias scaling.
    task automatic setup_identity_model;
        integer idx;
        begin
            clear_model();

            for (idx = 0; idx < N_IN; idx = idx + 1)
                model_hw[idx][idx] = 8'sd1;

            model_ow[0][0] =  8'sd1;
            model_ow[1][1] = -8'sd2;
            model_ow[2][2] =  8'sd1;
            model_ow[2][3] =  8'sd1;
            model_ow[3][7] =  8'sd1;

            model_ob[0] =  8'sd1;  // +5
            model_ob[1] = -8'sd1;  // -5
            model_ob[2] =  8'sd2;  // +10
            model_ob[3] = -8'sd2;  // -10
        end
    endtask

    // Model 2: deliberately creates positive and negative hidden
    // pre-activations so the test can observe both ReLU paths.
    task automatic setup_relu_model;
        begin
            clear_model();

            // With inputs [10, 5, 7, ...] and hidden bias scale 3:
            // h0 = ReLU(-2*10 + 3*3) = ReLU(-11) = 0
            // h1 = ReLU(-1*5  + 10*3) = 25
            // h2 = ReLU(0     - 2*3)  = 0
            // h3 = ReLU(2*7   - 1*3)  = 11
            model_hw[0][0] = -8'sd2;
            model_hb[0]    =  8'sd3;
            model_hw[1][1] = -8'sd1;
            model_hb[1]    =  8'sd10;
            model_hb[2]    = -8'sd2;
            model_hw[3][2] =  8'sd2;
            model_hb[3]    = -8'sd1;

            // Expected scores: [h0+h1, h2+h3, -h1, h1+h3]
            model_ow[0][0] =  8'sd1;
            model_ow[0][1] =  8'sd1;
            model_ow[1][2] =  8'sd1;
            model_ow[1][3] =  8'sd1;
            model_ow[2][1] = -8'sd1;
            model_ow[3][1] =  8'sd1;
            model_ow[3][3] =  8'sd1;
        end
    endtask

    // Model 3: all MAC weights are zero, so each output must equal only its
    // signed int8 bias multiplied by OUTPUT_BIAS_SCALE.
    task automatic setup_output_bias_model;
        begin
            clear_model();
            model_ob[0] = -8'sd3;   // -15
            model_ob[1] =  8'sd0;   //   0
            model_ob[2] =  8'sd4;   //  20
            model_ob[3] =  8'sd127; // 635
        end
    endtask

    // Scan-chain driver and checker
    task automatic load_model;
        integer hidden_idx;
        integer input_idx;
        integer output_idx;
        integer bit_idx;
        integer mismatches;
        logic [TOTAL_WEIGHT_BITS-1:0] scan_image;
        begin
            scan_image = '0;

            // Pack bytes in the same documented order used by weights.hex:
            // hidden weights, hidden biases, output weights, output biases.
            for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                for (input_idx = 0; input_idx < N_IN; input_idx = input_idx + 1)
                    scan_image[(hidden_idx*N_IN + input_idx)*W_WIDTH +: W_WIDTH] =
                        model_hw[hidden_idx][input_idx];

            for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                scan_image[(N_HIDDEN*N_IN + hidden_idx)*W_WIDTH +: W_WIDTH] =
                    model_hb[hidden_idx];

            for (output_idx = 0; output_idx < N_OUT; output_idx = output_idx + 1)
                for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                    scan_image[((N_HIDDEN*N_IN + N_HIDDEN) +
                                output_idx*N_HIDDEN + hidden_idx)*W_WIDTH +: W_WIDTH] =
                        model_ow[output_idx][hidden_idx];

            for (output_idx = 0; output_idx < N_OUT; output_idx = output_idx + 1)
                scan_image[((N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_HIDDEN) +
                            output_idx)*W_WIDTH +: W_WIDTH] = model_ob[output_idx];

            // The DUT shifts new bits into the MSB and shifts existing bits
            // downward, so sending bit 0 first reconstructs scan_image exactly.
            scan_en = 1'b1;
            for (bit_idx = 0; bit_idx < TOTAL_WEIGHT_BITS; bit_idx = bit_idx + 1) begin
                scan_in = scan_image[bit_idx];
                #(SCAN_PERIOD/2) scan_clk = 1'b1;
                #(SCAN_PERIOD/2) scan_clk = 1'b0;
            end
            scan_en = 1'b0;
            scan_in = 1'b0;

            // White-box verification is appropriate here because this test is
            // specifically checking the scan layout and named parameter arrays.
            mismatches = 0;
            if (dut.scan_reg !== scan_image) begin
                $display("FAIL scan: complete scan register image mismatch");
                mismatches = mismatches + 1;
            end

            for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1) begin
                if (dut.hb[hidden_idx] !== model_hb[hidden_idx])
                    mismatches = mismatches + 1;
                for (input_idx = 0; input_idx < N_IN; input_idx = input_idx + 1)
                    if (dut.hw[hidden_idx][input_idx] !== model_hw[hidden_idx][input_idx])
                        mismatches = mismatches + 1;
            end

            for (output_idx = 0; output_idx < N_OUT; output_idx = output_idx + 1) begin
                if (dut.ob[output_idx] !== model_ob[output_idx])
                    mismatches = mismatches + 1;
                for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                    if (dut.ow[output_idx][hidden_idx] !== model_ow[output_idx][hidden_idx])
                        mismatches = mismatches + 1;
            end

            if (mismatches != 0)
                $fatal(1, "Scan load failed with %0d mismatches", mismatches);

            $display("PASS scan load: %0d parameter bits verified", TOTAL_WEIGHT_BITS);
        end
    endtask

    // Integer reference model
    task automatic calculate_expected;
        integer hidden_idx;
        integer input_idx;
        integer output_idx;
        reg signed [63:0] accumulator;
        begin
            for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1) begin
                accumulator = $signed(model_hb[hidden_idx]) * HIDDEN_BIAS_SCALE;
                for (input_idx = 0; input_idx < N_IN; input_idx = input_idx + 1)
                    accumulator = accumulator +
                        $signed({1'b0, sbp_features[input_idx]}) *
                        $signed(model_hw[hidden_idx][input_idx]);

                if (accumulator < 0)
                    expected_hidden[hidden_idx] = '0;
                else
                    expected_hidden[hidden_idx] = accumulator[ACC_WIDTH-1:0];
            end

            for (output_idx = 0; output_idx < N_OUT; output_idx = output_idx + 1) begin
                accumulator = $signed(model_ob[output_idx]) * OUTPUT_BIAS_SCALE;
                for (hidden_idx = 0; hidden_idx < N_HIDDEN; hidden_idx = hidden_idx + 1)
                    accumulator = accumulator +
                        $signed(expected_hidden[hidden_idx]) *
                        $signed(model_ow[output_idx][hidden_idx]);

                expected_scores[output_idx] = accumulator[SCORE_WIDTH-1:0];
            end
        end
    endtask

    // Reusable inference transaction
    task automatic check_case;
        input integer test_id;
        input logic [IN_WIDTH-1:0] feature0;
        input logic [IN_WIDTH-1:0] feature1;
        input logic [IN_WIDTH-1:0] feature2;
        input logic [IN_WIDTH-1:0] feature3;
        input logic [IN_WIDTH-1:0] feature4;
        input logic [IN_WIDTH-1:0] feature5;
        input logic [IN_WIDTH-1:0] feature6;
        input logic [IN_WIDTH-1:0] feature7;
        input logic inject_busy_start;

        integer cycle_count;
        integer idx;
        logic case_pass;
        logic signed [SCORE_WIDTH-1:0] held_scores [0:N_OUT-1];
        begin
            case_pass = 1'b1;

            // Drive all features and assert start on a falling edge so they are
            // stable before the DUT accepts the request at the next rising edge.
            @(negedge clk);
            sbp_features[0] = feature0;
            sbp_features[1] = feature1;
            sbp_features[2] = feature2;
            sbp_features[3] = feature3;
            sbp_features[4] = feature4;
            sbp_features[5] = feature5;
            sbp_features[6] = feature6;
            sbp_features[7] = feature7;
            start = 1'b1;

            calculate_expected();

            // This rising edge accepts start while the FSM is in S_IDLE.
            @(posedge clk);
            #1;

            @(negedge clk);
            start = 1'b0;

            cycle_count = 0;
            while ((done !== 1'b1) && (cycle_count < MAX_WAIT_CYCLES)) begin
                @(posedge clk);
                #1;
                cycle_count = cycle_count + 1;

                // Optionally assert start for one complete clock while the FSM
                // is busy in the hidden MAC phase. Driving on falling edges
                // avoids races and keeps every rising edge in the latency count.
                if (inject_busy_start && (cycle_count == 10)) begin
                    @(negedge clk);
                    start = 1'b1;
                end
                if (inject_busy_start && (cycle_count == 11)) begin
                    @(negedge clk);
                    start = 1'b0;
                end
            end

            if (done !== 1'b1) begin
                $display("FAIL test %0d: timeout waiting for done", test_id);
                case_pass = 1'b0;
            end

            if (cycle_count != EXPECTED_LATENCY) begin
                $display("FAIL test %0d: latency got=%0d expected=%0d cycles",
                         test_id, cycle_count, EXPECTED_LATENCY);
                case_pass = 1'b0;
            end

            // White-box hidden checks localize failures to the first layer and
            // directly prove both the passing and clamping paths of ReLU.
            for (idx = 0; idx < N_HIDDEN; idx = idx + 1) begin
                if (dut.hidden_act[idx] !== expected_hidden[idx]) begin
                    $display("FAIL test %0d: hidden[%0d] got=%0d expected=%0d",
                             test_id, idx, $signed(dut.hidden_act[idx]),
                             $signed(expected_hidden[idx]));
                    case_pass = 1'b0;
                end
            end

            for (idx = 0; idx < N_OUT; idx = idx + 1) begin
                if (class_scores[idx] !== expected_scores[idx]) begin
                    $display("FAIL test %0d: score[%0d] got=%0d expected=%0d",
                             test_id, idx, $signed(class_scores[idx]),
                             $signed(expected_scores[idx]));
                    case_pass = 1'b0;
                end
                held_scores[idx] = class_scores[idx];
            end

            // done must be a one-cycle pulse, and scores must remain registered.
            @(posedge clk);
            #1;
            if (done !== 1'b0) begin
                $display("FAIL test %0d: done did not return low", test_id);
                case_pass = 1'b0;
            end
            for (idx = 0; idx < N_OUT; idx = idx + 1) begin
                if (class_scores[idx] !== held_scores[idx]) begin
                    $display("FAIL test %0d: score[%0d] changed after done",
                             test_id, idx);
                    case_pass = 1'b0;
                end
            end

            if (case_pass) begin
                $display("PASS test %0d: latency=%0d scores=[%0d,%0d,%0d,%0d]",
                         test_id, cycle_count,
                         $signed(class_scores[0]), $signed(class_scores[1]),
                         $signed(class_scores[2]), $signed(class_scores[3]));
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Verify asynchronous reset of architectural outputs after operation.
    task automatic check_async_reset;
        input integer test_id;
        integer idx;
        logic case_pass;
        begin
            case_pass = 1'b1;
            @(negedge clk);
            rst_n = 1'b0;
            #1;

            if (done !== 1'b0) begin
                $display("FAIL test %0d: done not cleared by reset", test_id);
                case_pass = 1'b0;
            end
            for (idx = 0; idx < N_OUT; idx = idx + 1) begin
                if (class_scores[idx] !== '0) begin
                    $display("FAIL test %0d: score[%0d] not cleared by reset",
                             test_id, idx);
                    case_pass = 1'b0;
                end
            end

            if (case_pass) begin
                $display("PASS test %0d: asynchronous reset", test_id);
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end

            @(negedge clk);
            rst_n = 1'b1;
        end
    endtask

    // Main test sequence
    initial begin
        rst_n     = 1'b0;
        start     = 1'b0;
        scan_en   = 1'b0;
        scan_clk  = 1'b0;
        scan_in   = 1'b0;
        pass_count = 0;
        fail_count = 0;

        for (integer init_idx = 0; init_idx < N_IN; init_idx = init_idx + 1)
            sbp_features[init_idx] = '0;

        repeat (3) @(posedge clk);
        #1;
        if (done !== 1'b0 ||
            class_scores[0] !== '0 || class_scores[1] !== '0 ||
            class_scores[2] !== '0 || class_scores[3] !== '0)
            $fatal(1, "MLP reset values are incorrect");

        // Load the first model while normal operation remains in reset.
        setup_identity_model();
        load_model();

        @(negedge clk);
        rst_n = 1'b1;

        // Identity/indexing, signed output weight, and output-bias test.
        // Hand calculation: hidden=[10,20,30,40,50,60,70,80]
        // scores=[10+5, -2*20-5, 30+40+10, 80-10]
        //       =[15, -45, 80, 70]
        check_case(1, 10, 20, 30, 40, 50, 60, 70, 80, 1'b0);

        // Values above 127 verify that unsigned SBP inputs are zero-extended,
        // rather than accidentally interpreted as negative signed bytes.
        check_case(2, 200, 130, 255, 128, 1, 2, 3, 4, 1'b0);

        // Directed hidden-bias and ReLU test. Also pulses start while busy to
        // prove the current FSM ignores requests outside S_IDLE.
        setup_relu_model();
        load_model();
        check_case(3, 10, 5, 7, 0, 0, 0, 0, 0, 1'b1);

        // Output bias-only test includes negative, zero, and positive scores.
        setup_output_bias_model();
        load_model();
        check_case(4, 0, 0, 0, 0, 0, 0, 0, 0, 1'b0);

        check_async_reset(5);

        $display("----------------------------------------------");
        $display("MLP INFERENCE TEST: %0d passed, %0d failed",
                 pass_count, fail_count);
        $display("----------------------------------------------");

        if (fail_count != 0)
            $fatal(1, "MLP inference regression failed");

        $display("ALL MLP INFERENCE TESTS PASS");
        $finish;
    end

    // Independent watchdog: a deadlocked FSM cannot hang the regression.
    initial begin
        #(200000);
        $fatal(1, "MLP inference testbench timeout");
    end

endmodule
