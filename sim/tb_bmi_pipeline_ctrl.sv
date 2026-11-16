// tb_bmi_pipeline_ctrl.sv - unit bench for the pipeline controller, with the
// real argmax attached so that single-cycle coincidences between a clear and
// a result can be produced exactly. In the full chip those coincidences
// depend on synchronizer timing and cannot be forced. Single clock.

`timescale 1ns/1ps

module tb_bmi_pipeline_ctrl;

    localparam [1:0] CMD_START = 2'd1, CMD_STOP = 2'd2, CMD_RESTART = 2'd3;

    reg clk, rst_n;
    reg         cmd_valid;
    reg  [1:0]  cmd_code;
    wire        cmd_done, cmd_running;
    reg         wt_valid;
    wire        wt_done, weights_valid;
    wire        frontend_enable, pipe_clear;
    reg         features_done;
    reg         scores_valid;                 // stands in for mlp_done
    reg  signed [31:0] class_scores [0:3];
    wire [1:0]  predicted_class;
    wire        decision_valid;
    wire        fifo_push;
    wire [129:0] fifo_wdata;
    reg         fifo_full;
    wire        overflow;

    bmi_pipeline_ctrl dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_code(cmd_code),
        .cmd_done(cmd_done), .cmd_running(cmd_running),
        .wt_valid(wt_valid), .wt_done(wt_done), .weights_valid(weights_valid),
        .frontend_enable(frontend_enable), .pipe_clear(pipe_clear),
        .features_done(features_done), .decision_valid(decision_valid),
        .predicted_class(predicted_class), .class_scores(class_scores),
        .fifo_push(fifo_push), .fifo_wdata(fifo_wdata), .fifo_full(fifo_full),
        .overflow(overflow)
    );

    argmax u_argmax (
        .clk(clk), .rst_n(rst_n), .clear(pipe_clear),
        .class_scores(class_scores), .scores_valid(scores_valid),
        .predicted_class(predicted_class), .decision_valid(decision_valid)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer errors = 0, tests_passed = 0, tests_failed = 0, errors_at_start;
    integer push_count, clear_count, done_count, decision_count;
    reg [129:0] last_push;

    always @(posedge clk) begin
        if (rst_n) begin
            if (fifo_push) begin push_count = push_count + 1; last_push = fifo_wdata; end
            if (pipe_clear)     clear_count    = clear_count + 1;
            if (cmd_done)       done_count     = done_count + 1;
            if (decision_valid) decision_count = decision_count + 1;
            if (cmd_done && !cmd_valid) begin
                $display("FAIL: cmd_done without cmd_valid"); errors = errors + 1;
            end
        end
    end

    task check; input ok; input [8*64-1:0] msg;
        begin if (ok !== 1'b1) begin $display("FAIL: %0s", msg); errors = errors + 1; end end
    endtask
    task begin_test; input [8*48-1:0] name;
        begin $display("TEST %0s", name); errors_at_start = errors; end
    endtask
    task end_test;
        begin if (errors == errors_at_start) tests_passed++; else tests_failed++; end
    endtask
    task tick; begin @(posedge clk); #1; end endtask

    // Present a command like the mailbox does: held until the cycle after
    // cmd_done. Returns the number of cycles it waited.
    task command; input [1:0] code; output integer waited;
        begin
            cmd_code = code; cmd_valid = 1; waited = 0;
            #1;     // let the combinational cmd_done settle
            while (!cmd_done) begin tick(); waited = waited + 1; end
            tick();
            cmd_valid = 0;
        end
    endtask

    // One window: features_done, then scores_valid `gap` cycles later.
    task start_inference;
        begin features_done = 1; tick(); features_done = 0; end
    endtask
    task finish_inference;
        begin scores_valid = 1; tick(); scores_valid = 0; end
    endtask

    integer w;

    initial begin
        rst_n = 0; cmd_valid = 0; cmd_code = 0; wt_valid = 0;
        features_done = 0; scores_valid = 0; fifo_full = 0;
        class_scores[0] = 32'sd10; class_scores[1] = -32'sd5;
        class_scores[2] = 32'sd700; class_scores[3] = 32'sd3;
        push_count = 0; clear_count = 0; done_count = 0; decision_count = 0;
        repeat (3) tick();
        rst_n = 1; tick();

        begin_test("START has no effect without weights");
        check(!frontend_enable && !weights_valid && !overflow, "idle after reset");
        command(CMD_START, w);
        check(!frontend_enable && done_count == 1, "START acknowledged but not running");
        end_test();

        begin_test("weight commit, then START");
        wt_valid = 1; #1; check(wt_done === 1'b1, "wt_done with wt_valid");
        tick(); wt_valid = 0;
        check(weights_valid, "weights_valid after commit");
        cmd_code = CMD_START; cmd_valid = 1; #1;
        check(cmd_done && cmd_running, "START returns RUNNING=1 with the ack");
        tick(); tick(); cmd_valid = 0;
        check(frontend_enable, "running after START");
        end_test();

        begin_test("result pushed as one entry");
        start_inference(); repeat (3) tick(); finish_inference();
        #1; check(decision_valid && fifo_push, "push in the decision cycle");
        tick();
        check(push_count == 1 &&
              last_push === {32'sd3, 32'sd700, -32'sd5, 32'sd10, 2'd2}, "entry = scores and class");
        end_test();

        begin_test("FIFO full: dropped, overflow sticky");
        fifo_full = 1;
        start_inference(); finish_inference(); tick();
        check(push_count == 1 && overflow, "no push, overflow set");
        fifo_full = 0;
        start_inference(); finish_inference(); tick();
        check(push_count == 2 && overflow, "next result pushed, overflow still set");
        end_test();

        begin_test("RESTART in the decision cycle discards the result");
        start_inference(); finish_inference();      // decision_valid is high now
        cmd_code = CMD_RESTART; cmd_valid = 1; #1;
        check(decision_valid && pipe_clear && !fifo_push, "clear wins over push");
        tick(); tick(); cmd_valid = 0; tick();
        check(push_count == 2 && frontend_enable, "nothing pushed, still running");
        end_test();

        begin_test("RESTART in the scores_valid cycle: no decision follows");
        w = decision_count;
        start_inference();
        scores_valid = 1; cmd_code = CMD_RESTART; cmd_valid = 1; #1;
        check(pipe_clear, "clear in the scores_valid cycle");
        tick(); scores_valid = 0;
        check(!decision_valid && predicted_class === 2'd0, "argmax cleared");
        tick(); cmd_valid = 0; repeat (3) tick();
        check(decision_count == w && push_count == 2, "no decision, no push");
        end_test();

        begin_test("STOP waits for the inference in flight");
        clear_count = 0; done_count = 0;
        start_inference();
        cmd_code = CMD_STOP; cmd_valid = 1;
        repeat (20) begin tick(); check(!cmd_done && !pipe_clear, "no ack while in flight"); end
        check(!frontend_enable, "RUNNING cleared at once");
        finish_inference();                          // decision this cycle
        #1; check(fifo_push && !cmd_done, "in-flight result pushed before the ack");
        tick(); #1;
        check(cmd_done && pipe_clear && !cmd_running, "ack with clear, RUNNING=0");
        tick(); cmd_valid = 0; tick();
        check(push_count == 3 && clear_count == 1 && done_count == 1, "one push, one clear, one ack");
        end_test();

        begin_test("STOP while idle, START again, features_done at STOP");
        command(CMD_STOP, w);
        check(w <= 2 && !frontend_enable, "STOP while stopped acknowledged at once");
        command(CMD_START, w);
        check(frontend_enable, "running again");
        // Window completes in the very cycle STOP is accepted.
        features_done = 1; cmd_code = CMD_STOP; cmd_valid = 1; tick(); features_done = 0;
        repeat (5) begin check(!cmd_done, "STOP waits for a window that just completed"); tick(); end
        finish_inference(); tick(); #1;
        check(cmd_done, "acknowledged after its decision");
        tick(); cmd_valid = 0; tick();
        check(push_count == 4, "that result was pushed");
        end_test();

        $display("------------------------------------------------");
        $display("PIPELINE CTRL TEST: %0d passed, %0d failed", tests_passed, tests_failed);
        if (errors == 0) begin $display("ALL PIPELINE CTRL TESTS PASS"); $finish; end
        else $fatal(1, "%0d errors", errors);
    end

    initial begin #1_000_000; $fatal(1, "tb_bmi_pipeline_ctrl timeout"); end

endmodule
