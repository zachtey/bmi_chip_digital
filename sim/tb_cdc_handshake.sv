// tb_cdc_handshake.sv - unit bench for the request/acknowledge mailbox.
//
// Two unrelated clocks (override the periods and phase to regress other
// ratios). Each side gets its reset from a real reset_synchronizer, as in
// bmi_chip_top. A scoreboard checks that every request is executed exactly
// once with the right data and that the right response comes back.

`timescale 1ns/1ps

module tb_cdc_handshake;

    parameter integer SRC_PERIOD_PS = 7000;
    parameter integer DST_PERIOD_PS = 100000;
    parameter integer DST_PHASE_PS  = 1300;
    parameter integer N_RANDOM      = 200;

    localparam integer REQ_W = 16;
    localparam integer RSP_W = 8;

    reg  arst_n;
    reg  src_clk, dst_clk;
    reg  src_run, dst_run;          // clock gates for the stopped-clock tests
    wire src_rst_n, dst_rst_n;

    reg              src_send;
    reg  [REQ_W-1:0] src_data;
    wire             src_busy, src_done;
    wire [RSP_W-1:0] src_rsp;
    wire             dst_valid;
    wire [REQ_W-1:0] dst_data;
    reg              dst_done;
    reg  [RSP_W-1:0] dst_rsp;

    reset_synchronizer u_src_rst (.clk(src_clk), .arst_n(arst_n), .rst_n_out(src_rst_n));
    reset_synchronizer u_dst_rst (.clk(dst_clk), .arst_n(arst_n), .rst_n_out(dst_rst_n));

    cdc_handshake #(.REQ_WIDTH(REQ_W), .RSP_WIDTH(RSP_W)) dut (
        .src_clk(src_clk), .src_rst_n(src_rst_n),
        .src_send(src_send), .src_data(src_data),
        .src_busy(src_busy), .src_done(src_done), .src_rsp(src_rsp),
        .dst_clk(dst_clk), .dst_rst_n(dst_rst_n),
        .dst_valid(dst_valid), .dst_data(dst_data),
        .dst_done(dst_done), .dst_rsp(dst_rsp)
    );

    // Clocks stop low when gated off.
    initial src_clk = 0;
    always begin
        #(SRC_PERIOD_PS/2000.0);
        if (src_run || src_clk) src_clk = ~src_clk;
    end
    initial begin
        dst_clk = 0;
        #(DST_PHASE_PS/1000.0);
        forever begin
            #(DST_PERIOD_PS/2000.0);
            if (dst_run || dst_clk) dst_clk = ~dst_clk;
        end
    end

    function [RSP_W-1:0] response_of;
        input [REQ_W-1:0] d;
        response_of = d[7:0] ^ d[15:8] ^ 8'hA5;
    endfunction

    integer errors = 0;
    integer tests_passed = 0, tests_failed = 0;
    integer test_errors_at_start;

    // ---- scoreboard ----
    integer sent_count;        // requests the source launched
    integer exec_count;        // requests the destination executed
    integer done_count;        // src_done pulses
    integer valid_rise_count;  // dst_valid rising edges
    reg [REQ_W-1:0] expected_data;
    reg             request_open;      // launched, not yet executed
    integer max_done_delay;            // dst cycles a request waits for dst_done

    // ---- destination model: execute after a random delay ----
    integer dst_delay;
    reg     prev_dst_valid;

    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            dst_done       <= 1'b0;
            dst_rsp        <= '0;
            dst_delay      <= 0;
            prev_dst_valid <= 1'b0;
        end else begin
            prev_dst_valid <= dst_valid;
            if (dst_valid && !prev_dst_valid) begin
                valid_rise_count = valid_rise_count + 1;
                dst_delay <= $urandom_range(0, max_done_delay);
            end

            if (dst_valid && dst_done) begin
                // This edge is the execution.
                exec_count = exec_count + 1;
                if (!request_open) begin
                    $display("FAIL: request executed twice or without a send");
                    errors = errors + 1;
                end
                if (dst_data !== expected_data) begin
                    $display("FAIL: dst_data 0x%04X expected 0x%04X", dst_data, expected_data);
                    errors = errors + 1;
                end
                request_open = 1'b0;
                dst_done <= 1'b0;
            end else if (dst_valid && prev_dst_valid && !dst_done) begin
                if (dst_delay == 0) begin
                    dst_done <= 1'b1;
                    dst_rsp  <= response_of(dst_data);
                end else begin
                    dst_delay <= dst_delay - 1;
                end
            end
        end
    end

    // dst_data must not change while dst_valid is high.
    reg [REQ_W-1:0] held_dst_data;
    always @(posedge dst_clk) begin
        #1;
        if (dst_rst_n && dst_valid && prev_dst_valid && dst_data !== held_dst_data) begin
            $display("FAIL: dst_data changed while dst_valid");
            errors = errors + 1;
        end
        held_dst_data = dst_data;
    end

    // ---- source monitor ----
    reg prev_src_done;
    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            prev_src_done <= 1'b0;
        end else begin
            prev_src_done <= src_done;
            if (src_done) begin
                done_count = done_count + 1;
                if (prev_src_done) begin
                    $display("FAIL: src_done wider than one cycle");
                    errors = errors + 1;
                end
                if (!src_busy) begin
                    $display("FAIL: src_done while src_busy is low");
                    errors = errors + 1;
                end
                if (src_rsp !== response_of(expected_data)) begin
                    $display("FAIL: src_rsp 0x%02X expected 0x%02X",
                             src_rsp, response_of(expected_data));
                    errors = errors + 1;
                end
                if (request_open) begin
                    $display("FAIL: src_done before the destination executed");
                    errors = errors + 1;
                end
            end
        end
    end

    // ---- source driver ----
    // Launch one request and wait for the complete four-phase cycle. While
    // busy, extra src_send pulses are driven (with the data held) and must be
    // ignored.
    task send_request;
        input [REQ_W-1:0] data;
        input integer     extra_sends;
        integer guard;
        begin
            if (src_busy !== 1'b0) begin
                $display("FAIL: src_busy high before send");
                errors = errors + 1;
            end
            src_data      = data;
            expected_data = data;
            request_open  = 1'b1;
            sent_count    = sent_count + 1;
            src_send      = 1'b1;
            @(posedge src_clk); #1;
            src_send      = 1'b0;
            if (src_busy !== 1'b1) begin
                $display("FAIL: src_busy not high the cycle after src_send");
                errors = errors + 1;
            end
            guard = 0;
            while (src_busy === 1'b1) begin
                src_send = (extra_sends != 0) && ($urandom_range(0, 2) == 0);
                @(posedge src_clk); #1;
                guard = guard + 1;
                if (guard > 2_000_000) begin
                    $display("FAIL: handshake never completed");
                    errors = errors + 1;
                    $fatal(1, "handshake timeout");
                end
            end
            src_send = 1'b0;
            if (request_open) begin
                $display("FAIL: src_busy fell before the request was executed");
                errors = errors + 1;
            end
        end
    endtask

    task check_counts;
        input integer expected;
        begin
            if (sent_count != expected || exec_count != expected ||
                done_count != expected || valid_rise_count != expected) begin
                $display("FAIL: counts sent=%0d executed=%0d done=%0d valid_rises=%0d expected=%0d",
                         sent_count, exec_count, done_count, valid_rise_count, expected);
                errors = errors + 1;
            end
        end
    endtask

    task clear_counts;
        begin
            sent_count = 0; exec_count = 0; done_count = 0; valid_rise_count = 0;
            request_open = 1'b0;
        end
    endtask

    task global_reset;
        begin
            arst_n = 1'b0;
            src_send = 1'b0;
            #(3*SRC_PERIOD_PS/1000.0 + 3*DST_PERIOD_PS/1000.0);
            arst_n = 1'b1;
            wait (src_rst_n === 1'b1);
            @(posedge src_clk); #1;
        end
    endtask

    task begin_test;
        input [8*40-1:0] name;
        begin
            $display("TEST %0s", name);
            test_errors_at_start = errors;
        end
    endtask

    task end_test;
        begin
            if (errors == test_errors_at_start) tests_passed = tests_passed + 1;
            else                                 tests_failed = tests_failed + 1;
        end
    endtask

    integer i;

    initial begin
        src_run = 1; dst_run = 1;
        src_send = 0; src_data = 0;
        max_done_delay = 5;
        clear_counts();

        // 1. Reset state
        begin_test("reset state");
        arst_n = 0;
        #(2*SRC_PERIOD_PS/1000.0);
        if (src_busy !== 0 || src_done !== 0 || dst_valid !== 0 || src_rsp !== 0) begin
            $display("FAIL: outputs not idle in reset");
            errors = errors + 1;
        end
        global_reset();
        repeat (4) @(posedge dst_clk);
        if (src_busy !== 0 || dst_valid !== 0) begin
            $display("FAIL: activity after reset release");
            errors = errors + 1;
        end
        check_counts(0);
        end_test();

        // 2. Random transfers with idle gaps
        begin_test("random transfers");
        for (i = 0; i < N_RANDOM; i = i + 1) begin
            repeat ($urandom_range(0, 3)) @(posedge src_clk);
            #1;
            send_request($urandom, 0);
        end
        check_counts(N_RANDOM);
        end_test();

        // 3. Back to back: send in the first cycle busy is low
        begin_test("back-to-back");
        clear_counts();
        max_done_delay = 0;
        for (i = 0; i < 50; i = i + 1)
            send_request($urandom, 0);
        check_counts(50);
        max_done_delay = 5;
        end_test();

        // 4. Sends while busy are ignored
        begin_test("send while busy ignored");
        clear_counts();
        for (i = 0; i < 50; i = i + 1)
            send_request($urandom, 1);
        check_counts(50);
        end_test();

        // 5. Destination clock stopped with a request pending
        begin_test("destination clock stopped");
        clear_counts();
        for (i = 0; i < 10; i = i + 1) begin
            dst_run = 0;
            src_data = $urandom; expected_data = src_data;
            request_open = 1; sent_count = sent_count + 1;
            src_send = 1; @(posedge src_clk); #1; src_send = 0;
            repeat (20 + i) @(posedge src_clk);
            if (src_busy !== 1 || exec_count != i) begin
                $display("FAIL: request did not wait for the stopped clock");
                errors = errors + 1;
            end
            // Stop it again part-way through on some iterations.
            dst_run = 1;
            if (i % 2) begin
                repeat (1 + i/2) @(posedge dst_clk);
                dst_run = 0;
                repeat (15) @(posedge src_clk);
                dst_run = 1;
            end
            wait (src_busy === 1'b0);
            @(posedge src_clk); #1;
        end
        check_counts(10);
        end_test();

        // 6. Source clock stopped while the acknowledgment is on its way
        begin_test("source clock stopped");
        clear_counts();
        for (i = 0; i < 10; i = i + 1) begin
            src_data = $urandom; expected_data = src_data;
            request_open = 1; sent_count = sent_count + 1;
            src_send = 1; @(posedge src_clk); #1; src_send = 0;
            repeat (i % 3) @(posedge src_clk);
            src_run = 0;
            repeat (12) @(posedge dst_clk);
            if (exec_count != i + 1) begin
                $display("FAIL: destination did not execute while source was stopped");
                errors = errors + 1;
            end
            src_run = 1;
            @(posedge src_clk); #1;
            wait (src_busy === 1'b0);
            @(posedge src_clk); #1;
        end
        check_counts(10);
        end_test();

        // 7. Global reset in the middle of a transfer, at several points
        begin_test("reset mid-transfer");
        for (i = 0; i < 12; i = i + 1) begin
            clear_counts();
            src_data = $urandom; expected_data = src_data;
            request_open = 1;
            src_send = 1; @(posedge src_clk); #1; src_send = 0;
            #((i * (DST_PERIOD_PS + 3*SRC_PERIOD_PS)) / 1000.0 / 2.0);
            global_reset();
            clear_counts();
            repeat (6) @(posedge dst_clk);
            repeat (6) @(posedge src_clk);
            #1;
            if (src_busy !== 0 || dst_valid !== 0 || exec_count != 0 ||
                done_count != 0) begin
                $display("FAIL: leftover activity after reset (iteration %0d)", i);
                errors = errors + 1;
            end
            send_request($urandom, 0);
            check_counts(1);
        end
        end_test();

        // 8. Request raised while the destination is still in reset
        begin_test("destination released late");
        clear_counts();
        arst_n = 0;
        #(3*DST_PERIOD_PS/1000.0);
        dst_run = 0;
        #(2*DST_PERIOD_PS/1000.0);
        arst_n = 1;
        wait (src_rst_n === 1'b1);
        @(posedge src_clk); #1;
        src_data = 16'hBEEF; expected_data = src_data;
        request_open = 1; sent_count = 1;
        src_send = 1; @(posedge src_clk); #1; src_send = 0;
        repeat (30) @(posedge src_clk);
        if (dst_rst_n !== 0 || src_busy !== 1 || exec_count != 0) begin
            $display("FAIL: request should wait for the destination reset");
            errors = errors + 1;
        end
        dst_run = 1;
        wait (src_busy === 1'b0);
        @(posedge src_clk); #1;
        check_counts(1);
        end_test();

        $display("------------------------------------------------");
        $display("CDC HANDSHAKE TEST (src %0d ps, dst %0d ps, phase %0d ps): %0d passed, %0d failed",
                 SRC_PERIOD_PS, DST_PERIOD_PS, DST_PHASE_PS, tests_passed, tests_failed);
        if (errors == 0) begin
            $display("ALL CDC HANDSHAKE TESTS PASS");
            $finish;
        end else
            $fatal(1, "%0d errors", errors);
    end

    initial begin
        #20_000_000;     // 20 ms; the longest configuration needs about 2 ms
        $fatal(1, "tb_cdc_handshake timeout");
    end

endmodule
