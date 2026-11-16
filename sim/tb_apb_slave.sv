// tb_apb_slave.sv - unit bench for the APB3 register block (pclk domain only).
//
// The CDC modules are not part of this bench. Small models stand in for
// them, with adjustable latency:
//   command mailbox   cmd_busy rises the cycle after cmd_send; RUNNING is
//                     updated one cycle state_before cmd_busy falls.
//   weight mailbox    wt_busy likewise; wt_done pulses state_before it falls.
//   result FIFO       a queue the tests push entries into.
// Every errored transfer is checked for "no side effects" by comparing all
// register state and the model counters state_before and after it.

`timescale 1ns/1ps

module tb_apb_slave;

    localparam integer N_OUT       = 4;
    localparam integer SCORE_WIDTH = 32;
    localparam integer WEIGHT_BITS = 864;
    localparam integer N_WORDS     = WEIGHT_BITS / 32;
    localparam integer ENTRY_W     = N_OUT*SCORE_WIDTH + 2;
    localparam integer CLK_PERIOD  = 10;

    localparam [11:0] A_COMMAND = 12'h000, A_STATUS = 12'h004, A_IRQ_EN = 12'h008,
                      A_ACK     = 12'h00C, A_RESULT = 12'h010, A_SCORE0 = 12'h014,
                      A_COMMIT  = 12'h024, A_ERROR  = 12'h028, A_WEIGHT0 = 12'h100;

    localparam [31:0] ST_RESULT_VALID = 32'h01, ST_CMD_BUSY = 32'h02,
                      ST_WT_BUSY = 32'h04, ST_WT_VALID = 32'h08, ST_RUNNING = 32'h10;

    reg         clk, rst_n;
    reg         psel, penable, pwrite;
    reg  [11:0] paddr;
    reg  [31:0] pwdata;
    wire [31:0] prdata;
    wire        pready, pslverr, irq;

    wire        cmd_send;
    wire [1:0]  cmd_code;
    reg         cmd_busy, cmd_running;
    wire        wt_send;
    wire [WEIGHT_BITS-1:0] weight_image;
    reg         wt_busy, wt_done;
    wire        fifo_empty, fifo_pop;
    wire [ENTRY_W-1:0] fifo_rdata;
    reg         overflow;

    apb_slave #(
        .N_OUT(N_OUT), .SCORE_WIDTH(SCORE_WIDTH), .WEIGHT_BITS(WEIGHT_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .psel(psel), .penable(penable), .pwrite(pwrite),
        .paddr(paddr), .pwdata(pwdata),
        .prdata(prdata), .pready(pready), .pslverr(pslverr), .irq(irq),
        .cmd_send(cmd_send), .cmd_code(cmd_code),
        .cmd_busy(cmd_busy), .cmd_running(cmd_running),
        .wt_send(wt_send), .weight_image(weight_image),
        .wt_busy(wt_busy), .wt_done(wt_done),
        .fifo_empty(fifo_empty), .fifo_rdata(fifo_rdata), .fifo_pop(fifo_pop),
        .overflow(overflow)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer errors = 0;
    integer tests_passed = 0, tests_failed = 0;
    integer test_errors_at_start;

    // ---- command mailbox model ----
    integer cmd_latency = 6;      // cycles cmd_busy stays high
    integer cmd_cnt;
    integer cmd_send_count;
    reg [1:0] cmd_code_seen;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_busy <= 0; cmd_running <= 0; cmd_cnt <= 0;
        end else if (cmd_send) begin
            cmd_send_count = cmd_send_count + 1;
            if (cmd_busy) begin
                $display("FAIL: cmd_send while the mailbox is busy");
                errors = errors + 1;
            end
            cmd_busy <= 1;
            cmd_cnt  <= cmd_latency;
        end else if (cmd_busy) begin
            if (cmd_cnt == cmd_latency)
                cmd_code_seen = cmd_code;
            else if (cmd_code !== cmd_code_seen) begin
                $display("FAIL: cmd_code changed while busy");
                errors = errors + 1;
            end
            if (cmd_cnt == 1) begin
                if (cmd_code == 2'd1) cmd_running <= 1;
                if (cmd_code == 2'd2) cmd_running <= 0;
            end
            if (cmd_cnt == 0) cmd_busy <= 0;
            else              cmd_cnt  <= cmd_cnt - 1;
        end
    end

    // ---- weight-commit mailbox model ----
    integer wt_latency = 8;
    integer wt_cnt;
    integer wt_send_count;
    reg [WEIGHT_BITS-1:0] committed_image;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_busy <= 0; wt_done <= 0; wt_cnt <= 0;
        end else begin
            wt_done <= 0;
            if (wt_send) begin
                wt_send_count = wt_send_count + 1;
                if (wt_busy) begin
                    $display("FAIL: wt_send while the mailbox is busy");
                    errors = errors + 1;
                end
                wt_busy <= 1;
                wt_cnt  <= wt_latency;
            end else if (wt_busy) begin
                if (wt_cnt == wt_latency)
                    committed_image = weight_image;
                else if (weight_image !== committed_image) begin
                    $display("FAIL: staged image changed while a commit was pending");
                    errors = errors + 1;
                end
                if (wt_cnt == 2) wt_done <= 1;
                if (wt_cnt == 0) wt_busy <= 0;
                else             wt_cnt  <= wt_cnt - 1;
            end
        end
    end

    // ---- result FIFO model ----
    reg [ENTRY_W-1:0] queue [0:63];
    integer q_head, q_tail, pop_count;
    reg     push_on_ack;                 // push next_entry in the access phase of a RESULT_ACK
    reg [ENTRY_W-1:0] next_entry;

    assign fifo_empty = (q_head == q_tail);
    assign fifo_rdata = queue[q_head % 64];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_head <= 0; q_tail <= 0;
        end else begin
            if (fifo_pop) begin
                if (fifo_empty) begin
                    $display("FAIL: fifo_pop while empty");
                    errors = errors + 1;
                end else begin
                    // Nonblocking, like a real FIFO pointer: rdata must not
                    // change until after this edge.
                    q_head <= q_head + 1;
                    pop_count = pop_count + 1;
                end
            end
            if (push_on_ack && psel && penable && pwrite && paddr == A_ACK) begin
                queue[q_tail % 64] = next_entry;
                q_tail <= q_tail + 1;
                push_on_ack = 0;
            end
        end
    end

    function [ENTRY_W-1:0] entry_of;
        input [1:0]  cls;
        input [31:0] s0, s1, s2, s3;
        entry_of = {s3, s2, s1, s0, cls};
    endfunction

    task push_result;
        input [ENTRY_W-1:0] entry;
        begin
            @(posedge clk); #1;
            queue[q_tail % 64] = entry;
            q_tail = q_tail + 1;
        end
    endtask

    // ---- background monitors ----
    integer monitor_errors = 0;
    integer irq_rise_count = 0;
    reg     prev_irq;
    reg     prev_valid;
    reg [ENTRY_W-1:0] prev_snap;

    always @(negedge clk) begin
        if (rst_n) begin
            if (pslverr && !(psel && penable)) begin
                $display("MONITOR FAIL: pslverr outside the access phase");
                monitor_errors = monitor_errors + 1;
            end
            if (!psel && prdata !== 32'd0) begin
                $display("MONITOR FAIL: prdata nonzero while not selected");
                monitor_errors = monitor_errors + 1;
            end
            if (pready !== 1'b1) begin
                $display("MONITOR FAIL: pready low");
                monitor_errors = monitor_errors + 1;
            end
            if (^prdata === 1'bx || pslverr === 1'bx || irq === 1'bx) begin
                $display("MONITOR FAIL: X on an APB output");
                monitor_errors = monitor_errors + 1;
            end
            // The snapshot may only change while RESULT_VALID is low.
            if (prev_valid && dut.result_valid_q && dut.snap_q !== prev_snap) begin
                $display("MONITOR FAIL: snapshot changed while RESULT_VALID");
                monitor_errors = monitor_errors + 1;
            end
            if (irq && !prev_irq) irq_rise_count = irq_rise_count + 1;
        end
        prev_irq   = irq;
        prev_valid = dut.result_valid_q;
        prev_snap  = dut.snap_q;
    end

    // ---- APB master ----
    reg back_to_back;   // when set, no idle cycle between transfers

    task apb_idle;
        begin psel = 0; penable = 0; pwrite = 0; paddr = 0; pwdata = 0; end
    endtask

    task apb_xfer;
        input         write;
        input  [11:0] addr;
        input  [31:0] wdata;
        input         expect_error;
        output [31:0] rdata;
        begin
            if (!psel) begin @(posedge clk); #1; end
            psel = 1; penable = 0; pwrite = write; paddr = addr; pwdata = wdata;
            @(posedge clk); #1;
            penable = 1;
            @(negedge clk);
            rdata = prdata;
            if (pslverr !== expect_error) begin
                $display("FAIL: %0s 0x%03X data 0x%08X pslverr=%b expected %b",
                         write ? "write" : "read", addr, wdata, pslverr, expect_error);
                errors = errors + 1;
            end
            @(posedge clk); #1;
            penable = 0;
            if (!back_to_back) apb_idle();
        end
    endtask

    reg [31:0] rd;

    task wr;  input [11:0] a; input [31:0] d; begin apb_xfer(1, a, d, 0, rd); end endtask

    task rd_expect;
        input [11:0] a; input [31:0] expected;
        begin
            apb_xfer(0, a, 0, 0, rd);
            if (rd !== expected) begin
                $display("FAIL: read 0x%03X = 0x%08X expected 0x%08X", a, rd, expected);
                errors = errors + 1;
            end
        end
    endtask

    // An errored transfer must return 0 and change nothing.
    task expect_error_no_effect;
        input write; input [11:0] a; input [31:0] d;
        reg [ENTRY_W+WEIGHT_BITS+8:0] state_before;
        integer cs, ws, pc;
        begin
            @(posedge clk); #1;
            state_before = {dut.irq_en_q, dut.cmd_code_q, dut.weights_valid_q,
                      dut.result_valid_q, dut.snap_q, weight_image};
            cs = cmd_send_count; ws = wt_send_count; pc = pop_count;
            psel = 1; penable = 0; pwrite = write; paddr = a; pwdata = d;
            @(posedge clk); #1;
            penable = 1;
            @(negedge clk);
            if (pslverr !== 1'b1) begin
                $display("FAIL: %0s 0x%03X data 0x%08X did not return pslverr",
                         write ? "write" : "read", a, d);
                errors = errors + 1;
            end
            if (prdata !== 32'd0) begin
                $display("FAIL: errored transfer at 0x%03X returned 0x%08X", a, prdata);
                errors = errors + 1;
            end
            @(posedge clk); #1;
            apb_idle();
            @(posedge clk); #1;
            if (state_before !== {dut.irq_en_q, dut.cmd_code_q, dut.weights_valid_q,
                            dut.result_valid_q, dut.snap_q, weight_image} ||
                cs != cmd_send_count || ws != wt_send_count || pc != pop_count) begin
                $display("FAIL: errored %0s at 0x%03X had a side effect",
                         write ? "write" : "read", a);
                errors = errors + 1;
            end
        end
    endtask

    task wait_idle;
        begin
            while (cmd_busy || wt_busy) @(posedge clk);
            #1;
        end
    endtask

    task check_result_regs;
        input [ENTRY_W-1:0] e;
        integer k;
        begin
            rd_expect(A_RESULT, {30'd0, e[1:0]});
            for (k = 0; k < 4; k = k + 1)
                rd_expect(A_SCORE0 + 4*k, e[2 + 32*k +: 32]);
        end
    endtask

    task do_reset;
        begin
            rst_n = 0; apb_idle(); overflow = 0;
            repeat (3) @(posedge clk);
            #1 rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    task begin_test;
        input [8*48-1:0] name;
        begin $display("TEST %0s", name); test_errors_at_start = errors; end
    endtask

    task end_test;
        begin
            if (errors == test_errors_at_start) tests_passed = tests_passed + 1;
            else                                 tests_failed = tests_failed + 1;
        end
    endtask

    reg [7:0]  weights [0:WEIGHT_BITS/8-1];
    reg [ENTRY_W-1:0] eA, eB, eC, eD;
    integer n, b, k, base;
    reg [31:0] word;

    initial begin
        $readmemh("weights.hex", weights);
        cmd_send_count = 0; wt_send_count = 0; pop_count = 0;
        push_on_ack = 0; back_to_back = 0; overflow = 0;
        prev_irq = 0; prev_valid = 0;
        eA = entry_of(2'd3, 32'h0001_85B3, 32'hFFFA_C19A, 32'h7FFF_FFFF, 32'h8000_0000);
        eB = entry_of(2'd0, 32'hDEAD_BEEF, 32'h0000_0001, 32'hFFFF_FFFF, 32'h1234_5678);
        eC = entry_of(2'd2, 32'h0000_0000, 32'hCAFE_F00D, 32'h0BAD_C0DE, 32'hFFFF_0000);
        eD = entry_of(2'd1, 32'h1111_1111, 32'h2222_2222, 32'h3333_3333, 32'h4444_4444);
        do_reset();

        // 1 ------------------------------------------------------------
        begin_test("reset values");
        rd_expect(A_COMMAND, 0); rd_expect(A_STATUS, 0); rd_expect(A_IRQ_EN, 0);
        rd_expect(A_ACK, 0);     rd_expect(A_RESULT, 0); rd_expect(A_COMMIT, 0);
        rd_expect(A_ERROR, 0);
        for (k = 0; k < 4; k = k + 1) rd_expect(A_SCORE0 + 4*k, 0);
        for (n = 0; n < N_WORDS; n = n + 1) rd_expect(A_WEIGHT0 + 4*n, 0);
        if (irq !== 0 || cmd_send !== 0 || wt_send !== 0 || weight_image !== '0) begin
            $display("FAIL: outputs not idle after reset"); errors = errors + 1;
        end
        end_test();

        // 2 ------------------------------------------------------------
        begin_test("staged weights: read-back and byte order");
        for (n = 0; n < N_WORDS; n = n + 1)
            wr(A_WEIGHT0 + 4*n, {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]});
        for (n = 0; n < N_WORDS; n = n + 1)
            rd_expect(A_WEIGHT0 + 4*n,
                      {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]});
        for (b = 0; b < WEIGHT_BITS/8; b = b + 1)
            if (weight_image[b*8 +: 8] !== weights[b]) begin
                $display("FAIL: image byte %0d", b); errors = errors + 1;
            end
        // A single word can be rewritten without disturbing its neighbours.
        wr(A_WEIGHT0 + 4*13, 32'hA5A5_5A5A);
        rd_expect(A_WEIGHT0 + 4*12, {weights[51], weights[50], weights[49], weights[48]});
        rd_expect(A_WEIGHT0 + 4*13, 32'hA5A5_5A5A);
        rd_expect(A_WEIGHT0 + 4*14, {weights[59], weights[58], weights[57], weights[56]});
        wr(A_WEIGHT0 + 4*13, {weights[55], weights[54], weights[53], weights[52]});
        end_test();

        // 3 ------------------------------------------------------------
        begin_test("START refused state_before a commit; STOP/RESTART legal");
        expect_error_no_effect(1, A_COMMAND, 32'd1);
        wr(A_COMMAND, 32'd2);
        rd_expect(A_STATUS, ST_CMD_BUSY);
        wait_idle();
        if (cmd_send_count != 1 || cmd_code_seen !== 2'd2) begin
            $display("FAIL: STOP not sent exactly once"); errors = errors + 1;
        end
        wr(A_COMMAND, 32'd3);
        wait_idle();
        if (cmd_send_count != 2 || cmd_code_seen !== 2'd3) begin
            $display("FAIL: RESTART not sent exactly once"); errors = errors + 1;
        end
        rd_expect(A_STATUS, 0);
        end_test();

        // 4 ------------------------------------------------------------
        begin_test("weight commit and WEIGHT_BUSY");
        wr(A_COMMIT, 32'd0);                       // bit 0 clear: nothing
        wr(A_COMMIT, 32'hFFFF_FFFE);
        if (wt_send_count != 0) begin
            $display("FAIL: commit sent with bit 0 clear"); errors = errors + 1;
        end
        wt_latency = 40;
        wr(A_COMMIT, 32'd1);
        rd_expect(A_STATUS, ST_WT_BUSY);
        expect_error_no_effect(1, A_WEIGHT0, 32'hFFFF_FFFF);          // staged image frozen
        expect_error_no_effect(1, A_WEIGHT0 + 4*26, 32'h1);
        expect_error_no_effect(1, A_COMMIT, 32'd1);                   // no second commit
        expect_error_no_effect(1, A_COMMAND, 32'd2);                  // no command during commit
        expect_error_no_effect(1, A_COMMAND, 32'd3);
        rd_expect(A_WEIGHT0, {weights[3], weights[2], weights[1], weights[0]});  // reads still work
        wr(A_COMMIT, 32'd0);                                          // harmless while busy
        wait_idle();
        rd_expect(A_STATUS, ST_WT_VALID);
        if (wt_send_count != 1) begin
            $display("FAIL: commit sent %0d times", wt_send_count); errors = errors + 1;
        end
        for (b = 0; b < WEIGHT_BITS/8; b = b + 1)
            if (committed_image[b*8 +: 8] !== weights[b]) begin
                $display("FAIL: committed byte %0d", b); errors = errors + 1;
            end
        wt_latency = 8;
        end_test();

        // 5 ------------------------------------------------------------
        begin_test("commands, COMMAND_BUSY and RUNNING");
        base = cmd_send_count;
        cmd_latency = 30;
        wr(A_COMMAND, 32'd1);
        rd_expect(A_STATUS, ST_WT_VALID | ST_CMD_BUSY);
        expect_error_no_effect(1, A_COMMAND, 32'd2);      // busy
        expect_error_no_effect(1, A_COMMAND, 32'd1);
        expect_error_no_effect(1, A_WEIGHT0, 32'h0);      // not confirmed stopped
        expect_error_no_effect(1, A_COMMIT, 32'd1);
        wait_idle();
        rd_expect(A_STATUS, ST_WT_VALID | ST_RUNNING);
        if (cmd_send_count != base + 1 || cmd_code_seen !== 2'd1) begin
            $display("FAIL: START not sent exactly once"); errors = errors + 1;
        end
        cmd_latency = 6;
        // Running: staged weights and commit are locked.
        expect_error_no_effect(1, A_WEIGHT0 + 4*5, 32'h0);
        expect_error_no_effect(1, A_COMMIT, 32'd1);
        // Invalid command codes.
        expect_error_no_effect(1, A_COMMAND, 32'd0);
        expect_error_no_effect(1, A_COMMAND, 32'd4);
        expect_error_no_effect(1, A_COMMAND, 32'h0000_0101);
        expect_error_no_effect(1, A_COMMAND, 32'hFFFF_FFFF);
        expect_error_no_effect(1, A_COMMAND, 32'h8000_0002);
        // START while running and RESTART are legal; RUNNING stays.
        wr(A_COMMAND, 32'd1); wait_idle();
        wr(A_COMMAND, 32'd3); wait_idle();
        if (cmd_send_count != base + 3 || cmd_code_seen !== 2'd3) begin
            $display("FAIL: RESTART while running"); errors = errors + 1;
        end
        rd_expect(A_STATUS, ST_WT_VALID | ST_RUNNING);
        // STOP: weights stay locked until the acknowledgment is back.
        cmd_latency = 30;
        wr(A_COMMAND, 32'd2);
        expect_error_no_effect(1, A_WEIGHT0, 32'h0);
        expect_error_no_effect(1, A_COMMIT, 32'd1);
        wait_idle();
        cmd_latency = 6;
        rd_expect(A_STATUS, ST_WT_VALID);
        wr(A_WEIGHT0, {weights[3], weights[2], weights[1], weights[0]});   // writable again
        // Recommit while stopped keeps WEIGHTS_VALID.
        wr(A_COMMIT, 32'd1);
        rd_expect(A_STATUS, ST_WT_VALID | ST_WT_BUSY);
        wait_idle();
        rd_expect(A_STATUS, ST_WT_VALID);
        end_test();

        // 6 ------------------------------------------------------------
        begin_test("result snapshot, RESULT_ACK, ordering");
        base = pop_count;
        rd_expect(A_RESULT, 0);
        expect_error_no_effect(1, A_ACK, 32'd1);          // nothing to release
        push_result(eA);
        repeat (3) @(posedge clk);
        rd_expect(A_STATUS, ST_WT_VALID | ST_RESULT_VALID);
        check_result_regs(eA);
        check_result_regs(eA);                            // reads do not pop
        push_result(eB);
        push_result(eC);
        repeat (20) @(posedge clk);
        check_result_regs(eA);                            // held while more are queued
        if (pop_count != base + 1) begin
            $display("FAIL: FIFO popped %0d times with the snapshot held",
                     pop_count - base);
            errors = errors + 1;
        end
        wr(A_ACK, 32'd0);                                 // bit 0 clear: no release
        wr(A_ACK, 32'hFFFF_FFFE);
        check_result_regs(eA);
        wr(A_ACK, 32'd1);
        check_result_regs(eB);
        wr(A_ACK, 32'h0000_0003);                         // only bit 0 matters
        check_result_regs(eC);
        // A new entry arriving in the very cycle of the acknowledgment.
        next_entry = eD; push_on_ack = 1;
        wr(A_ACK, 32'd1);
        check_result_regs(eD);
        wr(A_ACK, 32'd1);
        rd_expect(A_STATUS, ST_WT_VALID);
        rd_expect(A_RESULT, 0);
        for (k = 0; k < 4; k = k + 1) rd_expect(A_SCORE0 + 4*k, 0);
        expect_error_no_effect(1, A_ACK, 32'd1);          // double acknowledge
        if (pop_count != base + 4) begin
            $display("FAIL: %0d pops for 4 results", pop_count - base);
            errors = errors + 1;
        end
        end_test();

        // 7 ------------------------------------------------------------
        begin_test("interrupt masking and re-notification");
        irq_rise_count = 0;
        push_result(eA);
        repeat (4) @(posedge clk);
        rd_expect(A_STATUS, ST_WT_VALID | ST_RESULT_VALID);   // polling works
        if (irq !== 0) begin $display("FAIL: irq with IRQ_EN=0"); errors = errors + 1; end
        wr(A_IRQ_EN, 32'hFFFF_FFFE);                          // bit 0 clear
        rd_expect(A_IRQ_EN, 0);
        wr(A_IRQ_EN, 32'd1);
        rd_expect(A_IRQ_EN, 1);
        @(negedge clk);
        if (irq !== 1) begin $display("FAIL: irq low with result and IRQ_EN"); errors = errors + 1; end
        push_result(eB);
        repeat (4) @(posedge clk);
        wr(A_ACK, 32'd1);                                      // eB loads: irq must dip and rise
        repeat (4) @(posedge clk);
        @(negedge clk);
        if (irq !== 1 || irq_rise_count != 2) begin
            $display("FAIL: irq=%b after %0d rising edges (expected 2)", irq, irq_rise_count);
            errors = errors + 1;
        end
        check_result_regs(eB);
        wr(A_IRQ_EN, 32'd0);
        @(negedge clk);
        if (irq !== 0) begin $display("FAIL: irq not masked"); errors = errors + 1; end
        wr(A_ACK, 32'd1);
        wr(A_IRQ_EN, 32'd1);
        @(negedge clk);
        if (irq !== 0) begin $display("FAIL: irq with no result"); errors = errors + 1; end
        end_test();

        // 8 ------------------------------------------------------------
        begin_test("error responses have no side effects");
        push_result(eC);                                  // a held result, to detect damage
        repeat (3) @(posedge clk);
        // Unmapped
        expect_error_no_effect(0, 12'h02C, 0); expect_error_no_effect(1, 12'h02C, 32'hFFFF_FFFF);
        expect_error_no_effect(0, 12'h0FC, 0); expect_error_no_effect(1, 12'h0FC, 32'h1);
        expect_error_no_effect(0, 12'h16C, 0); expect_error_no_effect(1, 12'h16C, 32'h1);
        expect_error_no_effect(0, 12'hFFC, 0); expect_error_no_effect(1, 12'hFFC, 32'h1);
        expect_error_no_effect(0, 12'h400, 0); expect_error_no_effect(1, 12'h800, 32'h1);
        // Unaligned, on otherwise valid registers
        expect_error_no_effect(0, 12'h005, 0); expect_error_no_effect(0, 12'h012, 0);
        expect_error_no_effect(0, 12'h103, 0);
        expect_error_no_effect(1, 12'h009, 32'h0); expect_error_no_effect(1, 12'h102, 32'hFFFF_FFFF);
        expect_error_no_effect(1, 12'h001, 32'd2); expect_error_no_effect(1, 12'h00E, 32'd1);
        expect_error_no_effect(1, 12'h027, 32'd1);
        // Read-only registers
        expect_error_no_effect(1, A_STATUS, 32'hFFFF_FFFF);
        expect_error_no_effect(1, A_RESULT, 32'h0);
        for (k = 0; k < 4; k = k + 1) expect_error_no_effect(1, A_SCORE0 + 4*k, 32'h5);
        expect_error_no_effect(1, A_ERROR, 32'h1);
        // Write-only registers read 0 without error
        rd_expect(A_COMMAND, 0); rd_expect(A_ACK, 0); rd_expect(A_COMMIT, 0);
        check_result_regs(eC);
        rd_expect(A_IRQ_EN, 1);
        wr(A_ACK, 32'd1);
        end_test();

        // 9 ------------------------------------------------------------
        begin_test("writes need a completed access phase");
        base = cmd_send_count; n = wt_send_count;
        // Setup phase only, then the transfer is abandoned.
        @(posedge clk); #1;
        psel = 1; penable = 0; pwrite = 1; paddr = A_IRQ_EN; pwdata = 32'd0;
        repeat (3) @(posedge clk); #1;
        paddr = A_COMMAND; pwdata = 32'd1;
        repeat (2) @(posedge clk); #1;
        paddr = A_COMMIT; pwdata = 32'd1;
        repeat (2) @(posedge clk); #1;
        paddr = A_WEIGHT0; pwdata = 32'hFFFF_FFFF;
        repeat (2) @(posedge clk); #1;
        apb_idle();
        // penable without psel.
        penable = 1; pwrite = 1; paddr = A_IRQ_EN; pwdata = 32'd0;
        repeat (2) @(posedge clk); #1;
        paddr = 12'hFFC;
        @(negedge clk);
        if (pslverr !== 0) begin $display("FAIL: pslverr without psel"); errors = errors + 1; end
        @(posedge clk); #1;
        apb_idle();
        rd_expect(A_IRQ_EN, 1);
        rd_expect(A_WEIGHT0, {weights[3], weights[2], weights[1], weights[0]});
        if (cmd_send_count != base || wt_send_count != n) begin
            $display("FAIL: a setup-phase write had an effect"); errors = errors + 1;
        end
        end_test();

        // 10 -----------------------------------------------------------
        begin_test("back-to-back transfers");
        back_to_back = 1;
        for (n = 0; n < N_WORDS; n = n + 1)
            wr(A_WEIGHT0 + 4*n, 32'hC000_0000 + n);
        for (n = 0; n < N_WORDS; n = n + 1)
            rd_expect(A_WEIGHT0 + 4*n, 32'hC000_0000 + n);
        wr(A_IRQ_EN, 0); rd_expect(A_IRQ_EN, 0);
        wr(A_IRQ_EN, 1); rd_expect(A_IRQ_EN, 1);
        // Command directly followed by another: the second must see BUSY.
        base = cmd_send_count;
        wr(A_COMMAND, 32'd1);
        apb_xfer(1, A_COMMAND, 32'd2, 1, rd);
        apb_xfer(1, A_WEIGHT0, 32'h0, 1, rd);
        rd_expect(A_STATUS, ST_WT_VALID | ST_CMD_BUSY);
        back_to_back = 0; apb_idle();
        wait_idle();
        if (cmd_send_count != base + 1) begin
            $display("FAIL: back-to-back command sent twice"); errors = errors + 1;
        end
        // Commit directly followed by a weight write and a command.
        wr(A_COMMAND, 32'd2); wait_idle();
        back_to_back = 1;
        base = wt_send_count;
        wr(A_COMMIT, 32'd1);
        apb_xfer(1, A_WEIGHT0, 32'h0, 1, rd);
        apb_xfer(1, A_COMMAND, 32'd1, 1, rd);
        apb_xfer(1, A_COMMIT, 32'd1, 1, rd);
        back_to_back = 0; apb_idle();
        wait_idle();
        if (wt_send_count != base + 1 || committed_image[31:0] !== 32'hC000_0000) begin
            $display("FAIL: back-to-back commit"); errors = errors + 1;
        end
        end_test();

        // 11 -----------------------------------------------------------
        begin_test("ERROR_STATUS overflow flag");
        rd_expect(A_ERROR, 0);
        @(posedge clk); #1 overflow = 1;
        rd_expect(A_ERROR, 1);
        expect_error_no_effect(1, A_ERROR, 32'h1);
        rd_expect(A_ERROR, 1);
        rd_expect(A_STATUS, ST_WT_VALID);      // not mirrored into STATUS
        end_test();

        // 12 -----------------------------------------------------------
        begin_test("reset in the middle of a transfer");
        push_result(eA);
        repeat (3) @(posedge clk);
        @(posedge clk); #1;
        psel = 1; penable = 0; pwrite = 1; paddr = A_WEIGHT0 + 4*3; pwdata = 32'hFFFF_FFFF;
        @(posedge clk); #1;
        penable = 1;
        #2 rst_n = 0;
        #3;
        if (prdata !== 0 || irq !== 0 || weight_image !== '0) begin
            $display("FAIL: outputs not cleared by reset"); errors = errors + 1;
        end
        apb_idle(); overflow = 0;
        repeat (2) @(posedge clk);
        #1 rst_n = 1;
        rd_expect(A_STATUS, 0); rd_expect(A_IRQ_EN, 0); rd_expect(A_RESULT, 0);
        rd_expect(A_ERROR, 0);
        for (n = 0; n < N_WORDS; n = n + 1) rd_expect(A_WEIGHT0 + 4*n, 0);
        expect_error_no_effect(1, A_COMMAND, 32'd1);      // WEIGHTS_VALID was cleared
        wr(A_WEIGHT0, 32'h1234_5678);
        rd_expect(A_WEIGHT0, 32'h1234_5678);
        end_test();

        repeat (4) @(posedge clk);
        $display("------------------------------------------------");
        $display("APB SLAVE TEST: %0d passed, %0d failed, %0d monitor violations",
                 tests_passed, tests_failed, monitor_errors);
        if (errors == 0 && monitor_errors == 0) begin
            $display("ALL APB SLAVE TESTS PASS");
            $finish;
        end else
            $fatal(1, "%0d errors, %0d monitor violations", errors, monitor_errors);
    end

    initial begin
        #5_000_000;
        $fatal(1, "tb_apb_slave timeout");
    end

endmodule
