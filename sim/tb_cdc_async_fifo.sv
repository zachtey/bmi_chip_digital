// tb_cdc_async_fifo.sv - unit bench for the dual-clock result FIFO.
//
// The writer sends a running sequence number, the reader expects the same
// sequence, so a dropped, duplicated or reordered entry is caught at once.
// Background monitors check occupancy (never above the depth, never below
// zero) and that rdata is stable while an entry is waiting.

`timescale 1ns/1ps

module tb_cdc_async_fifo;

    parameter integer WCLK_PERIOD_PS = 100000;
    parameter integer RCLK_PERIOD_PS = 7000;
    parameter integer RCLK_PHASE_PS  = 1300;
    parameter integer N_RANDOM       = 3000;

    localparam integer WIDTH = 32;
    localparam integer AW    = 2;
    localparam integer DEPTH = 1 << AW;

    reg  arst_n;
    reg  wclk, rclk;
    wire wrst_n, rrst_n;

    reg              wr_en;
    reg  [WIDTH-1:0] wdata;
    wire             wfull;
    reg              rd_en;
    wire [WIDTH-1:0] rdata;
    wire             rempty;

    reset_synchronizer u_wrst (.clk(wclk), .arst_n(arst_n), .rst_n_out(wrst_n));
    reset_synchronizer u_rrst (.clk(rclk), .arst_n(arst_n), .rst_n_out(rrst_n));

    cdc_async_fifo #(.WIDTH(WIDTH), .ADDR_WIDTH(AW)) dut (
        .wclk(wclk), .wrst_n(wrst_n), .wr_en(wr_en), .wdata(wdata), .wfull(wfull),
        .rclk(rclk), .rrst_n(rrst_n), .rd_en(rd_en), .rdata(rdata), .rempty(rempty)
    );

    initial wclk = 0;
    always #(WCLK_PERIOD_PS/2000.0) wclk = ~wclk;
    initial begin
        rclk = 0;
        #(RCLK_PHASE_PS/1000.0);
        forever #(RCLK_PERIOD_PS/2000.0) rclk = ~rclk;
    end

    function [WIDTH-1:0] pattern_of;
        input integer seq;
        pattern_of = {~seq[15:0], seq[15:0]};
    endfunction

    integer errors = 0;
    integer tests_passed = 0, tests_failed = 0;
    integer test_errors_at_start;

    // ---- traffic generators ----
    // Mode 0: off. Mode 1: random. Mode 2: every cycle.
    // wr_one / rd_one request exactly one write / read attempt.
    integer wr_mode, rd_mode;
    reg     wr_one, rd_one;
    integer wseq, rseq;            // next sequence number to write / expect
    integer full_seen, empty_pops, full_writes;

    always @(negedge wclk) begin
        wr_en <= (wr_mode == 2) || (wr_mode == 1 && $urandom_range(0, 1)) || wr_one;
        wr_one = 1'b0;
        wdata <= pattern_of(wseq);
    end

    always @(posedge wclk) begin
        if (wrst_n && wr_en) begin
            if (!wfull) wseq = wseq + 1;
            else        full_writes = full_writes + 1;
        end
        if (wrst_n && wfull) full_seen = full_seen + 1;
    end

    always @(negedge rclk) begin
        rd_en <= (rd_mode == 2) || (rd_mode == 1 && $urandom_range(0, 1)) || rd_one;
        rd_one = 1'b0;
    end

    always @(posedge rclk) begin
        if (rrst_n && rd_en) begin
            if (!rempty) begin
                if (rdata !== pattern_of(rseq)) begin
                    $display("FAIL: read 0x%08X expected 0x%08X (entry %0d)",
                             rdata, pattern_of(rseq), rseq);
                    errors = errors + 1;
                end
                rseq = rseq + 1;
            end else begin
                empty_pops = empty_pops + 1;
            end
        end
    end

    // ---- monitors ----
    // Occupancy from the bench's own counts of accepted writes and reads.
    always @(wseq or rseq) begin
        if (wseq - rseq > DEPTH) begin
            $display("FAIL: %0d entries in a depth-%0d FIFO", wseq - rseq, DEPTH);
            errors = errors + 1;
        end
        if (wseq - rseq < 0) begin
            $display("FAIL: read more entries than were written");
            errors = errors + 1;
        end
    end

    // rdata must hold while an entry is waiting and is not being popped.
    reg [WIDTH-1:0] prev_rdata;
    reg             prev_waiting;
    always @(posedge rclk) begin
        if (rrst_n && prev_waiting && !rempty && rdata !== prev_rdata) begin
            $display("FAIL: rdata changed while an entry was waiting");
            errors = errors + 1;
        end
        if (rrst_n && prev_waiting && rempty) begin
            $display("FAIL: rempty rose without a read");
            errors = errors + 1;
        end
        prev_rdata   <= rdata;
        prev_waiting <= rrst_n && !rempty && !rd_en;
    end

    task global_reset;
        begin
            wr_mode = 0; rd_mode = 0;
            arst_n = 0;
            #(3*WCLK_PERIOD_PS/1000.0 + 3*RCLK_PERIOD_PS/1000.0);
            wseq = 0; rseq = 0; full_seen = 0; empty_pops = 0; full_writes = 0;
            arst_n = 1;
            wait (wrst_n === 1'b1 && rrst_n === 1'b1);
            @(negedge wclk); @(negedge rclk);
        end
    endtask

    task settle;   // long enough for both pointers to cross
        begin
            repeat (4) @(posedge wclk);
            repeat (4) @(posedge rclk);
            repeat (2) @(posedge wclk);
            #1;
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

    integer i, held;

    initial begin
        wr_en = 0; rd_en = 0; wdata = 0;
        wr_mode = 0; rd_mode = 0; wr_one = 0; rd_one = 0;
        wseq = 0; rseq = 0; full_seen = 0; empty_pops = 0; full_writes = 0;
        prev_waiting = 0;

        // 1. Reset state
        begin_test("reset state");
        arst_n = 0;
        #(2*WCLK_PERIOD_PS/1000.0);
        if (rempty !== 1 || wfull !== 0) begin
            $display("FAIL: in reset rempty=%b wfull=%b", rempty, wfull);
            errors = errors + 1;
        end
        global_reset();
        settle();
        if (rempty !== 1 || wfull !== 0) begin
            $display("FAIL: after reset rempty=%b wfull=%b", rempty, wfull);
            errors = errors + 1;
        end
        end_test();

        // 2. Fill to full, writes beyond full are dropped, then drain
        begin_test("fill, overfill, drain");
        wr_mode = 2;
        repeat (DEPTH + 6) @(posedge wclk);
        wr_mode = 0;
        settle();
        if (wseq != DEPTH || wfull !== 1 || full_writes == 0) begin
            $display("FAIL: accepted %0d writes (expected %0d), wfull=%b, refused=%0d",
                     wseq, DEPTH, wfull, full_writes);
            errors = errors + 1;
        end
        if (rempty !== 0) begin
            $display("FAIL: rempty high with a full FIFO");
            errors = errors + 1;
        end
        rd_mode = 2;
        repeat (DEPTH + 6) @(posedge rclk);
        rd_mode = 0;
        settle();
        if (rseq != DEPTH || rempty !== 1 || wfull !== 0 || empty_pops == 0) begin
            $display("FAIL: drained %0d (expected %0d), rempty=%b wfull=%b empty_pops=%0d",
                     rseq, DEPTH, rempty, wfull, empty_pops);
            errors = errors + 1;
        end
        end_test();

        // 3. One entry at a time; the entry waits, unchanged, until popped
        begin_test("single entry held until popped");
        for (i = 0; i < 40; i = i + 1) begin
            held = wseq;
            @(posedge wclk); wr_one = 1'b1;
            wait (rempty === 1'b0);
            repeat (1 + i % 7) @(posedge rclk);
            if (rdata !== pattern_of(held) || rempty !== 0) begin
                $display("FAIL: waiting entry not presented");
                errors = errors + 1;
            end
            @(posedge rclk); rd_one = 1'b1;
            wait (rseq == held + 1);
            #1;
            if (rempty !== 1 || wseq != held + 1) begin
                $display("FAIL: single entry not popped exactly once");
                errors = errors + 1;
            end
        end
        end_test();

        // 4. Random traffic on both sides
        begin_test("random traffic");
        wr_mode = 1; rd_mode = 1;
        wait (rseq >= N_RANDOM);
        wr_mode = 0;
        settle();
        rd_mode = 2;
        settle(); settle();
        rd_mode = 0;
        if (rseq != wseq || rempty !== 1) begin
            $display("FAIL: wrote %0d read %0d rempty=%b", wseq, rseq, rempty);
            errors = errors + 1;
        end
        end_test();

        // 5. Writer always on, reader random: runs against full
        begin_test("writer saturating");
        full_seen = 0; held = rseq;
        wr_mode = 2; rd_mode = 1;
        wait (rseq >= held + N_RANDOM/4);
        wr_mode = 0; settle(); rd_mode = 2; settle(); settle(); rd_mode = 0;
        if (rseq != wseq || rempty !== 1) begin
            $display("FAIL: wrote %0d read %0d rempty=%b", wseq, rseq, rempty);
            errors = errors + 1;
        end
        end_test();

        // 6. Reader always on, writer random: runs against empty
        begin_test("reader saturating");
        held = rseq;
        wr_mode = 1; rd_mode = 2;
        wait (rseq >= held + N_RANDOM/4);
        wr_mode = 0; settle(); settle(); rd_mode = 0;
        if (rseq != wseq || rempty !== 1) begin
            $display("FAIL: wrote %0d read %0d rempty=%b", wseq, rseq, rempty);
            errors = errors + 1;
        end
        end_test();

        // 7. Global reset with entries queued and traffic running
        begin_test("reset with entries queued");
        for (i = 0; i < 6; i = i + 1) begin
            wr_mode = 2; rd_mode = (i % 2);
            repeat (3 + i) @(posedge wclk);
            #((i * RCLK_PERIOD_PS) / 3000.0);
            global_reset();
            settle();
            if (rempty !== 1 || wfull !== 0) begin
                $display("FAIL: after reset %0d rempty=%b wfull=%b", i, rempty, wfull);
                errors = errors + 1;
            end
            wr_mode = 1; rd_mode = 1;
            wait (rseq >= 20);
            wr_mode = 0; settle(); rd_mode = 2; settle(); settle(); rd_mode = 0;
            if (rseq != wseq) begin
                $display("FAIL: after reset %0d wrote %0d read %0d", i, wseq, rseq);
                errors = errors + 1;
            end
        end
        end_test();

        $display("------------------------------------------------");
        $display("CDC FIFO TEST (wclk %0d ps, rclk %0d ps, phase %0d ps): %0d passed, %0d failed",
                 WCLK_PERIOD_PS, RCLK_PERIOD_PS, RCLK_PHASE_PS, tests_passed, tests_failed);
        if (errors == 0) begin
            $display("ALL CDC FIFO TESTS PASS");
            $finish;
        end else
            $fatal(1, "%0d errors", errors);
    end

    initial begin
        #40_000_000;     // 40 ms; the longest configuration needs about 5 ms
        $fatal(1, "tb_cdc_async_fifo timeout");
    end

endmodule
