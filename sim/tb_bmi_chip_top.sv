// tb_bmi_chip_top.sv - RTL simulation of the whole chip.
//
// Two unrelated clocks: bmi_clk for the pipeline and pclk for APB. The APB
// master and host model run on pclk; the ADC model, the pipeline monitor and
// the scoreboard run on bmi_clk. PCLK_PERIOD_PS and PCLK_PHASE_PS set the APB
// clock; override them to regress other ratios and phases.
//
// Part 1 - golden vectors. The host loads weights.hex over APB, commits it,
//   starts the pipeline and reads 40 results. SBP features, internal scores,
//   and the class and full 32-bit scores read over APB are compared with the
//   Python golden files. Even vectors use the interrupt, odd vectors poll.
//   On some vectors the host sleeps for more than two windows, so results
//   queue in the FIFO while the snapshot is held.
// Part 2 - directed scenarios (see the tasks named scenario_*): back-to-back
//   commands, STOP during an inference, weight recommit, RESTART, FIFO full
//   and overflow, acknowledgment races, and a global reset while running.
//
// Scoreboard: every entry the pipeline pushes into the result FIFO is
// recorded on the bmi_clk side. Everything the host reads must match those
// entries in order, with none skipped or repeated.

`timescale 1ns/1ps

module tb_bmi_chip_top;

    parameter N_CH              = 8;
    parameter N_SAMPLES         = 250;
    parameter ADC_WIDTH         = 8;
    parameter N_VECTORS         = 40;
    parameter TOTAL_WEIGHT_BITS = 864;
    parameter N_WEIGHT_WORDS    = TOTAL_WEIGHT_BITS / 32;
    parameter RUN_SCENARIOS     = 1;

    parameter CLK_PERIOD     = 10;       // bmi_clk period, ns
    parameter PCLK_PERIOD_PS = 7000;     // pclk period, ps
    parameter PCLK_PHASE_PS  = 0;
    localparam real PCLK_PERIOD = PCLK_PERIOD_PS / 1000.0;
    parameter CLK_FREQ_HZ        = 1_000_000_000 / CLK_PERIOD;
    parameter ADC_SAMPLE_FREQ_HZ = 25_000_000;   // capture every 4 clocks
    localparam integer CLK_DIV =
        (CLK_FREQ_HZ + ADC_SAMPLE_FREQ_HZ - 1) / ADC_SAMPLE_FREQ_HZ;
    localparam integer WINDOW_CLKS = N_CH * N_SAMPLES * CLK_DIV;

    localparam [11:0] ADDR_COMMAND = 12'h000, ADDR_STATUS = 12'h004,
                      ADDR_IRQ_EN  = 12'h008, ADDR_ACK    = 12'h00C,
                      ADDR_RESULT  = 12'h010, ADDR_SCORE0 = 12'h014,
                      ADDR_COMMIT  = 12'h024, ADDR_ERROR  = 12'h028,
                      ADDR_WEIGHT0 = 12'h100;
    localparam [31:0] ST_RESULT_VALID = 32'h01, ST_CMD_BUSY = 32'h02,
                      ST_WT_BUSY = 32'h04, ST_WT_VALID = 32'h08, ST_RUNNING = 32'h10;
    localparam [31:0] CMD_START = 32'd1, CMD_STOP = 32'd2, CMD_RESTART = 32'd3;

    reg                  bmi_clk, pclk, presetn;
    reg  [ADC_WIDTH-1:0] adc_sample;
    wire [2:0]           adc_channel;
    reg                  psel, penable, pwrite;
    reg  [11:0]          paddr;
    reg  [31:0]          pwdata;
    wire [31:0]          prdata;
    wire                 pready, pslverr, irq;

    bmi_chip_top #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ), .ADC_SAMPLE_FREQ_HZ(ADC_SAMPLE_FREQ_HZ)
    ) dut (
        .bmi_clk(bmi_clk), .pclk(pclk), .presetn(presetn),
        .adc_sample(adc_sample), .adc_channel(adc_channel),
        .psel(psel), .penable(penable), .pwrite(pwrite),
        .paddr(paddr), .pwdata(pwdata),
        .prdata(prdata), .pready(pready), .pslverr(pslverr),
        .irq(irq)
    );

    initial bmi_clk = 0;
    always #(CLK_PERIOD/2.0) bmi_clk = ~bmi_clk;

    initial begin
        pclk = 0;
        #(PCLK_PHASE_PS/1000.0);
        forever #(PCLK_PERIOD/2.0) pclk = ~pclk;
    end

    // ---- test data ----
    reg [7:0] adc_mem  [0:N_CH*N_SAMPLES-1];
    reg [7:0] exp_class [0:N_VECTORS-1];
    reg [7:0] weights  [0:TOTAL_WEIGHT_BITS/8-1];
    reg [7:0]         exp_sbp_all    [0:N_VECTORS-1][0:N_CH-1];
    reg signed [63:0] exp_scores_all [0:N_VECTORS-1][0:3];
    reg [7:0]         tmp_sbp    [0:N_CH-1];
    reg signed [63:0] tmp_scores [0:3];

    integer pass_count = 0, fail_count = 0;
    integer apb_error_count = 0;
    integer weight_load_errors = 0;
    integer pipeline_error_count = 0;
    integer scenario_error_count = 0;
    integer scenario_checks = 0;

    // =====================================================================
    // bmi_clk side: scoreboard, golden checks, pipeline monitor
    // =====================================================================
    // Signals are sampled at the clock edge, before the DUT's nonblocking
    // updates, so each value is the one that was valid during the cycle.
    reg [129:0] sb [0:1023];        // entries pushed into the result FIFO
    integer sb_wr, sb_rd;
    integer pushed_count;           // results queued, over the whole run
    integer dropped_count;          // results dropped because the FIFO was full
    integer aborted_count;          // inferences discarded by a clear
    integer window_count, mlp_count, decision_count;
    integer cmd_exec_count;         // commands executed by the controller
    integer commit_exec_count;
    integer cyc;
    integer window_done_cyc, mlp_done_cyc, clear_cyc;
    reg     in_flight;              // between features_done and decision
    reg     awaiting_decision;
    reg     prev_sbp_done, prev_mlp_done, prev_decision, prev_clear;
    reg     resume_expected;
    reg     stop_hit_inference;     // last STOP was accepted with an inference in flight
    reg     restart_hit_inference;  // last RESTART discarded an inference
    reg [TOTAL_WEIGHT_BITS-1:0] prev_active_weights;

    reg     golden_active;
    integer fe_vec, dec_vec;        // golden vector index at the frontend / decision

    localparam integer FRONTEND_TO_MLP_DONE = 122;
    localparam integer MLP_TO_DECISION      = 1;

    task pipe_fail;
        input [8*80-1:0] msg;
        begin
            $display("PIPELINE FAIL @%0t: %0s", $time, msg);
            pipeline_error_count = pipeline_error_count + 1;
        end
    endtask

    integer mi;
    always @(posedge bmi_clk) begin : bmi_monitor
        cyc = cyc + 1;
        if (!dut.bmi_rst_n) begin
            sb_wr = 0; sb_rd = 0;
            in_flight = 0; awaiting_decision = 0; resume_expected = 0;
            prev_sbp_done = 0; prev_mlp_done = 0; prev_decision = 0; prev_clear = 0;
            prev_active_weights = '0;
        end else begin
            // -- pulses are one cycle wide --
            if (dut.sbp_done && prev_sbp_done)        pipe_fail("features_done wider than one cycle");
            if (dut.mlp_done && prev_mlp_done)        pipe_fail("mlp_done wider than one cycle");
            if (dut.decision_valid && prev_decision)  pipe_fail("decision_valid wider than one cycle");
            if (dut.pipe_clear && prev_clear)         pipe_fail("pipe_clear wider than one cycle");

            // -- the frontend resumed, at channel 0, right after the decision --
            if (resume_expected) begin
                if (dut.u_frontend.state !== dut.u_frontend.COLLECT)
                    pipe_fail("frontend did not resume after the decision");
                if (dut.adc_channel !== 0)
                    pipe_fail("frontend did not resume at channel 0");
                resume_expected = 0;
            end

            // -- window complete --
            if (dut.sbp_done) begin
                window_count = window_count + 1;
                if (in_flight) pipe_fail("new window completed during an inference");
                if (!dut.weights_valid_b) pipe_fail("window completed without valid weights");
                for (mi = 0; mi < N_CH; mi = mi + 1)
                    if (^dut.sbp_features[mi] === 1'bx) pipe_fail("X in SBP features");
                in_flight = 1;
                window_done_cyc = cyc;
                if (golden_active && fe_vec < N_VECTORS) begin
                    for (mi = 0; mi < N_CH; mi = mi + 1)
                        if (dut.sbp_features[mi] !== exp_sbp_all[fe_vec][mi]) begin
                            $display("FAIL vec%02d: SBP[%0d] got=%0d expected=%0d", fe_vec, mi,
                                     dut.sbp_features[mi], exp_sbp_all[fe_vec][mi]);
                            fail_count = fail_count + 1;
                        end
                    // The frontend is paused for the inference; present the
                    // next vector before it resumes.
                    if (fe_vec + 1 < N_VECTORS)
                        $readmemh($sformatf("vectors/vec%02d_adc.hex", fe_vec + 1), adc_mem);
                    fe_vec = fe_vec + 1;
                end
            end

            if (dut.mlp_done) begin
                mlp_count = mlp_count + 1;
                if (!in_flight)
                    pipe_fail("mlp_done without a completed window");
                else if (cyc - window_done_cyc != FRONTEND_TO_MLP_DONE)
                    pipe_fail("frontend->MLP latency is not 122 clocks");
                awaiting_decision = 1;
                mlp_done_cyc = cyc;
            end

            if (dut.decision_valid) begin
                decision_count = decision_count + 1;
                if (!awaiting_decision)
                    pipe_fail("decision without MLP completion");
                else if (cyc - mlp_done_cyc != MLP_TO_DECISION)
                    pipe_fail("MLP->decision latency is not 1 clock");
                if (^{dut.predicted_class, dut.class_scores[0], dut.class_scores[1],
                      dut.class_scores[2], dut.class_scores[3]} === 1'bx)
                    pipe_fail("X in a valid result");
                if (golden_active && dec_vec < N_VECTORS) begin
                    for (mi = 0; mi < 4; mi = mi + 1)
                        if (dut.class_scores[mi] !== exp_scores_all[dec_vec][mi][31:0]) begin
                            $display("FAIL vec%02d: internal score[%0d] got=%0d expected=%0d",
                                     dec_vec, mi, $signed(dut.class_scores[mi]),
                                     exp_scores_all[dec_vec][mi]);
                            fail_count = fail_count + 1;
                        end
                    dec_vec = dec_vec + 1;
                end
                // Result leaves the pipeline: queued, dropped, or discarded.
                if (dut.pipe_clear) begin
                    if (dut.fifo_push_b) pipe_fail("result pushed in a clear cycle");
                end else if (dut.fifo_full_b) begin
                    dropped_count = dropped_count + 1;
                    if (dut.fifo_push_b) pipe_fail("push while the FIFO is full");
                end else begin
                    if (!dut.fifo_push_b) pipe_fail("result neither pushed nor dropped");
                    sb[sb_wr] = {dut.class_scores[3], dut.class_scores[2],
                                 dut.class_scores[1], dut.class_scores[0],
                                 dut.predicted_class};
                    sb_wr = sb_wr + 1;
                    pushed_count = pushed_count + 1;
                    resume_expected = 1;
                end
                if (!dut.pipe_clear && dut.fifo_full_b) resume_expected = 1;
                in_flight = 0;
                awaiting_decision = 0;
            end else if (dut.fifo_push_b) begin
                pipe_fail("FIFO push without a decision");
            end

            // -- synchronous clear (RESTART, or the end of a STOP) --
            if (dut.pipe_clear) begin
                clear_cyc = cyc;
                if (in_flight || dut.sbp_done) begin
                    aborted_count = aborted_count + 1;
                    restart_hit_inference = 1;
                end
                in_flight = 0; awaiting_decision = 0; resume_expected = 0;
            end

            // -- command execution --
            if (dut.cmd_valid_b && dut.u_ctrl.state_q == 0 && dut.cmd_code_b == 2'd2)
                stop_hit_inference = in_flight || dut.sbp_done;
            if (dut.cmd_done_b) begin
                cmd_exec_count = cmd_exec_count + 1;
                if (!dut.cmd_valid_b) pipe_fail("cmd_done without a pending command");
                if (dut.cmd_code_b == 2'd2) begin
                    // STOP may only be acknowledged with nothing in flight.
                    if (in_flight && !dut.pipe_clear) pipe_fail("STOP acknowledged mid-inference");
                    if (dut.u_mlp.state !== dut.u_mlp.S_IDLE || dut.mlp_done ||
                        dut.decision_valid || dut.sbp_done)
                        pipe_fail("STOP acknowledged while the pipeline was active");
                    if (dut.u_ctrl.running_q !== 1'b0) pipe_fail("STOP acknowledged while running");
                end
            end
            if (dut.wt_done_b) begin
                commit_exec_count = commit_exec_count + 1;
                if (dut.u_ctrl.running_q) pipe_fail("weights committed while running");
            end

            // -- active weights: change only while stopped, all bits at once --
            if (dut.active_weights_b !== prev_active_weights) begin
                if (dut.u_ctrl.running_q || in_flight)
                    pipe_fail("active weights changed while the pipeline was busy");
                if (dut.active_weights_b !== dut.staged_weights_p)
                    pipe_fail("active weights are not the complete staged image");
            end

            // -- stopped means stopped --
            if (!dut.u_ctrl.running_q && !dut.frontend_enable && prev_clear &&
                (dut.adc_channel !== 0 ||
                 dut.u_frontend.state !== dut.u_frontend.COLLECT))
                pipe_fail("frontend not in its reset state after a clear");

            prev_sbp_done = dut.sbp_done;
            prev_mlp_done = dut.mlp_done;
            prev_decision = dut.decision_valid;
            prev_clear    = dut.pipe_clear;
            prev_active_weights = dut.active_weights_b;
        end
    end

    // ---- ADC capture-rate monitor ----
    // adc_channel advances only on a capture, so channel changes must be
    // exactly CLK_DIV clocks apart. Reset, clear, disable and the pause for
    // an inference all restart the interval.
    integer capture_gap, capture_gap_errors;
    integer previous_monitor_channel;
    reg     capture_was_active;

    always @(posedge bmi_clk) begin : capture_rate_monitor
        #1;
        if (!dut.bmi_rst_n || !capture_was_active || !dut.frontend_enable ||
            prev_clear) begin
            capture_gap = 1;
            previous_monitor_channel = dut.adc_channel;
        end else begin
            if (dut.adc_channel !== previous_monitor_channel) begin
                if (capture_gap != CLK_DIV) begin
                    $display("PIPELINE FAIL @%0t: ADC capture spacing got=%0d expected=%0d clocks",
                             $time, capture_gap, CLK_DIV);
                    capture_gap_errors = capture_gap_errors + 1;
                end
                capture_gap = 1;
            end else if (dut.u_frontend.state === dut.u_frontend.COLLECT) begin
                capture_gap = capture_gap + 1;
            end else begin
                capture_gap = 0;
            end
            previous_monitor_channel = dut.adc_channel;
        end
        capture_was_active = (dut.bmi_rst_n === 1'b1) && (dut.frontend_enable === 1'b1) &&
                             (dut.u_frontend.state === dut.u_frontend.COLLECT ||
                              capture_gap == 0);
    end

    // ---- background ADC model ----
    // Presents adc_mem[ch*N_SAMPLES + index[ch]] at every falling edge. A
    // channel's index advances when the MUX moves off it. The indices return
    // to 0 whenever the frontend restarts a window.
    integer drv_cnt [0:N_CH-1];
    integer frontend_was_paused, prev_channel;

    initial begin : adc_model
        integer ch;
        for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
        frontend_was_paused = 0;
        prev_channel = 0;
        forever begin
            @(negedge bmi_clk);
            ch = dut.adc_channel;
            if (ch != prev_channel && drv_cnt[prev_channel] < N_SAMPLES)
                drv_cnt[prev_channel] = drv_cnt[prev_channel] + 1;
            prev_channel = ch;

            if ((frontend_was_paused &&
                 dut.u_frontend.state === dut.u_frontend.COLLECT) ||
                dut.pipe_clear === 1'b1 || dut.bmi_rst_n !== 1'b1) begin
                for (ch = 0; ch < N_CH; ch++) drv_cnt[ch] = 0;
                prev_channel = 0;
            end
            frontend_was_paused = (dut.u_frontend.state === dut.u_frontend.PAUSE);

            ch = dut.adc_channel;
            adc_sample = adc_mem[ch * N_SAMPLES +
                                 (drv_cnt[ch] < N_SAMPLES ? drv_cnt[ch] : N_SAMPLES - 1)];
        end
    end

    // =====================================================================
    // pclk side: APB master and host model
    // =====================================================================
    integer accepted_cmd_count = 0;     // COMMAND writes that returned OKAY
    integer accepted_commit_count = 0;

    task apb_bus_idle;
        begin psel = 0; penable = 0; pwrite = 0; paddr = 0; pwdata = 0; end
    endtask

    // What the register block should answer to a write whose legality
    // depends on the busy and running flags, evaluated in the access phase.
    function predicted_error;
        input [11:0] addr;
        input [31:0] wdata;
        reg locked;
        begin
            locked = dut.cmd_busy_p || dut.wt_busy_p;
            if (addr == ADDR_COMMAND)
                predicted_error = locked ||
                                  (wdata == CMD_START && !dut.u_apb.weights_valid_q);
            else
                predicted_error = locked || dut.cmd_running_p;
        end
    endfunction

    integer refused_while_busy = 0;   // probes that found the block busy

    // One APB transfer. expect_mode: 0 = must succeed, 1 = must return
    // pslverr, 2 = a probe whose answer depends on how far a handshake has
    // got; the expected answer is then taken from predicted_error.
    task apb_transfer;
        input         write;
        input  [11:0] addr;
        input  [31:0] wdata;
        input  [1:0]  expect_mode;
        output [31:0] rdata;
        reg           expect_error;
        begin
            @(posedge pclk); #1;
            psel = 1; penable = 0; pwrite = write; paddr = addr; pwdata = wdata;
            @(posedge pclk); #1;
            penable = 1;
            @(negedge pclk);
            rdata = prdata;
            expect_error = (expect_mode == 2) ? predicted_error(addr, wdata) : expect_mode[0];
            if (expect_mode == 2 && expect_error)
                refused_while_busy = refused_while_busy + 1;
            if (pslverr !== expect_error || pready !== 1'b1) begin
                $display("APB FAIL @%0t: addr=0x%03X write=%b data=0x%08X pslverr=%b (expected %b) pready=%b",
                         $time, addr, write, wdata, pslverr, expect_error, pready);
                apb_error_count = apb_error_count + 1;
            end
            if (write && pslverr === 1'b0 && addr == ADDR_COMMAND)
                accepted_cmd_count = accepted_cmd_count + 1;
            if (write && pslverr === 1'b0 && addr == ADDR_COMMIT && wdata[0])
                accepted_commit_count = accepted_commit_count + 1;
            @(posedge pclk); #1;
            apb_bus_idle();
        end
    endtask

    reg [31:0] scratch;

    task apb_read;  input [11:0] addr; output [31:0] rdata;
        begin apb_transfer(1'b0, addr, 32'd0, 2'd0, rdata); end
    endtask
    task apb_write; input [11:0] addr; input [31:0] wdata;
        begin apb_transfer(1'b1, addr, wdata, 2'd0, scratch); end
    endtask
    task apb_write_refused; input [11:0] addr; input [31:0] wdata;
        begin apb_transfer(1'b1, addr, wdata, 2'd1, scratch); end
    endtask
    // A write right behind a command or commit. Whether the handshake is
    // still busy depends on the clock ratio, so every probe is chosen to be
    // harmless if it is accepted.
    task apb_write_probe; input [11:0] addr; input [31:0] wdata;
        begin apb_transfer(1'b1, addr, wdata, 2'd2, scratch); end
    endtask

    task check;
        input            ok;
        input [8*80-1:0] msg;
        begin
            scenario_checks = scenario_checks + 1;
            if (ok !== 1'b1) begin
                $display("SCENARIO FAIL @%0t: %0s", $time, msg);
                scenario_error_count = scenario_error_count + 1;
            end
        end
    endtask

    // Poll STATUS until COMMAND_BUSY and WEIGHT_BUSY are both clear.
    task wait_not_busy;
        output [31:0] status;
        integer polls;
        begin
            polls = 0;
            apb_read(ADDR_STATUS, status);
            while (status & (ST_CMD_BUSY | ST_WT_BUSY)) begin
                apb_read(ADDR_STATUS, status);
                polls = polls + 1;
                if (polls > 200000) $fatal(1, "busy flag never cleared");
            end
        end
    endtask

    task command;
        input [31:0] code;
        reg   [31:0] status;
        begin
            apb_write(ADDR_COMMAND, code);
            wait_not_busy(status);
        end
    endtask

    // Write an image to the staging registers (byte b XOR xor_mask), read it
    // back, commit it and check the active image the MLP sees.
    task load_and_commit_weights;
        input [7:0] xor_mask;
        integer    n, b;
        reg [31:0] word, readback, status;
        begin
            for (n = 0; n < N_WEIGHT_WORDS; n = n + 1) begin
                word = {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]}
                       ^ {4{xor_mask}};
                apb_write(ADDR_WEIGHT0 + 4*n, word);
            end
            for (n = 0; n < N_WEIGHT_WORDS; n = n + 1) begin
                word = {weights[4*n+3], weights[4*n+2], weights[4*n+1], weights[4*n]}
                       ^ {4{xor_mask}};
                apb_read(ADDR_WEIGHT0 + 4*n, readback);
                if (readback !== word) begin
                    $display("WEIGHT FAIL: word %0d read 0x%08X expected 0x%08X", n, readback, word);
                    weight_load_errors = weight_load_errors + 1;
                end
            end
            apb_write(ADDR_COMMIT, 32'd1);
            // Frozen while WEIGHT_BUSY, and no commands during a commit.
            // (Rewrites word 0 with its own value; STOP while stopped.)
            apb_write_probe(ADDR_WEIGHT0, {weights[3], weights[2], weights[1], weights[0]}
                                          ^ {4{xor_mask}});
            apb_write_probe(ADDR_COMMAND, CMD_STOP);
            wait_not_busy(status);
            if (!(status & ST_WT_VALID)) begin
                $display("WEIGHT FAIL: WEIGHTS_VALID not set after the commit");
                weight_load_errors = weight_load_errors + 1;
            end
            for (b = 0; b < TOTAL_WEIGHT_BITS/8; b = b + 1)
                if (dut.active_weights_b[b*8 +: 8] !== (weights[b] ^ xor_mask)) begin
                    $display("WEIGHT FAIL: active byte %0d is 0x%02X expected 0x%02X",
                             b, dut.active_weights_b[b*8 +: 8], weights[b] ^ xor_mask);
                    weight_load_errors = weight_load_errors + 1;
                end
        end
    endtask

    // Read the snapshot: RESULT and SCORE0..3 into one entry.
    task read_snapshot;
        output [129:0] entry;
        reg [31:0] r;
        integer i;
        begin
            apb_read(ADDR_RESULT, r);
            entry[1:0] = r[1:0];
            if (r[31:2] !== 30'd0) begin
                $display("APB FAIL: reserved RESULT bits 0x%08X", r);
                apb_error_count = apb_error_count + 1;
            end
            for (i = 0; i < 4; i = i + 1) begin
                apb_read(ADDR_SCORE0 + 4*i, r);
                entry[2 + 32*i +: 32] = r;
            end
        end
    endtask

    task wait_result_valid;
        reg [31:0] status;
        begin
            status = 0;
            while (!(status & ST_RESULT_VALID)) apb_read(ADDR_STATUS, status);
        end
    endtask

    // Take the next result: read it, compare it with the scoreboard, read it
    // again after hold_pclks (it must not have changed), then acknowledge.
    task take_result;
        input  integer hold_pclks;
        output [129:0] entry;
        reg [129:0] again;
        begin
            read_snapshot(entry);
            scenario_checks = scenario_checks + 1;
            if (sb_rd >= sb_wr || entry !== sb[sb_rd]) begin
                $display("SCOREBOARD FAIL @%0t: host read class=%0d scores %08X %08X %08X %08X, entry %0d of %0d expected class=%0d %08X %08X %08X %08X",
                         $time, entry[1:0], entry[33:2], entry[65:34], entry[97:66], entry[129:98],
                         sb_rd, sb_wr, sb[sb_rd][1:0], sb[sb_rd][33:2], sb[sb_rd][65:34],
                         sb[sb_rd][97:66], sb[sb_rd][129:98]);
                scenario_error_count = scenario_error_count + 1;
            end
            if (hold_pclks > 0) begin
                repeat (hold_pclks) @(posedge pclk);
                read_snapshot(again);
                scenario_checks = scenario_checks + 1;
                if (again !== entry) begin
                    $display("SCOREBOARD FAIL @%0t: snapshot changed while held", $time);
                    scenario_error_count = scenario_error_count + 1;
                end
            end
            sb_rd = sb_rd + 1;
            apb_write(ADDR_ACK, 32'd1);
        end
    endtask

    // Read and acknowledge until nothing is queued any more.
    task drain_results;
        reg [129:0] entry;
        reg [31:0]  status;
        integer     idle_polls;
        begin
            idle_polls = 0;
            while (idle_polls < 12) begin
                apb_read(ADDR_STATUS, status);
                if (status & ST_RESULT_VALID) begin
                    take_result(0, entry);
                    idle_polls = 0;
                end else
                    idle_polls = idle_polls + 1;
            end
        end
    endtask

    task wait_bmi; input integer n; begin repeat (n) @(posedge bmi_clk); end endtask

    // ---------------------------------------------------------------------
    // Part 1: golden vectors
    // ---------------------------------------------------------------------
    task golden_vectors;
        input integer n_vec;
        integer    vec, i;
        reg [129:0] entry;
        reg [31:0]  status;
        reg         vector_pass;
        begin
            fe_vec = 0; dec_vec = 0;
            $readmemh("vectors/vec00_adc.hex", adc_mem);
            golden_active = 1;
            apb_write(ADDR_IRQ_EN, 32'd1);
            command(CMD_START);

            for (vec = 0; vec < n_vec; vec = vec + 1) begin
                if (vec % 2 == 0) begin
                    wait (irq === 1'b1);
                end else begin
                    wait_result_valid();
                    if (irq !== 1'b0) begin
                        $display("APB FAIL vec%02d: irq high while IRQ_EN is 0", vec);
                        apb_error_count = apb_error_count + 1;
                    end
                end

                // On some vectors the host is slow: more than two windows
                // pass, so later results queue behind this snapshot.
                if (vec % 8 == 5)
                    wait_bmi(2*WINDOW_CLKS + WINDOW_CLKS/3);

                apb_read(ADDR_STATUS, status);
                if (status !== (ST_RUNNING | ST_WT_VALID | ST_RESULT_VALID)) begin
                    $display("APB FAIL vec%02d: STATUS=0x%08X with a result waiting", vec, status);
                    apb_error_count = apb_error_count + 1;
                end

                // Detection method for the next vector, chosen before this
                // one is released.
                apb_write(ADDR_IRQ_EN, ((vec + 1) % 2 == 0) ? 32'd1 : 32'd0);
                take_result(3 + (vec % 4) * 40, entry);

                vector_pass = 1'b1;
                if (entry[1:0] !== exp_class[vec][1:0]) begin
                    $display("FAIL vec%02d: class got=%0d expected=%0d",
                             vec, entry[1:0], exp_class[vec][1:0]);
                    vector_pass = 1'b0;
                end
                for (i = 0; i < 4; i = i + 1) begin
                    if (exp_scores_all[vec][i][63:32] !== {32{exp_scores_all[vec][i][31]}}) begin
                        $display("FAIL vec%02d: golden score[%0d] exceeds 32 bits", vec, i);
                        vector_pass = 1'b0;
                    end
                    if (entry[2 + 32*i +: 32] !== exp_scores_all[vec][i][31:0]) begin
                        $display("FAIL vec%02d: SCORE%0d got=0x%08X expected=0x%08X",
                                 vec, i, entry[2 + 32*i +: 32], exp_scores_all[vec][i][31:0]);
                        vector_pass = 1'b0;
                    end
                end
                if (vector_pass) begin
                    $display("PASS vec%02d: class=%0d scores=%0d %0d %0d %0d", vec, entry[1:0],
                             $signed(entry[33:2]), $signed(entry[65:34]),
                             $signed(entry[97:66]), $signed(entry[129:98]));
                    pass_count = pass_count + 1;
                end else
                    fail_count = fail_count + 1;
            end

            command(CMD_STOP);
            golden_active = 0;
            apb_write(ADDR_IRQ_EN, 32'd0);
            drain_results();          // windows completed after the last vector
            apb_read(ADDR_STATUS, status);
            check(status === ST_WT_VALID, "STATUS after STOP and drain");
            apb_read(ADDR_ERROR, status);
            check(status === 32'd0, "no overflow in the golden run");
        end
    endtask

    // ---------------------------------------------------------------------
    // Part 2: directed scenarios. Each starts and ends stopped and drained.
    // ---------------------------------------------------------------------

    // Commands issued as fast as the bus allows. A second command while the
    // first is busy must be refused and must not execute.
    task scenario_back_to_back_commands;
        integer    i, exec_before, accepted_before;
        reg [31:0] status, code;
        reg        model_running;
        begin
            $display("SCENARIO back-to-back commands");
            exec_before = cmd_exec_count;
            accepted_before = accepted_cmd_count;
            model_running = 0;
            for (i = 0; i < 18; i = i + 1) begin
                case (i % 6)
                    0: code = CMD_START;
                    1: code = CMD_RESTART;
                    2: code = CMD_START;      // START while running: legal, no change
                    3: code = CMD_STOP;
                    4: code = CMD_STOP;       // STOP while stopped: legal
                    default: code = CMD_RESTART;
                endcase
                apb_write(ADDR_COMMAND, code);
                // Right behind it: the same command again, a staged-weight
                // write (word 0, its own value) and a commit (same image).
                apb_write_probe(ADDR_COMMAND, code);
                apb_write_probe(ADDR_WEIGHT0, {weights[3], weights[2], weights[1], weights[0]});
                apb_write_probe(ADDR_COMMIT, 32'd1);
                if (code == CMD_START) model_running = 1;
                if (code == CMD_STOP)  model_running = 0;
                wait_not_busy(status);
                check(status[4] === model_running, "RUNNING after a command");
                check(dut.u_ctrl.running_q === model_running, "controller RUNNING after a command");
            end
            wait_bmi(8);
            check(cmd_exec_count - exec_before == accepted_cmd_count - accepted_before,
                  "each accepted command executed exactly once");
            apb_write_refused(ADDR_COMMAND, 32'd0);
            apb_write_refused(ADDR_COMMAND, 32'd4);
            apb_write_refused(ADDR_COMMAND, 32'hFFFF_FFFF);
            wait_bmi(40);
            check(cmd_exec_count - exec_before == accepted_cmd_count - accepted_before,
                  "invalid command codes not executed");
            drain_results();
        end
    endtask

    // STOP sent just as an inference starts: the acknowledgment must wait for
    // it, and its result must be queued. Then the weights are replaced.
    task scenario_stop_and_recommit;
        integer    before_sb, before_windows, i;
        reg [31:0] status;
        reg [129:0] entry;
        begin
            $display("SCENARIO STOP during an inference, weight recommit");
            command(CMD_START);
            apb_write_refused(ADDR_WEIGHT0 + 4, 32'h0);     // running: locked
            apb_write_refused(ADDR_COMMIT, 32'd1);
            before_sb = sb_wr;
            @(posedge dut.sbp_done);
            apb_write(ADDR_COMMAND, CMD_STOP);
            wait_not_busy(status);
            check(status[4] === 1'b0, "RUNNING clear after STOP");
            if (stop_hit_inference)
                check(sb_wr == before_sb + 1, "in-flight result queued before STOP acknowledged");
            else
                $display("  note: STOP arrived after the inference at this clock ratio");
            wait_result_valid();
            take_result(5, entry);

            // Stopped: no samples for half a window.
            before_windows = window_count;
            for (i = 0; i < 50; i = i + 1) begin
                wait_bmi(WINDOW_CLKS/100);
                check(dut.adc_channel === 0 && dut.u_frontend.sample_cnt[0] === 0,
                      "frontend idle while stopped");
            end
            check(window_count == before_windows, "no window while stopped");

            // Replace the weights with a different image, then restore.
            load_and_commit_weights(8'hFF);
            load_and_commit_weights(8'h00);
            apb_read(ADDR_STATUS, status);
            check(status === ST_WT_VALID, "STATUS after recommit");
        end
    endtask

    // RESTART: mid-window, during an inference, and while stopped. Queued
    // results and the weights must survive.
    task scenario_restart;
        integer    before_sb, before_abort, t0, i;
        reg [31:0] status;
        reg [129:0] entry;
        reg [TOTAL_WEIGHT_BITS-1:0] weights_before;
        begin
            $display("SCENARIO RESTART");
            weights_before = dut.active_weights_b;
            command(CMD_START);
            // Two results left unread: one in the snapshot, one in the FIFO.
            before_sb = sb_wr;
            wait (sb_wr == before_sb + 2);
            // a) mid-window
            wait (dut.u_frontend.sample_cnt[3] == 100);
            before_abort = aborted_count;
            apb_write(ADDR_COMMAND, CMD_RESTART);
            wait_not_busy(status);
            check(status === (ST_RUNNING | ST_WT_VALID | ST_RESULT_VALID),
                  "RESTART keeps RUNNING, WEIGHTS_VALID and the pending result");
            check(aborted_count == before_abort, "mid-window RESTART discards no inference");
            t0 = clear_cyc;
            @(posedge dut.sbp_done);
            @(posedge bmi_clk);
            check(cyc - t0 >= WINDOW_CLKS && cyc - t0 <= WINDOW_CLKS + 3,
                  "a full fresh window follows RESTART");
            // The two results from before the RESTART, in order.
            take_result(4, entry);
            wait_result_valid();
            take_result(4, entry);
            check(sb_rd == before_sb + 2, "queued results preserved across RESTART");

            // b) during an inference: that result is discarded
            drain_results();
            @(posedge dut.sbp_done);
            before_sb = sb_wr; before_abort = aborted_count;
            restart_hit_inference = 0;
            apb_write(ADDR_COMMAND, CMD_RESTART);
            wait_not_busy(status);
            wait_bmi(300);
            if (restart_hit_inference) begin
                check(aborted_count == before_abort + 1 && sb_wr == before_sb,
                      "RESTART during an inference discards exactly that result");
                apb_read(ADDR_STATUS, status);
                check(status[0] === (sb_rd < sb_wr), "no result from a discarded inference");
            end else
                $display("  note: RESTART arrived after the inference at this clock ratio");
            check(dut.active_weights_b === weights_before, "weights preserved across RESTART");

            // c) STOP in the middle of a window discards the partial window,
            //    and RESTART while stopped stays stopped
            wait (dut.u_frontend.sample_cnt[2] == 50);
            command(CMD_STOP);
            check(dut.adc_channel === 0 && dut.u_frontend.sample_cnt[0] === 0 &&
                  dut.u_frontend.sample_cnt[2] === 0 && dut.u_frontend.deviation_sum[1] === 0,
                  "STOP discards the partial window");
            drain_results();
            command(CMD_RESTART);
            apb_read(ADDR_STATUS, status);
            check(status === ST_WT_VALID, "RESTART while stopped stays stopped");
            for (i = 0; i < 10; i = i + 1) begin
                wait_bmi(60);
                check(dut.adc_channel === 0, "frontend idle after RESTART while stopped");
            end
        end
    endtask

    // The host stops acknowledging. One result sits in the snapshot, four in
    // the FIFO, the next is dropped and the sticky overflow flag is set. The
    // snapshot must not move during all of this.
    task scenario_fifo_overflow;
        integer    before_sb, before_drop, before_windows, i;
        reg [31:0] status;
        reg [129:0] held_entry, entry;
        begin
            $display("SCENARIO FIFO full and overflow");
            apb_read(ADDR_ERROR, status);
            check(status === 32'd0, "ERROR_STATUS clear before the overflow");
            before_sb = sb_wr; before_drop = dropped_count; before_windows = window_count;
            command(CMD_START);
            wait_result_valid();
            read_snapshot(held_entry);
            check(held_entry === sb[sb_rd], "first result in the snapshot");
            while (dropped_count == before_drop) begin
                wait_bmi(WINDOW_CLKS/3 + 17);
                // All five registers, read while new results keep arriving.
                read_snapshot(entry);
                check(entry === held_entry, "snapshot stable while results queue");
                check(irq === 1'b0, "irq low with IRQ_EN clear");
            end
            check(sb_wr - before_sb == 5, "five results held: one snapshot, four in the FIFO");
            repeat (6) @(posedge pclk);
            apb_read(ADDR_ERROR, status);
            check(status === 32'd1, "ERROR_STATUS set after a dropped result");
            apb_write_refused(ADDR_ERROR, 32'd1);
            // The pipeline did not wait for the host.
            wait_bmi(WINDOW_CLKS + 200);
            check(window_count - before_windows >= 7, "pipeline kept running while full");
            check(dropped_count - before_drop >= 2, "further results dropped while full");
            // Drain: the five held results come out in order, unchanged.
            for (i = 0; i < 5; i = i + 1) begin
                wait_result_valid();
                take_result(i, entry);
            end
            check(sb_rd == before_sb + 5, "five results drained in order");
            command(CMD_STOP);
            drain_results();
            apb_read(ADDR_ERROR, status);
            check(status === 32'd1, "overflow flag is sticky");
        end
    endtask

    // Acknowledge at a swept offset around the moment the next result
    // arrives, so RESULT_ACK, the FIFO write and the next fetch overlap in
    // every order.
    task scenario_ack_races;
        integer    i, before_sb;
        reg [129:0] entry;
        reg [31:0]  status;
        begin
            $display("SCENARIO acknowledgment races");
            before_sb = sb_rd;
            apb_write(ADDR_IRQ_EN, 32'd1);
            command(CMD_START);
            wait_result_valid();
            for (i = 0; i < 10; i = i + 1) begin
                // A result is held. Wait for the next decision, then
                // acknowledge i pclk cycles later.
                @(posedge dut.decision_valid);
                repeat (i) @(posedge pclk);
                take_result(0, entry);
                // The next one must be presented, with a fresh irq edge.
                wait (irq === 1'b0);
                wait (irq === 1'b1);
            end
            command(CMD_STOP);
            apb_write(ADDR_IRQ_EN, 32'd0);
            drain_results();
            check(sb_rd == sb_wr && sb_rd - before_sb >= 11, "no result lost or repeated");
            apb_read(ADDR_STATUS, status);
            check(status === ST_WT_VALID, "STATUS after the race scenario");
        end
    endtask

    // Global reset while running with results queued, then a complete
    // restart from nothing with two golden vectors.
    task scenario_global_reset;
        integer    n;
        reg [31:0] status, r;
        reg [129:0] entry;
        begin
            $display("SCENARIO global reset while running");
            command(CMD_START);
            n = sb_wr;
            wait (sb_wr == n + 2);              // snapshot + one FIFO entry
            wait (dut.u_frontend.sample_cnt[5] == 77);
            #3 presetn = 0;
            apb_bus_idle();
            #(2.5*PCLK_PERIOD + 2.5*CLK_PERIOD);
            check(irq === 1'b0 && prdata === 32'd0 && dut.adc_channel === 0, "outputs idle in reset");
            presetn = 1;
            repeat (3) @(posedge pclk);
            apb_read(ADDR_STATUS, status);
            check(status === 32'd0, "STATUS clear after reset");
            apb_read(ADDR_ERROR, r);   check(r === 32'd0, "ERROR_STATUS clear after reset");
            apb_read(ADDR_IRQ_EN, r);  check(r === 32'd0, "IRQ_EN clear after reset");
            apb_read(ADDR_RESULT, r);  check(r === 32'd0, "RESULT clear after reset");
            for (n = 0; n < N_WEIGHT_WORDS; n = n + 1) begin
                apb_read(ADDR_WEIGHT0 + 4*n, r);
                check(r === 32'd0, "staged weights clear after reset");
            end
            apb_write_refused(ADDR_COMMAND, CMD_START);   // weights no longer valid
            wait_bmi(WINDOW_CLKS/4);
            check(dut.adc_channel === 0 && dut.u_ctrl.running_q === 1'b0 &&
                  dut.active_weights_b === '0 && dut.fifo_empty_p === 1'b1,
                  "pipeline, active weights and FIFO cleared by reset");

            load_and_commit_weights(8'h00);
            fe_vec = 0; dec_vec = 0;
            $readmemh("vectors/vec00_adc.hex", adc_mem);
            golden_active = 1;
            command(CMD_START);
            for (n = 0; n < 2; n = n + 1) begin
                wait_result_valid();
                take_result(2, entry);
                check(entry[1:0] === exp_class[n][1:0] &&
                      entry[33:2]   === exp_scores_all[n][0][31:0] &&
                      entry[65:34]  === exp_scores_all[n][1][31:0] &&
                      entry[97:66]  === exp_scores_all[n][2][31:0] &&
                      entry[129:98] === exp_scores_all[n][3][31:0],
                      "golden result after reset and reload");
            end
            command(CMD_STOP);
            golden_active = 0;
            drain_results();
        end
    endtask

    // ---------------------------------------------------------------------
    // Main sequence
    // ---------------------------------------------------------------------
    integer v, c;
    reg [31:0] startup_status;

    initial begin
        presetn = 0;
        adc_sample = 0;
        apb_bus_idle();
        cyc = 0; sb_wr = 0; sb_rd = 0;
        dropped_count = 0; aborted_count = 0; pushed_count = 0;
        window_count = 0; mlp_count = 0; decision_count = 0;
        cmd_exec_count = 0; commit_exec_count = 0;
        capture_gap_errors = 0; capture_was_active = 0;
        golden_active = 0; fe_vec = 0; dec_vec = 0;
        stop_hit_inference = 0; restart_hit_inference = 0;
        clear_cyc = 0;

        $readmemh("weights.hex", weights);
        $readmemh("vectors/all_expected.hex", exp_class);
        for (v = 0; v < N_VECTORS; v = v + 1) begin
            $readmemh($sformatf("vectors/vec%02d_sbp.hex", v), tmp_sbp);
            $readmemh($sformatf("vectors/vec%02d_scores.hex", v), tmp_scores);
            for (c = 0; c < N_CH; c = c + 1) exp_sbp_all[v][c] = tmp_sbp[c];
            for (c = 0; c < 4; c = c + 1)    exp_scores_all[v][c] = tmp_scores[c];
        end
        $readmemh("vectors/vec00_adc.hex", adc_mem);

        repeat (4) @(posedge pclk);
        repeat (4) @(posedge bmi_clk);
        @(negedge pclk);
        presetn = 1;
        repeat (3) @(posedge pclk);

        // Out of reset: stopped, nothing valid, START refused.
        apb_read(ADDR_STATUS, startup_status);
        check(startup_status === 32'd0, "STATUS is 0 after reset");
        apb_write_refused(ADDR_COMMAND, CMD_START);
        wait_bmi(200);
        check(dut.u_ctrl.running_q === 1'b0 && dut.adc_channel === 0,
              "pipeline idle before any command");

        load_and_commit_weights(8'h00);
        $display("Weights loaded and committed over APB: %0d words, %0d errors.",
                 N_WEIGHT_WORDS, weight_load_errors);

        golden_vectors(N_VECTORS);

        $display("----------------------------------------------");
        $display("RTL SIM  %0d / %0d  PASS", pass_count, N_VECTORS);
        $display("----------------------------------------------");

        if (RUN_SCENARIOS) begin
            scenario_back_to_back_commands();
            scenario_stop_and_recommit();
            scenario_restart();
            scenario_fifo_overflow();
            scenario_ack_races();
            scenario_global_reset();
        end

        wait_bmi(20);
        if (accepted_cmd_count != cmd_exec_count) begin
            $display("PIPELINE FAIL: %0d commands accepted over APB, %0d executed",
                     accepted_cmd_count, cmd_exec_count);
            pipeline_error_count = pipeline_error_count + 1;
        end
        if (accepted_commit_count != commit_exec_count) begin
            $display("PIPELINE FAIL: %0d commits accepted over APB, %0d executed",
                     accepted_commit_count, commit_exec_count);
            pipeline_error_count = pipeline_error_count + 1;
        end
        // Every completed window became exactly one of: queued, dropped
        // (FIFO full), or discarded (RESTART).
        if (window_count != pushed_count + dropped_count + aborted_count) begin
            $display("PIPELINE FAIL: %0d windows but %0d queued + %0d dropped + %0d discarded",
                     window_count, pushed_count, dropped_count, aborted_count);
            pipeline_error_count = pipeline_error_count + 1;
        end
        if (sb_rd != sb_wr) begin
            $display("PIPELINE FAIL: %0d results queued but never read", sb_wr - sb_rd);
            pipeline_error_count = pipeline_error_count + 1;
        end

        pipeline_error_count = pipeline_error_count + capture_gap_errors;
        $display("Windows %0d, decisions %0d, dropped %0d, discarded %0d, commands %0d, commits %0d",
                 window_count, decision_count, dropped_count, aborted_count,
                 cmd_exec_count, commit_exec_count);
        $display("APB CHECKS: %0d errors, WEIGHT LOAD: %0d errors, %0d writes refused while busy",
                 apb_error_count, weight_load_errors, refused_while_busy);
        $display("SCENARIO CHECKS: %0d checks, %0d errors", scenario_checks, scenario_error_count);
        $display("PIPELINE CHECKS: %0d errors", pipeline_error_count);
        $display("bmi_clk %0d ns, pclk %0d ps (phase %0d ps), ADC capture every %0d clocks",
                 CLK_PERIOD, PCLK_PERIOD_PS, PCLK_PHASE_PS, CLK_DIV);

        if (fail_count == 0 && pass_count == N_VECTORS && pipeline_error_count == 0 &&
            apb_error_count == 0 && weight_load_errors == 0 && scenario_error_count == 0) begin
            $display("ALL PASS");
            $finish;
        end else
            $fatal(1, "%0d vector failures, %0d pipeline, %0d APB, %0d weight, %0d scenario errors",
                   fail_count, pipeline_error_count, apb_error_count, weight_load_errors,
                   scenario_error_count);
    end

    // Independent watchdog: a deadlock fails instead of hanging.
    initial begin
        #(64'd120 * WINDOW_CLKS * CLK_PERIOD + 4000.0 * PCLK_PERIOD);
        $fatal(1, "Top-level regression timeout");
    end

endmodule
