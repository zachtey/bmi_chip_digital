// apb_slave.sv
// AMBA APB3 register block for the BMI classifier. Everything here is in the
// pclk domain (clk = pclk, rst_n = the pclk-synchronized global reset). The
// pipeline runs on an unrelated clock; this block reaches it only through
// the CDC modules instantiated in bmi_chip_top:
//   command mailbox       cmd_send / cmd_code  ->  cmd_busy, cmd_running
//   weight-commit mailbox wt_send / weight_image -> wt_busy, wt_done
//   result FIFO           fifo_empty / fifo_rdata / fifo_pop
//   overflow              already synchronized level
//
// Register map (32-bit, word aligned)
//   0x000 COMMAND        WO  1 = START, 2 = STOP, 3 = RESTART
//   0x004 STATUS         RO  [0] RESULT_VALID  [1] COMMAND_BUSY
//                            [2] WEIGHT_BUSY   [3] WEIGHTS_VALID  [4] RUNNING
//   0x008 IRQ_EN         RW  [0] result interrupt enable
//   0x00C RESULT_ACK     WO  write [0] = 1 to release the result snapshot
//   0x010 RESULT         RO  [1:0] predicted class
//   0x014 SCORE0..3      RO  signed 32-bit score (0x014, 0x018, 0x01C, 0x020)
//   0x024 WEIGHT_COMMIT  WO  write [0] = 1 to commit the staged image
//   0x028 ERROR_STATUS   RO  [0] sticky result overflow
//   0x100 WEIGHT0..26    RW  staged weight image, four bytes per word:
//                            image byte i is bits [8*(i%4) +: 8] of word i/4
//
// Behaviour
//   * Result snapshot: when no snapshot is held and the FIFO is not empty,
//     one entry is popped into RESULT/SCOREn and RESULT_VALID is set. The
//     snapshot then does not change until RESULT_ACK. Reads never pop. After
//     RESULT_ACK, RESULT_VALID is low for at least one pclk before the next
//     entry (if any) is loaded, so irq always makes a fresh rising edge.
//   * irq = RESULT_VALID & IRQ_EN.
//   * RUNNING is the value returned with the last command acknowledgment. It
//     is updated before COMMAND_BUSY falls.
//   * "Stopped and idle" below means RUNNING = 0, COMMAND_BUSY = 0 and
//     WEIGHT_BUSY = 0. Staged weights can only be written, and committed,
//     then. While WEIGHT_BUSY is high the staged image is therefore frozen.
//   * WEIGHTS_VALID is set by the first commit acknowledgment and stays set
//     until the global reset; a later commit replaces the active image.
//
// APB3 (no PSTRB: every register is written as a whole word)
//   * pready is tied high. A write takes effect only in the access phase.
//   * pslverr, access phase only, for:
//       - an unaligned address (paddr[1:0] != 0) or an unmapped address,
//       - a write to STATUS, RESULT, SCOREn or ERROR_STATUS,
//       - COMMAND: a value other than 1, 2, 3; any command while
//         COMMAND_BUSY or WEIGHT_BUSY; START without WEIGHTS_VALID,
//       - RESULT_ACK with bit 0 set while RESULT_VALID is 0,
//       - WEIGHT_COMMIT with bit 0 set, or any WEIGHTn write, when not
//         stopped and idle.
//     A transfer that gets pslverr changes nothing.
//   * START while already running and STOP while already stopped are legal
//     and are still acknowledged. RESULT_ACK and WEIGHT_COMMIT written with
//     bit 0 clear do nothing and are not errors.
//   * Write-only registers read 0 without error. RESULT and SCOREn read 0
//     while RESULT_VALID is 0. An errored read returns 0.

module apb_slave #(
    parameter int N_OUT       = 4,      // number of class scores (max 4)
    parameter int SCORE_WIDTH = 32,     // must be <= 32
    parameter int WEIGHT_BITS = 864,    // size of the packed weight image
    parameter int ADDR_WIDTH  = 12      // width of paddr (byte address)
)(
    input  wire                  clk,     // pclk
    input  wire                  rst_n,   // pclk-domain global reset

    // APB3 slave interface
    input  wire                  psel,
    input  wire                  penable,
    input  wire                  pwrite,
    input  wire [ADDR_WIDTH-1:0] paddr,
    input  wire [31:0]           pwdata,
    output logic [31:0]          prdata,
    output wire                  pready,
    output logic                 pslverr,

    output wire                  irq,

    // Command mailbox, source side
    output logic                 cmd_send,       // one-cycle pulse
    output wire  [1:0]           cmd_code,       // held while cmd_busy
    input  wire                  cmd_busy,
    input  wire                  cmd_running,    // RUNNING from the last ack

    // Weight-commit mailbox, source side
    output logic                 wt_send,        // one-cycle pulse
    output wire  [WEIGHT_BITS-1:0] weight_image, // staged image, held while wt_busy
    input  wire                  wt_busy,
    input  wire                  wt_done,        // one-cycle pulse: committed

    // Result FIFO, read side. Entry = {score[N_OUT-1], ..., score[0], class}.
    input  wire                           fifo_empty,
    input  wire [N_OUT*SCORE_WIDTH+1:0]   fifo_rdata,
    output logic                          fifo_pop,

    input  wire                  overflow        // synchronized sticky flag
);

    localparam int WORD_ADDR_WIDTH  = ADDR_WIDTH - 2;
    localparam int N_WEIGHT_WORDS   = (WEIGHT_BITS + 31) / 32;   // 27
    localparam int WEIGHT_IDX_WIDTH =
        (N_WEIGHT_WORDS > 1) ? $clog2(N_WEIGHT_WORDS) : 1;
    localparam int ENTRY_WIDTH      = N_OUT*SCORE_WIDTH + 2;

    // Word addresses (byte address >> 2).
    localparam logic [WORD_ADDR_WIDTH-1:0] COMMAND_WORD =
        WORD_ADDR_WIDTH'('h000 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] STATUS_WORD =
        WORD_ADDR_WIDTH'('h004 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] IRQ_EN_WORD =
        WORD_ADDR_WIDTH'('h008 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] RESULT_ACK_WORD =
        WORD_ADDR_WIDTH'('h00C >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] RESULT_WORD =
        WORD_ADDR_WIDTH'('h010 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] FIRST_SCORE_WORD =
        WORD_ADDR_WIDTH'('h014 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] LAST_SCORE_WORD =
        WORD_ADDR_WIDTH'(('h014 >> 2) + N_OUT - 1);
    localparam logic [WORD_ADDR_WIDTH-1:0] WEIGHT_COMMIT_WORD =
        WORD_ADDR_WIDTH'('h024 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] ERROR_STATUS_WORD =
        WORD_ADDR_WIDTH'('h028 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] FIRST_WEIGHT_WORD =
        WORD_ADDR_WIDTH'('h100 >> 2);
    localparam logic [WORD_ADDR_WIDTH-1:0] LAST_WEIGHT_WORD =
        WORD_ADDR_WIDTH'(('h100 >> 2) + N_WEIGHT_WORDS - 1);

    // Command codes (same values as in bmi_pipeline_ctrl).
    localparam logic [31:0] CMD_START   = 32'd1;
    localparam logic [31:0] CMD_STOP    = 32'd2;
    localparam logic [31:0] CMD_RESTART = 32'd3;

    // Registered state
    logic                   irq_en_q, irq_en_d;
    logic [1:0]             cmd_code_q, cmd_code_d;
    logic                   weights_valid_q, weights_valid_d;
    logic                   result_valid_q, result_valid_d;
    logic [ENTRY_WIDTH-1:0] snap_q;                          // result snapshot
    logic [31:0]            weight_word [0:N_WEIGHT_WORDS-1]; // staged image

    // Decode and write enables (combinational)
    logic [WORD_ADDR_WIDTH-1:0]  word_addr;
    logic [WORD_ADDR_WIDTH-1:0]  weight_offset;
    logic [WEIGHT_IDX_WIDTH-1:0] weight_idx;
    logic sel_command, sel_status, sel_irq_en, sel_result_ack, sel_result;
    logic sel_score, sel_commit, sel_error, sel_weight, sel_any;
    logic aligned, access_phase, stopped_idle, cmd_legal;
    logic write_error, write_ok;
    logic wr_weight;

    always_comb begin
        // ---- address decode ----
        word_addr     = paddr[ADDR_WIDTH-1:2];
        weight_offset = word_addr - FIRST_WEIGHT_WORD;
        weight_idx    = weight_offset[WEIGHT_IDX_WIDTH-1:0];
        aligned       = (paddr[1:0] == 2'b00);

        sel_command    = (word_addr == COMMAND_WORD);
        sel_status     = (word_addr == STATUS_WORD);
        sel_irq_en     = (word_addr == IRQ_EN_WORD);
        sel_result_ack = (word_addr == RESULT_ACK_WORD);
        sel_result     = (word_addr == RESULT_WORD);
        sel_score      = (word_addr >= FIRST_SCORE_WORD) &&
                         (word_addr <= LAST_SCORE_WORD);
        sel_commit     = (word_addr == WEIGHT_COMMIT_WORD);
        sel_error      = (word_addr == ERROR_STATUS_WORD);
        sel_weight     = (word_addr >= FIRST_WEIGHT_WORD) &&
                         (word_addr <= LAST_WEIGHT_WORD);
        sel_any = sel_command || sel_status || sel_irq_en || sel_result_ack ||
                  sel_result || sel_score || sel_commit || sel_error ||
                  sel_weight;

        access_phase = psel && penable;
        stopped_idle = !cmd_running && !cmd_busy && !wt_busy;
        cmd_legal    = (pwdata == CMD_START) || (pwdata == CMD_STOP) ||
                       (pwdata == CMD_RESTART);

        // ---- which writes are refused ----
        write_error =
            sel_status || sel_result || sel_score || sel_error ||
            (sel_command && (!cmd_legal || cmd_busy || wt_busy ||
                             ((pwdata == CMD_START) && !weights_valid_q))) ||
            (sel_result_ack && pwdata[0] && !result_valid_q) ||
            (sel_commit && pwdata[0] && !stopped_idle) ||
            (sel_weight && !stopped_idle);

        pslverr  = access_phase &&
                   (!aligned || !sel_any || (pwrite && write_error));
        write_ok = access_phase && pwrite && !pslverr;

        // ---- next state ----
        irq_en_d        = irq_en_q;
        cmd_code_d      = cmd_code_q;
        weights_valid_d = weights_valid_q;
        result_valid_d  = result_valid_q;
        cmd_send        = 1'b0;
        wt_send         = 1'b0;
        wr_weight       = 1'b0;

        if (write_ok) begin
            if (sel_irq_en)
                irq_en_d = pwdata[0];
            if (sel_command) begin
                cmd_send   = 1'b1;
                cmd_code_d = pwdata[1:0];
            end
            if (sel_commit && pwdata[0])
                wt_send = 1'b1;
            if (sel_weight)
                wr_weight = 1'b1;
            if (sel_result_ack && pwdata[0])
                result_valid_d = 1'b0;
        end

        if (wt_done)
            weights_valid_d = 1'b1;

        // Pop and capture are the same event, so an entry can be neither
        // skipped nor loaded twice. Needs RESULT_VALID low, so it can never
        // coincide with an accepted RESULT_ACK.
        fifo_pop = !result_valid_q && !fifo_empty;
        if (fifo_pop)
            result_valid_d = 1'b1;

        // ---- read data ----
        prdata = 32'd0;
        if (psel && !pwrite && aligned) begin
            if (sel_status)
                prdata = {27'd0, cmd_running, weights_valid_q, wt_busy,
                          cmd_busy, result_valid_q};
            else if (sel_irq_en)
                prdata = {31'd0, irq_en_q};
            else if (sel_error)
                prdata = {31'd0, overflow};
            else if (sel_result) begin
                if (result_valid_q)
                    prdata = {30'd0, snap_q[1:0]};
            end else if (sel_score) begin
                if (result_valid_q)
                    for (int i = 0; i < N_OUT; i++)
                        if (word_addr == FIRST_SCORE_WORD + WORD_ADDR_WIDTH'(i))
                            prdata = 32'($signed(
                                snap_q[2 + i*SCORE_WIDTH +: SCORE_WIDTH]));
            end else if (sel_weight)
                prdata = weight_word[weight_idx];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_en_q        <= 1'b0;
            cmd_code_q      <= 2'd0;
            weights_valid_q <= 1'b0;
            result_valid_q  <= 1'b0;
            snap_q          <= '0;
            for (int i = 0; i < N_WEIGHT_WORDS; i++)
                weight_word[i] <= 32'd0;
        end else begin
            irq_en_q        <= irq_en_d;
            cmd_code_q      <= cmd_code_d;
            weights_valid_q <= weights_valid_d;
            result_valid_q  <= result_valid_d;
            // fifo_rdata is bundled data from the bmi_clk domain: stable
            // whenever fifo_empty is low (see cdc_async_fifo).
            if (fifo_pop)
                snap_q <= fifo_rdata;
            if (wr_weight)
                weight_word[weight_idx] <= pwdata;
        end
    end

    assign irq      = result_valid_q && irq_en_q;
    assign cmd_code = cmd_code_q;
    assign pready   = 1'b1;

    // Word n of the staged image supplies image bits [32n +: 32].
    wire [N_WEIGHT_WORDS*32-1:0] weight_flat;
    genvar w;
    generate
        for (w = 0; w < N_WEIGHT_WORDS; w = w + 1) begin : g_weight_flat
            assign weight_flat[w*32 +: 32] = weight_word[w];
        end
    endgenerate
    assign weight_image = weight_flat[WEIGHT_BITS-1:0];

    // Only the low bits of weight_offset index the weight array.
    wire unused_ok = &{1'b0,
                       weight_offset[WORD_ADDR_WIDTH-1:WEIGHT_IDX_WIDTH]};

`ifndef SYNTHESIS
    initial begin
        if (SCORE_WIDTH < 1 || SCORE_WIDTH > 32)
            $fatal(1, "SCORE_WIDTH must be 1..32 to fit a register");
        if (N_OUT < 1 || N_OUT > 4)
            $fatal(1, "N_OUT must be 1..4: SCOREn sits between RESULT and WEIGHT_COMMIT");
        if (WEIGHT_BITS < 1)
            $fatal(1, "WEIGHT_BITS must be at least one");
        if ((('h100 >> 2) + N_WEIGHT_WORDS) > (1 << WORD_ADDR_WIDTH))
            $fatal(1, "ADDR_WIDTH is too small for the weight registers");
    end
`endif

endmodule
