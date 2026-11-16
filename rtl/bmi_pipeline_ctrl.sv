// bmi_pipeline_ctrl.sv
// Pipeline controller, entirely in the bmi_clk domain. It executes the
// commands delivered by the command mailbox, owns the RUNNING state, marks
// the active weight image valid, and hands each classification result to the
// result FIFO.
//
// Commands (cmd_code is stable while cmd_valid is high)
//   START    RUNNING <- 1 if a weight image has been committed; otherwise no
//            effect. Collection begins with a fresh window.
//   STOP     RUNNING <- 0 at once, so the frontend takes no more samples. An
//            inference already in flight is allowed to finish and its result
//            is queued. cmd_done is given only when nothing is in flight; the
//            partial window is then discarded with one pipe_clear cycle.
//   RESTART  One pipe_clear cycle: the frontend, MLP and argmax return to
//            their reset state and any inference in flight is discarded.
//            RUNNING, the active weights and the result FIFO are untouched.
//   cmd_running is the value RUNNING has after the command; it travels back
//   with the acknowledgment, so the APB side never has to guess the order of
//   "command finished" and "status changed".
//
// Results: on decision_valid the class and all scores, which are stable
// registers at that point, are pushed as one FIFO entry. If the FIFO is full
// the result is dropped and overflow is set until the global reset. The
// frontend, which pauses while an inference runs, is resumed by the same
// decision_valid; it never waits for software.

module bmi_pipeline_ctrl #(
    parameter int N_OUT       = 4,
    parameter int SCORE_WIDTH = 32
)(
    input  wire  clk,                 // bmi_clk
    input  wire  rst_n,               // bmi_clk-domain global reset

    // Command mailbox, destination side
    input  wire        cmd_valid,
    input  wire [1:0]  cmd_code,
    output logic       cmd_done,
    output logic       cmd_running,

    // Weight-commit mailbox, destination side. The image itself is the
    // mailbox's registered data; this block only tracks that it is valid.
    input  wire        wt_valid,
    output wire        wt_done,
    output wire        weights_valid,

    // Pipeline
    output wire        frontend_enable,   // level: take samples
    output logic       pipe_clear,        // one-cycle synchronous clear
    input  wire        features_done,     // frontend finished a window
    input  wire        decision_valid,    // argmax produced a result
    input  wire [1:0]  predicted_class,
    input  wire signed [SCORE_WIDTH-1:0] class_scores [0:N_OUT-1],

    // Result FIFO, write side. Entry = {score[N_OUT-1], ..., score[0], class}.
    output logic                          fifo_push,
    output wire [N_OUT*SCORE_WIDTH+1:0]   fifo_wdata,
    input  wire                           fifo_full,

    output wire        overflow           // sticky: a result was dropped
);

    // Command codes (same values as the COMMAND register in apb_slave).
    localparam logic [1:0] CMD_START   = 2'd1;
    localparam logic [1:0] CMD_STOP    = 2'd2;
    localparam logic [1:0] CMD_RESTART = 2'd3;

    typedef enum logic {ST_IDLE, ST_STOP_WAIT} state_t;

    state_t state_q, state_d;
    logic   running_q, running_d;
    logic   weights_valid_q, weights_valid_d;
    logic   infer_busy_q, infer_busy_d;     // from features_done to decision_valid
    logic   overflow_q, overflow_d;

    always_comb begin
        state_d         = state_q;
        running_d       = running_q;
        weights_valid_d = weights_valid_q;
        infer_busy_d    = infer_busy_q;
        overflow_d      = overflow_q;
        cmd_done        = 1'b0;
        pipe_clear      = 1'b0;
        fifo_push       = 1'b0;

        // The APB side only commits while stopped and with no command pending.
        if (wt_valid)
            weights_valid_d = 1'b1;

        if (features_done)
            infer_busy_d = 1'b1;
        if (decision_valid)
            infer_busy_d = 1'b0;

        unique case (state_q)
            ST_IDLE: begin
                if (cmd_valid) begin
                    unique case (cmd_code)
                        CMD_START: begin
                            if (weights_valid_q)
                                running_d = 1'b1;
                            cmd_done = 1'b1;
                        end
                        CMD_STOP: begin
                            running_d = 1'b0;
                            state_d   = ST_STOP_WAIT;
                        end
                        CMD_RESTART: begin
                            pipe_clear = 1'b1;
                            cmd_done   = 1'b1;
                        end
                        default: begin
                            cmd_done = 1'b1;    // never sent by apb_slave
                        end
                    endcase
                end
            end

            ST_STOP_WAIT: begin
                // RUNNING has been low since the first cycle of this state,
                // so no new window can complete; wait out the one in flight.
                if (!features_done && !infer_busy_q) begin
                    pipe_clear = 1'b1;
                    cmd_done   = 1'b1;
                    state_d    = ST_IDLE;
                end
            end
        endcase

        // A result is queued unless a RESTART discards it in the same cycle.
        if (pipe_clear) begin
            infer_busy_d = 1'b0;
        end else if (decision_valid) begin
            if (fifo_full)
                overflow_d = 1'b1;
            else
                fifo_push = 1'b1;
        end

        cmd_running = running_d;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q         <= ST_IDLE;
            running_q       <= 1'b0;
            weights_valid_q <= 1'b0;
            infer_busy_q    <= 1'b0;
            overflow_q      <= 1'b0;
        end else begin
            state_q         <= state_d;
            running_q       <= running_d;
            weights_valid_q <= weights_valid_d;
            infer_busy_q    <= infer_busy_d;
            overflow_q      <= overflow_d;
        end
    end

    assign wt_done         = wt_valid;
    assign weights_valid   = weights_valid_q;
    assign frontend_enable = running_q;
    assign overflow        = overflow_q;

    // One coherent entry: class in the low two bits, score i above it.
    assign fifo_wdata[1:0] = predicted_class;
    genvar s;
    generate
        for (s = 0; s < N_OUT; s = s + 1) begin : g_pack_score
            assign fifo_wdata[2 + s*SCORE_WIDTH +: SCORE_WIDTH] = class_scores[s];
        end
    endgenerate

endmodule
