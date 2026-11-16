# APB interface and clock-domain crossing walkthrough

This explains how a host talks to the BMI classifier over APB, and how the
APB side (`pclk`) and the pipeline (`bmi_clk`) exchange commands, weights and
results when the two clocks are unrelated. Read it next to the RTL:

| File | Domain | What it is |
|---|---|---|
| [apb_slave.sv](../rtl/apb_slave.sv) | `pclk` | Registers, staged weights, result snapshot, interrupt |
| [bmi_pipeline_ctrl.sv](../rtl/bmi_pipeline_ctrl.sv) | `bmi_clk` | Executes commands, owns RUNNING, pushes results |
| [cdc_handshake.sv](../rtl/cdc_handshake.sv) | both | Request/acknowledge mailbox (commands, weight commit) |
| [cdc_async_fifo.sv](../rtl/cdc_async_fifo.sv) | both | Depth-4 dual-clock FIFO (results) |
| [cdc_sync_2ff.sv](../rtl/cdc_sync_2ff.sv) | destination | Two-flop synchronizer |
| [reset_synchronizer.sv](../rtl/reset_synchronizer.sv) | destination | Async-assert, sync-release reset |
| [bmi_chip_top.sv](../rtl/bmi_chip_top.sv) | both | Wiring only |

## 1. The picture

```
                 pclk domain                 |              bmi_clk domain
                                             |
 APB --> apb_slave                           |
          COMMAND ------ cmd_send, code ---> u_cmd_hs ---> bmi_pipeline_ctrl
          STATUS.RUNNING <--- RUNNING with the acknowledgment ---'   |  enable, clear
          WEIGHTn (staged) -- wt_send -----> u_weight_hs --> active weights --> MLP
          RESULT, SCOREn <--- snapshot <---- u_result_fifo <-- {scores, class} on decision
          ERROR_STATUS   <------------------ u_ovf_sync <----- sticky overflow
 irq <--  RESULT_VALID & IRQ_EN              |
                                             |   ADC -> frontend -> MLP -> argmax
```

Rule of the design: **only single bits and Gray-coded pointers go through
synchronizers. Everything wider crosses as "bundled data"**, which means it is
held still by its source and sampled by the destination only after a
synchronized single-bit signal says it is safe.

## 2. Register map

| Offset | Name | Access | Contents |
|---|---|---|---|
| `0x000` | `COMMAND` | WO | 1 = START, 2 = STOP, 3 = RESTART |
| `0x004` | `STATUS` | RO | [0] RESULT_VALID, [1] COMMAND_BUSY, [2] WEIGHT_BUSY, [3] WEIGHTS_VALID, [4] RUNNING |
| `0x008` | `IRQ_EN` | RW | [0] result interrupt enable |
| `0x00C` | `RESULT_ACK` | WO | write [0] = 1 to release the snapshot |
| `0x010` | `RESULT` | RO | [1:0] predicted class |
| `0x014`–`0x020` | `SCORE0`–`SCORE3` | RO | signed 32-bit scores |
| `0x024` | `WEIGHT_COMMIT` | WO | write [0] = 1 to commit the staged image |
| `0x028` | `ERROR_STATUS` | RO | [0] sticky result overflow |
| `0x100`–`0x168` | `WEIGHT0`–`WEIGHT26` | RW | staged image; byte `i` is bits `[8*(i%4) +: 8]` of word `i/4` |

APB3, `pready` always high, so every transfer is two `pclk` cycles. No `PSTRB`:
every register is a whole word.

What gets `pslverr` (and then changes nothing):

| Access | Refused when |
|---|---|
| Any | address not a multiple of 4, or not in the map |
| Write to `STATUS`, `RESULT`, `SCOREn`, `ERROR_STATUS` | always |
| Write `COMMAND` | value is not exactly 1, 2 or 3; or COMMAND_BUSY; or WEIGHT_BUSY; or START with WEIGHTS_VALID = 0 |
| Write `RESULT_ACK` with bit 0 set | RESULT_VALID = 0 |
| Write `WEIGHT_COMMIT` with bit 0 set, or any `WEIGHTn` | RUNNING = 1, or COMMAND_BUSY, or WEIGHT_BUSY |

Defined corner cases: write-only registers read 0 with no error. `RESULT` and
`SCOREn` read 0 while RESULT_VALID is 0. `RESULT_ACK` and `WEIGHT_COMMIT`
written with bit 0 clear do nothing and are not errors. START while running
and STOP while stopped are legal no-ops that are still acknowledged.

## 3. Software sequences

**Load weights and start**

1. Release `presetn`; wait at least two `pclk` cycles.
2. Write `WEIGHT0`..`WEIGHT26` (optionally read them back).
3. Write `WEIGHT_COMMIT = 1`. Poll `STATUS` until WEIGHT_BUSY = 0; WEIGHTS_VALID is now 1.
4. Write `IRQ_EN = 1` if interrupts are wanted.
5. Write `COMMAND = 1` (START). Poll `STATUS` until COMMAND_BUSY = 0; RUNNING is now 1.

**Read results** (repeat)

1. Wait for `irq`, or poll `STATUS` until RESULT_VALID = 1.
2. Read `RESULT` and `SCORE0`..`SCORE3` in any order, as slowly as you like.
   They are a frozen snapshot of one inference.
3. Write `RESULT_ACK = 1`. RESULT_VALID and `irq` drop for at least one `pclk`
   cycle. If another result is queued it is loaded and `irq` rises again.

The pipeline does not wait for you. If you fall more than five results behind
(one snapshot plus four FIFO entries), new results are dropped and
`ERROR_STATUS[0]` is set until the next global reset.

**Change the weights**

1. Write `COMMAND = 2` (STOP). Poll until COMMAND_BUSY = 0; RUNNING is now 0.
2. Write the new `WEIGHTn`, then `WEIGHT_COMMIT = 1`, poll WEIGHT_BUSY.
3. START again. Results queued before the STOP are still readable.

**Restart the current window**: write `COMMAND = 3` and poll COMMAND_BUSY.

## 4. APB timing

A write. The register changes on the edge that ends the access phase.

```
pclk     _/‾\_/‾\_/‾\_/‾\_
psel     ___/‾‾‾‾‾‾‾\_____
penable  _______/‾‾‾\_____
pwdata   ---<   D   >-----
                    ^ accepted here; pslverr is valid during the access phase
```

A read. `prdata` is combinational from the address, so it is valid through
both phases and is 0 whenever this slave is not selected.

```
pclk     _/‾\_/‾\_/‾\_/‾\_
psel     ___/‾‾‾‾‾‾‾\_____
penable  _______/‾‾‾\_____
prdata   000<   Q   >00000
```

## 5. The mailbox: `cdc_handshake`

Used twice: `u_cmd_hs` (2-bit command out, RUNNING back) and `u_weight_hs`
(864-bit image out, nothing back).

```
source (pclk)                              destination (bmi_clk)
src_send  _/‾\___________________________
req_q     ___/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\_________   level, through 2 flops ->
src_data  ===< held stable while busy >==
                                           req_sync  ______/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\____
                                           dst_valid ________/‾‾‾‾‾‾\___________  request waiting
                                           dst_done  ______________/‾\__________  executed
          <- level, through 2 flops        ack_q     ________________/‾‾‾‾‾‾‾‾\__
ack_sync  _____________________/‾‾‾‾‾‾\__
src_done  _______________________/‾\_____   response captured
src_busy  ___/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\__   busy until ack is back to 0
```

Why it is safe:

- **Nothing is lost.** `req_q` is a level, not a pulse. It stays high until
  the source has seen the acknowledgment, however slow or even stopped the
  other clock is.
- **Nothing runs twice.** `dst_valid` is set once per rising `req_sync` and
  cleared by `dst_done`. It cannot be set again until `ack_q` has fallen,
  which needs `req_sync` low first.
- **The data is stable when sampled.** `src_data` is launched no later than
  `req_q`. The destination copies it one cycle after `req_sync` rises, which
  is at least two `bmi_clk` periods later.
- **The response is consistent.** RUNNING comes back on the same handshake.
  `src_done` (which updates `STATUS.RUNNING`) comes before `src_busy` falls,
  so when software sees COMMAND_BUSY = 0, RUNNING is already final. There is
  no separately synchronized status bit that could arrive in a different
  order.
- **Reset is consistent.** Both ends reset to req = ack = 0. Levels, unlike
  toggles, cannot create a false event when one side leaves reset later than
  the other, so no flush delay is needed.

## 6. Commands: `bmi_pipeline_ctrl`

| Command | What the controller does | When it acknowledges |
|---|---|---|
| START | Sets RUNNING (only if weights are valid). The frontend starts a fresh window; the first capture is `CLK_DIV` clocks later. | Same cycle |
| STOP | Clears RUNNING at once, so no more samples are taken. An inference already in flight finishes and its result is queued. Then one `pipe_clear` cycle discards the partial window. | When nothing is in flight (up to about 125 clocks) |
| RESTART | One `pipe_clear` cycle: frontend, MLP and argmax return to their reset state. An inference in flight is discarded. RUNNING, the active weights and the FIFO are untouched. | Same cycle |

`pipe_clear` is a **synchronous** clear. The pipeline blocks gained a `clear`
input for it; the frontend also gained `enable` (= RUNNING). The asynchronous
reset of those blocks is now used only for the global reset.

During a weight commit all commands are refused (`pslverr`), and a commit is
refused while a command is busy, so the controller never sees both at once.

## 7. Weight commit

The 27 `WEIGHTn` registers are only a staging area. The MLP reads a second,
`bmi_clk`-domain copy: the destination register of `u_weight_hs`.

1. `WEIGHT_COMMIT` is accepted only when stopped and idle. From that edge
   WEIGHT_BUSY is high and every `WEIGHTn` write is refused, so the staged
   image is frozen.
2. `req` crosses. One cycle after it is seen, all 864 bits are copied into the
   active register on a single `bmi_clk` edge. No bit is synchronized on its
   own, so the MLP can never see a mix of old and new weights.
3. The acknowledgment returns; `WEIGHTS_VALID` is set and WEIGHT_BUSY falls.

## 8. Results: FIFO, snapshot, overflow

On `decision_valid` the class and the four scores are stable registers. The
controller pushes them as one 130-bit entry. The same `decision_valid` resumes
the frontend, which had paused so the MLP could read stable features. The
frontend therefore never waits for software.

`cdc_async_fifo` in brief:

- Each side keeps a binary pointer (for addressing) and a Gray copy (for
  crossing). Gray code changes one bit per step, so a pointer sampled in
  mid-change is read as either the old or the new value, never garbage.
- Pointers are 3 bits for 4 entries. Empty: read pointer equals the
  synchronized write pointer. Full: they differ in exactly the top two Gray
  bits (one full lap apart).
- The synchronized pointer is always a little old, which makes both flags
  err on the safe side: full may linger, empty may linger.
- `rdata` is the entry at the read pointer. It was written at least two `pclk`
  periods before `rempty` falls for it.

The snapshot in `apb_slave`:

```
fifo_pop = !result_valid && !fifo_empty      // pop and capture are one event
```

That one line gives all the required properties. Reads cannot pop (only this
does). The snapshot cannot change while valid (pop needs it invalid). An
entry cannot be skipped or loaded twice (the pointer moves on the same edge
the data is captured). After `RESULT_ACK` there is at least one cycle with
RESULT_VALID low before the next entry loads, so `irq` makes a new rising
edge for every result.

If the FIFO is full on `decision_valid` the result is dropped and `overflow`
latches. It only ever goes 0 to 1 (until reset), so a plain two-flop
synchronizer is enough to bring it to `pclk`.

## 9. Reset

`presetn` is the one global reset. Each domain has its own
`reset_synchronizer`: assertion is immediate, release happens on the second
clock edge of that domain. Global reset clears the registers, staged and
active weights, WEIGHTS_VALID, the snapshot, both FIFO pointers, both
mailboxes and the overflow flag. Nothing else ever resets a FIFO pointer or a
mailbox, so the two sides of a crossing cannot disagree.

## 10. Timing constraints the crossings need

A false path does not make a crossing safe; it only stops the tool looking.
[syn/bmi_chip_top.sdc](../syn/bmi_chip_top.sdc) instead limits every
`pclk`↔`bmi_clk` path to the shorter of the two clock periods. That covers:

| Crossing | Requirement |
|---|---|
| Into any `cdc_sync_2ff` | settle within one destination period |
| Mailbox data and response | data path and request path each under one destination period |
| FIFO `mem` to snapshot | under one `pclk` period |
| Gray pointer bits | skew between bits under one source period |

Synchronizer flops carry `ASYNC_REG`/`dont_touch` attributes; they should be
placed adjacent and excluded from retiming. None of this has been run through
Genus or a CDC tool yet.

## 11. Decisions and the alternatives rejected

| Decision | Rejected alternative | Why |
|---|---|---|
| Four-phase level handshake | Toggle (two-phase) pulse synchronizer, as in the previous version | A toggle makes a false event when one side is reset alone, which needed a fixed flush delay. Levels do not. |
| RUNNING returned with the acknowledgment | Separate synchronizer for RUNNING | Two paths could arrive in either order; software could see "not busy" with a stale RUNNING. |
| Second, `bmi_clk`-domain weight register | MLP reads the APB registers directly | Costs 864 flops, but the active image changes on one edge and never during an inference. |
| FIFO plus snapshot | One result register and a paused frontend | The pipeline no longer depends on software timing; a slow host loses data visibly (overflow) instead of stalling sampling. |
| Frontend pauses during inference, resumed by `decision_valid` | Free-running frontend | The MLP reads the feature registers for 64 cycles; keeping them still is simpler than copying them. Costs about 124 clocks per window. |
| STOP lets an in-flight inference finish | STOP aborts it | A completed window is real data; RESTART exists for the abort case. |
| START always begins a fresh window | Resume a half-collected window | Samples from before and after a gap must not be mixed. |
| `RESULT_ACK` with nothing valid is an error | Silently ignore | A double acknowledge is a software bug worth reporting. |
| Unaligned address is an error | Ignore `paddr[1:0]` (previous version) | Requested; also catches byte-addressing mistakes. |
| Overflow cleared only by global reset | Software-clearable flag | Requested for this version; a clear would need its own crossing. |

## 12. Assumptions to confirm at integration

- `presetn` may be asserted at any time; no APB transfer starts until two
  `pclk` edges after it is released.
- `bmi_clk` may be stopped: a pending command or commit then waits (BUSY stays
  high) and completes exactly once when the clock returns. `pclk` must run for
  APB to work at all.
- `adc_sample` is synchronous to `bmi_clk`.
- The `pclk` period in the SDC is a placeholder.
- The window period is now `8000 + 124` `bmi_clk` cycles (collection plus the
  pause for inference), independent of software.

## 13. Where each behaviour is tested

| Behaviour | Bench |
|---|---|
| Mailbox: exactly once, data, response, stopped clocks, resets | [tb_cdc_handshake.sv](../sim/tb_cdc_handshake.sv) |
| FIFO: order, full, empty, read-data stability, resets | [tb_cdc_async_fifo.sv](../sim/tb_cdc_async_fifo.sv) |
| Controller: commands, STOP wait, clear coinciding with a result, overflow | [tb_bmi_pipeline_ctrl.sv](../sim/tb_bmi_pipeline_ctrl.sv) |
| Register map, every error response, no side effects, snapshot, interrupt | [tb_apb_slave.sv](../sim/tb_apb_slave.sv) |
| Golden vectors end to end, commands, STOP/RESTART, recommit, overflow, ack races, global reset | [tb_bmi_chip_top.sv](../sim/tb_bmi_chip_top.sv) |
| Ports-only run, also on a synthesized netlist | [tb_gls.sv](../sim/tb_gls.sv) |
