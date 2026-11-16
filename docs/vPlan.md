# BMI ASIC Digital Verification Plan

## 1. Purpose

This verification plan defines how the BMI digital classifier is tested and how verification completion is measured. It is a living sign-off document: an entry is marked **Implemented** only when an automated checker exists and the named regression passes.

The plan distinguishes two different results:

- **RTL equivalence:** RTL outputs match the integer Python golden model.
- **Model accuracy:** the predicted class matches the recorded biological label.

The RTL regression establishes equivalence, not model accuracy.

## 2. Design Under Test

The top-level DUT is `bmi_chip_top` with the default configuration:

| Parameter | Value |
|---|---:|
| ADC channels | 8 |
| Samples per channel | 250 |
| ADC width | 8 bits |
| SBP features | 8 |
| Hidden neurons | 8 |
| Output classes | 4 |
| Weight width | Signed 8 bits |
| Accumulator/score width | Signed 32 bits |
| Host interface | APB3 register block, 12-bit address, on its own clock `pclk` |
| Result buffering | One snapshot plus a depth-4 dual-clock FIFO |
| Clocks | `bmi_clk` (pipeline, about 10 MHz) and `pclk` (APB), unrelated; one global reset `presetn` |
| Weight image | 864 bits (108 bytes), staged as 27 APB words, then committed to the pipeline domain |

The verified datapath is:

```text
ADC collection → SBP extraction → 8→8→4 MLP → argmax → result FIFO
               → APB register block (result snapshot, interrupt)
```

`apb_slave` runs on the APB clock `pclk`; the frontend, MLP, argmax, the
active weights, and `bmi_pipeline_ctrl` run on the pipeline clock `bmi_clk`.
The two clocks are unrelated. Commands (START, STOP, RESTART) and the weight
commit cross through acknowledged handshakes (`cdc_handshake`), results cross
through a dual-clock FIFO (`cdc_async_fifo`), and the pipeline runs without
waiting for software. The design is explained in
[apb_walkthrough.md](apb_walkthrough.md).

The APB register block (`apb_slave`) is the only host interface. It replaced
three blocks of the previous interface, which are kept in `rtl/` and are still
unit-tested but are commented out in `bmi_chip_top`, so the integration and
gate-level regressions no longer cover them:

| Previous block | Was used for | Still verified by |
|---|---|---|
| `weight_scan_loader` | Serial weight loading on `scan_clk` (split out of `mlp_inference`, behaviour unchanged) | `tb_mlp_inference.sv` |
| `output_formatter` | 80-bit result packet | `tb_output_formatter.sv` |
| `spi_slave` | SPI Mode 0 transmitter | `tb_spi_slave.sv` |

## 3. Verification Methodology

### 3.1 Verification levels

| Level | Purpose | Environment |
|---|---|---|
| Block | Exercise arithmetic, control, reset, and corner cases in isolation | Dedicated SystemVerilog testbench per block |
| Integration | Prove end-to-end dataflow and handshakes using generated vectors | `sim/tb_bmi_chip_top.sv` |
| Gate-level | Confirm synthesized netlist behavior | `sim/tb_gls.sv` |
| Static | Find width, signedness, latch, CDC, reset, and constraint problems | Lint, CDC review, synthesis reports |

### 3.2 Reference models and stimulus

- ADC stimulus comes from `sim/vectors/vecNN_adc.hex`.
- Expected SBP values come from `sim/vectors/vecNN_sbp.hex`.
- Expected MLP scores come from `sim/vectors/vecNN_scores.hex`.
- Expected classes come from `sim/vectors/all_expected.hex`.
- Model parameters come from `sim/weights.hex`. The integration and gate-level benches write them into the `WEIGHTn` staging registers over APB, read every word back, and commit them.
- The Python integer pipeline is the arithmetic reference model.

The 40 generated vectors are useful regression stimulus, but they are not sufficient for corner-case closure. Directed block tests and assertions are required in addition to vector replay.

### 3.3 Checking strategy

The integration bench checks each vector at multiple observation points:

1. Eight SBP features against the Python golden values, at `features_done`.
2. Four complete signed MLP scores against the golden values, at the decision.
3. Golden scores fit within the RTL's signed 32-bit score format.
4. Reserved bits of the `RESULT` register are zero.
5. Predicted class, read from `RESULT`.
6. All four full 32-bit scores, read from `SCORE0`–`SCORE3`.
7. APB response on every transfer (`pslverr` as expected, `pready` high), and `STATUS` and `irq` around each result.
8. The active weight image seen by the MLP matches `weights.hex` byte for byte after the commit.
9. Everything the host reads matches, in order, what the pipeline pushed into the result FIFO (scoreboard on the `bmi_clk` side).
10. Overall regression failure through `$fatal` if any check fails.

Four-state comparisons (`===`/`!==`) are used so unknown or high-impedance values cannot silently pass.

### 3.4 Checker changes made for the two-clock architecture

The golden files and every expected value in them are unchanged. The benches
below had to change because the behaviour they checked was replaced on
purpose; none was changed to make a failing design pass.

| Bench / checker | Change | Why |
|---|---|---|
| `tb_streaming_sbp_frontend.sv`, `tb_mlp_inference.sv`, `tb_argmax.sv` | New `enable`/`clear` ports tied to 1/0 in the instantiation | The pipeline blocks gained a synchronous clear (and the frontend an enable). No checker or expected value changed; all tests pass as before |
| `tb_apb_slave.sv` | Rewritten for the new register map | `CTRL`, `IRQ_STATUS` (W1C) and the soft-reset bit no longer exist. Reset values, weight read-back and byte order, blocked weight writes, result read-out, interrupt masking, error responses, access-phase qualification, back-to-back transfers, and reset mid-transfer are all still tested against the new map |
| `tb_apb_slave.sv`: unaligned addresses | Used to expect the low two address bits to be ignored; now expects `pslverr` | Requested behaviour change |
| `tb_apb_pipeline_cdc.sv` | Removed with the module it tested | The toggle synchronizers and the flush window were replaced by `cdc_handshake` and `cdc_async_fifo`, which have their own benches |
| Top level: "frontend stays paused until the host clears the result" (old TOP-003, TOP-004) | Inverted: the frontend must resume at channel 0 the cycle after the decision | The pipeline now runs independently of software. Result stability is checked on the snapshot instead (TOP-004) |
| Top level: decision-to-result-flag latency bound | Removed | Results pass through a FIFO; order and completeness are checked by the scoreboard instead. The two latencies inside the pipeline are still checked exactly |
| Top level: SBP feature and internal score compares | Moved from the host thread to the `bmi_clk` monitor, at `features_done` and `decision_valid` | The host is no longer in step with the pipeline, so those registers may already hold the next window when the host reads a result. Same golden values |
| Top level: capture-spacing monitor | Interval restarts on enable, clear, and the inference pause instead of on a pipeline reset | The frontend is no longer reset to start and stop it. Expected spacing (`CLK_DIV`) unchanged |
| Top level: event totals | "40 events at every boundary" replaced by "every window was queued, dropped, or discarded" and "commands accepted over APB = commands executed" | The run now includes stops, restarts, and a deliberate overflow |
| `tb_gls.sv` | New host sequence; the ADC model follows `adc_channel` only | The bench cannot know when a window ends from the bus any more |

## 4. Verification Entries

Status values:

- **Implemented:** automated checking exists and currently passes.
- **Partial:** some behavior is exercised, but coverage or checking is incomplete.
- **Planned:** no adequate automated check exists yet.

### 4.1 Streaming sample collection

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| SC-001 | Cycle through channels 0–7 in order | Four complete directed windows | Check every `adc_channel` selection | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SC-002 | Accept 250 samples for every channel | Directed windows and 40 generated ADC vectors | Reference sums plus end-to-end equivalence | Implemented (`tb_streaming_sbp_frontend.sv`, `tb_bmi_chip_top.sv`) |
| SC-003 | Pulse `features_done` on the final capture | Complete directed window | Check final-edge assertion and one-cycle pulse | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SC-004 | Pause accumulation and hold features until resume | Delay resume for multiple clocks | Check channel, pulse, and feature stability | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SC-005 | Resume at channel/sample zero after `resume` (the decision, in the full chip) | Back-to-back directed and generated windows | Check complete consecutive windows | Implemented (`tb_streaming_sbp_frontend.sv`, `tb_bmi_chip_top.sv`) |
| SC-006 | Reset clears state, counters, accumulators, and features | Reset at startup and mid-collection | Directed asynchronous reset test | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SC-007 | Counter and accumulator widths support legal terminal values | Parameter review and rail patterns | Elaboration checks and maximum-sum comparison | Implemented (`streaming_sbp_frontend.sv`, `tb_streaming_sbp_frontend.sv`) |
| SC-008 | Capture once every `ceil(CLK_FREQ_HZ / ADC_SAMPLE_FREQ_HZ)` clocks; hold `adc_channel` and ignore `adc_sample` between captures | Corrupted ADC data on non-capture cycles; `ADC_SAMPLE_FREQ_HZ` swept over integer and non-integer ratios (divide by 1–5) | Feature mismatch on any off-cycle capture; channel-hold check; top-level capture-spacing monitor | Implemented (`tb_streaming_sbp_frontend.sv`, `tb_bmi_chip_top.sv`) |

### 4.2 SBP feature extraction

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| SBP-001 | Calculate `sum(abs(sample-128)) >> 8` while samples arrive | Directed patterns and 40 generated vectors | Compare all 8 results against independent references | Implemented (`tb_streaming_sbp_frontend.sv`, `tb_bmi_chip_top.sv`) |
| SBP-002 | All samples equal 128 produce zero | Directed constant window | Check all outputs equal zero | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SBP-003 | ADC code 0 exercises maximum deviation 128 | Directed channel at all-zero rail | Check expected result 125 | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SBP-004 | ADC code 255 exercises positive rail deviation 127 | Directed channel at all-255 rail | Check expected result 124 | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SBP-005 | Accumulator includes the final sample | Deviations at indices 0 and 249 | Check exact shifted result of one | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SBP-006 | `features_done` is a one-cycle pulse at expected capture count | Complete directed windows | Check pulse on capture 2000 and low afterward | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SBP-007 | Samples presented while paused do not alter completed features | Change ADC input while delaying resume | Check feature and control stability | Implemented (`tb_streaming_sbp_frontend.sv`) |

### 4.3 Weight loading and MLP inference

`mlp_inference` now takes the packed weight image and a `weights_loaded` level
as inputs. `tb_mlp_inference.sv` instantiates it together with
`weight_scan_loader`, wired as they were when the loader was internal, so the
scan-loading entries below are tested exactly as before. In the integration
bench the same image is written to the APB staging registers and committed
to the pipeline domain through the weight mailbox instead.

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| MLP-001 | Load all 864 parameter bits in the documented order | Synthetic unit models through the scan loader; production `weights.hex` through the APB staging registers and commit | Check scan image, unpacked arrays, APB read-back, the image at the MLP input, and end-to-end scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-002 | Hidden MAC uses all 8 features and signed weights | Directed signed weights, unsigned boundary inputs, and 40 generated vectors | Compare hidden activations and four final scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-003 | Add scaled hidden bias | Directed positive and negative hidden biases | Procedural reference-model comparison | Implemented (`tb_mlp_inference.sv`) |
| MLP-004 | Clamp negative hidden pre-activations to zero | Directed positive and negative pre-activations | Check every internal hidden activation | Implemented (`tb_mlp_inference.sv`) |
| MLP-005 | Output MAC uses all 8 hidden activations | Directed output connectivity and 40 generated vectors | Compare all four full scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-006 | Add scaled output bias without output ReLU | Negative, zero, and positive bias-only scores | Compare signed scores | Implemented (`tb_mlp_inference.sv`) |
| MLP-007 | Produce four valid scores and one-cycle `done` | Repeated inference transactions | Check derived latency, all scores, pulse width, and score stability | Implemented (`tb_mlp_inference.sv`) |
| MLP-008 | Arithmetic does not overflow configured widths | Positive/negative rail models with maximum unsigned inputs | Elaboration range checks plus reference-model comparison | Implemented (`mlp_inference.sv`, `tb_mlp_inference.sv`) |
| MLP-009 | Ignore `start` while busy | Reassert start during hidden MAC | Check original transaction result and latency are unchanged | Implemented (`tb_mlp_inference.sv`) |
| MLP-010 | Inference cannot start with partial weights | Shift one bit fewer than the complete image and pulse `start` | Check `weights_loaded` remains low and FSM remains idle | Implemented (`mlp_inference.sv`, `tb_mlp_inference.sv`) |

For the production 8/8/4 configuration, unsigned features span 0–255 and
signed weights/biases span −128–127. Including the configured bias scales:

```text
hidden maximum = 8(255)(127) + 127(228) =  288,036
hidden minimum = 8(255)(-128) - 128(228) = -290,304

output maximum = 8(288,036)(127) + 127(193) =  292,669,087
output minimum = 8(288,036)(-128) - 128(193) = -294,973,568
```

The hidden range requires 20 signed bits and the output range requires 30
signed bits. Therefore, the configured 32-bit accumulator and score outputs
cannot overflow for any legal 8-bit feature, weight, or bias value. Elaboration
checks in `mlp_inference.sv` reject parameter combinations that do not fit.

### 4.4 Argmax

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| ARG-001 | Select each of classes 0–3 when uniquely largest | Four directed cases | Check class and valid pulse | Implemented (`tb_argmax.sv`) |
| ARG-002 | Compare scores as signed values | All-negative and signed-boundary cases | Check expected winner | Implemented (`tb_argmax.sv`) |
| ARG-003 | Resolve ties toward the lowest index | Pair and four-way ties | Check documented tie policy | Implemented (`tb_argmax.sv`) |
| ARG-004 | Pulse `decision_valid` for one cycle | Valid input pulse | Check pulse returns low | Implemented (`tb_argmax.sv`) |
| ARG-005 | Hold previous class when scores are invalid | Change scores with valid low | Check output stability | Implemented (`tb_argmax.sv`) |
| ARG-006 | End-to-end predicted class matches golden model | 40 generated vectors | Class comparison on the `RESULT` register read over APB | Implemented (`tb_bmi_chip_top.sv`) |

### 4.5 Packet formatter (retained, not instantiated)

`output_formatter` is no longer part of `bmi_chip_top`. FMT-001 to FMT-004 used
to be checked on the 40 generated vectors through the SPI packet; they are now
covered only by the directed unit test, which compares the complete packet
(sync byte, reserved bits, class, and all four score slices) against a
reference model for 13 cases.

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| FMT-001 | Byte 0 is `0xAA` | 13 directed cases | Complete-packet comparison | Implemented (`tb_output_formatter.sv`) |
| FMT-002 | Class byte upper six bits are zero | 13 directed cases | Complete-packet comparison | Implemented (`tb_output_formatter.sv`) |
| FMT-003 | Class byte lower two bits contain prediction | 13 directed cases | Complete-packet comparison | Implemented (`tb_output_formatter.sv`) |
| FMT-004 | Pack score `[31:16]` fields in class order | 13 directed cases | Complete-packet comparison | Implemented (`tb_output_formatter.sv`) |
| FMT-005 | Hold packet stable while valid and not ready | Delay ready while changing formatter inputs | Check packet and valid over multiple cycles | Implemented (`tb_output_formatter.sv`) |
| FMT-006 | Clear valid on ready | Pulse ready, including simultaneous new decision | Check clear and documented ready priority | Implemented (`tb_output_formatter.sv`) |

### 4.6 APB register block

`apb_slave` is an APB3 slave entirely in the `pclk` domain. `tb_apb_slave.sv`
tests it alone, with small models standing in for the two mailboxes and the
FIFO. Every errored transfer is checked for "no side effects" by comparing
all register state and the model counters before and after it.

| Offset | Name | Access | Contents |
|---|---|---|---|
| `0x000` | `COMMAND` | WO | 1 = START, 2 = STOP, 3 = RESTART |
| `0x004` | `STATUS` | RO | [0] RESULT_VALID, [1] COMMAND_BUSY, [2] WEIGHT_BUSY, [3] WEIGHTS_VALID, [4] RUNNING |
| `0x008` | `IRQ_EN` | RW | [0] result interrupt enable |
| `0x00C` | `RESULT_ACK` | WO | write [0] = 1 to release the snapshot |
| `0x010` | `RESULT` | RO | predicted class in bits `[1:0]` |
| `0x014`–`0x020` | `SCORE0`–`SCORE3` | RO | signed 32-bit scores |
| `0x024` | `WEIGHT_COMMIT` | WO | write [0] = 1 to commit the staged image |
| `0x028` | `ERROR_STATUS` | RO | [0] sticky result overflow |
| `0x100`–`0x168` | `WEIGHT0`–`WEIGHT26` | RW | staged weight image, four bytes per word, little-endian, in `weights.hex` order |

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| APB-001 | Reset values | Read every register after reset | All zero; `irq`, `cmd_send`, `wt_send` low | Implemented (`tb_apb_slave.sv`) |
| APB-002 | Staged weights read back; byte order matches the image | Write and read all 27 words from `weights.hex`; rewrite one word | Read-back, `weight_image` bytes, neighbours untouched | Implemented |
| APB-003 | START refused before a commit; STOP and RESTART legal | Commands with `WEIGHTS_VALID` = 0 | `pslverr` and no `cmd_send` for START; one `cmd_send` each for the others | Implemented |
| APB-004 | Commit sends once, freezes the staged image, sets `WEIGHTS_VALID` at the acknowledgment | Commit with a long mailbox latency; weight writes, a second commit, and commands while busy | One `wt_send`; all the others refused with no effect; image stable while busy | Implemented |
| APB-005 | Commands: one send each, `COMMAND_BUSY`, `RUNNING`, invalid codes | START, RESTART, STOP, repeats, codes 0, 4, `0x101`, `0xFFFFFFFF`, `0x80000002` | Send count, code held while busy, `STATUS` | Implemented |
| APB-006 | Staged weights and commit locked while running or while a command is busy | Weight and commit writes in each state | `pslverr`, no effect | Implemented |
| APB-007 | Snapshot: loaded once, held until `RESULT_ACK`, reads never pop, entries in order | Three queued results, repeated reads, acknowledgments, an entry arriving in the acknowledge cycle | Pop count, register values, background stability monitor | Implemented |
| APB-008 | `RESULT_ACK` with bit 0 clear does nothing; with nothing valid it is an error | Both cases, including a double acknowledge | Snapshot unchanged; `pslverr` | Implemented |
| APB-009 | `irq` = RESULT_VALID & IRQ_EN; new rising edge per result; polling works with the interrupt masked | Mask, unmask, two queued results | `irq` level and rising-edge count | Implemented |
| APB-010 | Error responses: unmapped, unaligned, read-only writes; write-only registers read 0 | 30 errored transfers with a result held | `pslverr`, `prdata` = 0, no state change | Implemented |
| APB-011 | Writes need a completed access phase | Setup phase abandoned; `penable` without `psel` | No register change, no send, no `pslverr` | Implemented |
| APB-012 | Back-to-back transfers; a command or commit directly followed by another | No idle cycle between transfers | Second one refused, first sent once | Implemented |
| APB-013 | `ERROR_STATUS` reflects the overflow input and is read-only | Drive the flag | Read value; write refused | Implemented |
| APB-014 | Reset in the middle of a transfer | Assert reset in the access phase | Everything back to reset values; START refused again | Implemented |
| APB-015 | Protocol invariants | Whole run | `pslverr` only in the access phase, `prdata` 0 when unselected, `pready` high, no X | Implemented (background monitor) |

### 4.6.1 SPI transmitter (retained, not instantiated)

These entries apply to `spi_slave.sv` on its own. The block is no longer part
of `bmi_chip_top`, so SPI is covered by its unit test only.

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| SPI-001 | Transmit 80 bits MSB-first in Mode 0 | Multiple non-symmetric packets | Compare complete externally sampled packet | Implemented (`tb_spi_slave.sv`) |
| SPI-002 | Present first MISO bit before first sample edge | Check MISO after CS synchronization and before rising SCLK | Compare against packet bit 79 | Implemented (`tb_spi_slave.sv`) |
| SPI-003 | Pulse `packet_ready` after exactly 80 sampled bits | Monitor all 80 master sample edges | Reject early pulse; require exactly one one-cycle pulse after bit 80 | Implemented (`tb_spi_slave.sv`) |
| SPI-004 | Abort safely when CS deasserts early | Transfers of 1, 17, and 79 bits | Return MISO low; require no ready pulse | Implemented (`tb_spi_slave.sv`) |
| SPI-005 | Ignore clocks while CS is inactive | Toggle SCLK for 100 edges with CS high | Check MISO and ready remain inactive | Implemented (`tb_spi_slave.sv`) |
| SPI-006 | Support `f_clk >= 8 × f_spi` | Sweep ratios 8, 10, and 12 with different phases | Compare complete packets and completion pulses | Implemented (`tb_spi_slave.sv`) |
| SPI-007 | Reset during transfer returns interface to idle | Assert asynchronous reset after 23 bits | Check MISO, ready, abort, and subsequent recovery | Implemented (`tb_spi_slave.sv`) |

### 4.7 Clock-domain crossing

Every crossing is inside a `cdc_*` instance. The two reusable blocks have
their own benches, each run at five clock relationships (ratios of 14:1 and
90:1 in both directions, equal periods with a phase offset, and nearly equal
periods so the phase drifts through every alignment). Both use real
`reset_synchronizer` instances for their resets.

**Limit of simulation.** RTL simulation cannot model metastability. These
entries prove the protocols: nothing lost, duplicated, reordered, or torn.
That every crossing goes through a synchronizer, and that nothing else
crosses, is a structural property that needs a CDC tool (STA-003).

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| CDC-001 | Mailbox idle out of reset | Release reset | No busy, valid, or done | Implemented (`tb_cdc_handshake.sv`) |
| CDC-002 | Each request executed exactly once with the right data; the right response returns before busy falls | 200 random requests, destination answers after 0 to 5 cycles | Scoreboard: sent = executed = done = `dst_valid` rises; data and response compare | Implemented |
| CDC-003 | Back-to-back requests | 50 requests sent in the first idle cycle | Same scoreboard | Implemented |
| CDC-004 | A send while busy is ignored | Random extra `src_send` pulses during 50 transfers | Counts unchanged | Implemented |
| CDC-005 | Destination clock stopped: the request waits and then runs once | Stop the clock before and during 10 transfers | Busy held; counts | Implemented |
| CDC-006 | Source clock stopped: the destination still executes; completes once when the clock returns | 10 transfers | Counts | Implemented |
| CDC-007 | Global reset mid-transfer leaves both sides idle with no leftover event | Reset at 12 points through a transfer | No activity afterwards; next transfer normal | Implemented |
| CDC-008 | A request raised while the destination is still in reset is not lost | Hold the destination clock through the reset release | Executed once after release | Implemented |
| CDC-009 | FIFO empty and not full out of reset | Release reset | Flags | Implemented (`tb_cdc_async_fifo.sv`) |
| CDC-010 | Fill to depth, writes beyond full dropped, drain in order, reads beyond empty ignored | 10 writes, 10 reads | Accepted counts, flags, data | Implemented |
| CDC-011 | An entry is presented unchanged until popped, and popped once | 40 single entries with varying hold times | `rdata` value and stability monitor | Implemented |
| CDC-012 | No loss, duplication, or reordering under random traffic, writer-saturated, and reader-saturated | 3000 + 750 + 750 entries | Sequence numbers; occupancy never above 4 or below 0 | Implemented |
| CDC-013 | Global reset with entries queued empties the FIFO | Reset at 6 points with traffic running | Flags, then 20 clean entries | Implemented |
| CDC-014 | Whole chip with unrelated clocks | Top-level regression at `pclk` 3, 7, 9.7, 10 (two phases), 27, and 130 ns against a 10 ns `bmi_clk` | All TOP entries | Implemented (`tb_bmi_chip_top.sv`) |

**Mutation run.** The benches were run against 46 deliberately broken copies
of the RTL (9 in `cdc_handshake`, 8 in `cdc_async_fifo`, 16 in `apb_slave`, 7
in `bmi_pipeline_ctrl`, 4 in the top-level wiring, one each in the frontend
and argmax). 44 were detected. The first run detected 41 and exposed three
missing tests: STOP leaving a partial window in place, a result pushed in the
same cycle as a clear, and argmax ignoring the clear. Those coincidences
depend on synchronizer timing at the top level, so `tb_bmi_pipeline_ctrl.sv`
(CTL-001 to CTL-008) was added to force them, and a mid-window STOP check was
added to TOP-012. The two that remain undetected cannot be seen in RTL
simulation, because they only remove a synchronizer: accepting a request
before it has been synchronized, and wiring the MLP to the staged weights
instead of the committed copy. Both are structural CDC properties (STA-003).

### 4.7.1 Pipeline controller

`tb_bmi_pipeline_ctrl.sv` tests `bmi_pipeline_ctrl` on one clock with the real
`argmax` attached.

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| CTL-001 | START without committed weights is acknowledged but does not run | START after reset | RUNNING stays 0 | Implemented (`tb_bmi_pipeline_ctrl.sv`) |
| CTL-002 | Commit marks weights valid; START returns RUNNING = 1 with the acknowledgment | Commit, START | `cmd_running` in the `cmd_done` cycle | Implemented |
| CTL-003 | A result is pushed as one entry in the decision cycle | One inference | Push count and entry contents | Implemented |
| CTL-004 | FIFO full: no push, overflow set and sticky | Decision with `fifo_full` | Flags and counts | Implemented |
| CTL-005 | RESTART in the decision cycle discards that result | Coincident command | Clear wins; nothing pushed; still running | Implemented |
| CTL-006 | RESTART in the `scores_valid` cycle: argmax produces no decision | Coincident command | No decision, class cleared | Implemented |
| CTL-007 | STOP waits for the inference in flight, then clears once | STOP 20 cycles before the decision | No acknowledgment or clear early; result pushed first | Implemented |
| CTL-008 | STOP while idle is immediate; a window completing in the STOP cycle is still waited for | Both cases | Acknowledge timing; result pushed | Implemented |

### 4.8 Top-level pipeline, commands, and reset

`tb_bmi_chip_top.sv` keeps a scoreboard on the `bmi_clk` side: every entry
pushed into the result FIFO is recorded, and everything the host reads must
match those entries in order.

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| TOP-001 | Process 40 vectors without deadlock | Load, commit, START, read 40 results; even vectors by interrupt, odd by polling | Independent watchdog; 40 results | Implemented |
| TOP-002 | Match all golden intermediate and final values | Same run | SBP features at `features_done`, internal scores at the decision, class and full 32-bit `SCOREn` as read over APB | Implemented |
| TOP-003 | The pipeline runs independently of software | Host sleeps more than two windows on vectors 5, 13, 21, 29, 37 | Frontend resumes at channel 0 the cycle after each decision; results queue and are read in order | Implemented |
| TOP-004 | The snapshot is stable across a multi-register read while results keep arriving | Every result read twice with 3 to 123 `pclk` between; overflow scenario reads all five registers repeatedly | Second read equals the first and the scoreboard entry | Implemented |
| TOP-005 | Stage latencies | All windows | Frontend-to-MLP done = 122 and MLP-to-decision = 1, exactly; ADC capture spacing = `CLK_DIV` | Implemented |
| TOP-006 | Global reset while running with results queued | Assert `presetn` mid-window | Registers, staged and active weights, FIFO, RUNNING all cleared; START refused; reload, commit, START, two golden vectors | Implemented (one reset point; not every pipeline stage) |
| TOP-007 | No X/Z on valid outputs | All windows | Features, scores, class at their valid events | Implemented |
| TOP-008 | Nothing runs before START; START refused before a commit | After reset | `STATUS` 0, `pslverr`, pipeline idle | Implemented |
| TOP-009 | Weight commit is atomic and only while stopped | Initial load, a different image, then the original | Active image equals the complete staged image whenever it changes; never while running or in flight | Implemented |
| TOP-010 | Back-to-back commands | 18 commands, each followed at once by a repeat, a weight write, and a commit | Commands accepted over APB = commands executed; RUNNING matches a model | Implemented |
| TOP-011 | STOP during an inference waits for it and keeps its result | STOP sent at `features_done` | Acknowledged only when nothing is in flight; result queued; no samples for half a window afterwards | Implemented |
| TOP-012 | RESTART mid-window, during an inference, and while stopped; STOP mid-window | Four cases | A full fresh window follows; queued results and weights preserved; in-flight result discarded; stopped stays stopped; STOP discards the partial window | Implemented |
| TOP-013 | FIFO full: drop, sticky overflow, pipeline keeps running | Host stops acknowledging for eight windows | Exactly five results held; `ERROR_STATUS` = 1 and sticky; five drained in order | Implemented |
| TOP-014 | Acknowledgment races | `RESULT_ACK` 0 to 9 `pclk` after the next decision | No result lost or repeated; fresh `irq` edge each time | Implemented |
| TOP-015 | Accounting | Whole run | Windows = queued + dropped + discarded; nothing left unread | Implemented |

### 4.9 Static and backend verification

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| STA-001 | RTL compiles and runs with an open-source simulator | Icarus 40-vector regression | Implemented |
| STA-002 | RTL is clean under lint or all waivers are documented | Warning-free Verilator `-Wall`; Cadence Genus check pending | Partial (`sim/run_lint.sh`) |
| STA-003 | Clock-domain crossings are structurally safe | Protocols are documented in the `cdc_*` modules and simulated (section 4.7). Still needed: a structural CDC tool run to confirm every `pclk`/`bmi_clk` path is a synchronizer input or one of the four documented bundled-data paths. The scan and SPI crossings exist only in the retained blocks | Partial |
| STA-004 | All functional paths are correctly constrained | SDC review and unconstrained-path report. The SDC bounds every `pclk`/`bmi_clk` path with `set_max_delay` instead of cutting it, and uses a placeholder `pclk` period; it has not been run through Genus | Planned |
| STA-005 | Post-route setup and hold timing meet constraints | Existing Innovus reports | Partial |
| STA-006 | Post-route electrical design rules are clean | Transition/fanout reports | Planned |
| GLS-001 | Gate-level outputs match golden vectors | `tb_gls.sv` loads and commits the weights, starts the pipeline, and reads class and full scores for all 40 vectors over APB, using top-level ports only; passes on a Yosys generic netlist at two `pclk` periods. The Genus netlist predates this architecture and must be re-synthesized and re-run | Partial |


