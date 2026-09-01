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
| SPI packet | 80 bits |

The verified datapath is:

```text
ADC collection → SBP extraction → 8→8→4 MLP → argmax
               → packet formatter → SPI transmitter
```

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
- Model parameters come from `sim/weights.hex`.
- The Python integer pipeline is the arithmetic reference model.

The 40 generated vectors are useful regression stimulus, but they are not sufficient for corner-case closure. Directed block tests and assertions are required in addition to vector replay.

### 3.3 Checking strategy

The integration scoreboard checks each vector at multiple observation points:

1. Eight SBP features against the Python golden values.
2. Four complete signed MLP scores against the golden values.
3. Golden scores fit within the RTL's signed 32-bit score format.
4. SPI sync byte and reserved class-byte bits.
5. Predicted class.
6. All four transmitted score slices.
7. Overall regression failure through `$fatal` if any vector fails.

Four-state comparisons (`===`/`!==`) are used so unknown or high-impedance values cannot silently pass.

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
| SC-005 | Resume at channel/sample zero after `packet_ready` | Back-to-back directed and generated windows | Check complete consecutive windows | Implemented (`tb_streaming_sbp_frontend.sv`, `tb_bmi_chip_top.sv`) |
| SC-006 | Reset clears state, counters, accumulators, and features | Reset at startup and mid-collection | Directed asynchronous reset test | Implemented (`tb_streaming_sbp_frontend.sv`) |
| SC-007 | Counter and accumulator widths support legal terminal values | Parameter review and rail patterns | Elaboration checks and maximum-sum comparison | Implemented (`streaming_sbp_frontend.sv`, `tb_streaming_sbp_frontend.sv`) |

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

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| MLP-001 | Load all 864 parameter bits in the documented order | Synthetic unit models and production `weights.hex` | Check scan image, unpacked arrays, and end-to-end scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-002 | Hidden MAC uses all 8 features and signed weights | Directed signed weights, unsigned boundary inputs, and 40 generated vectors | Compare hidden activations and four final scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-003 | Add scaled hidden bias | Directed positive and negative hidden biases | Procedural reference-model comparison | Implemented (`tb_mlp_inference.sv`) |
| MLP-004 | Clamp negative hidden pre-activations to zero | Directed positive and negative pre-activations | Check every internal hidden activation | Implemented (`tb_mlp_inference.sv`) |
| MLP-005 | Output MAC uses all 8 hidden activations | Directed output connectivity and 40 generated vectors | Compare all four full scores | Implemented (`tb_mlp_inference.sv`, `tb_bmi_chip_top.sv`) |
| MLP-006 | Add scaled output bias without output ReLU | Negative, zero, and positive bias-only scores | Compare signed scores | Implemented (`tb_mlp_inference.sv`) |
| MLP-007 | Produce four valid scores and one-cycle `done` | Repeated inference transactions | Check derived latency, all scores, pulse width, and score stability | Implemented (`tb_mlp_inference.sv`) |
| MLP-008 | Arithmetic does not overflow configured widths | Positive/negative rail models with maximum unsigned inputs | Elaboration range checks plus reference-model comparison | Implemented (`mlp_inference.sv`, `tb_mlp_inference.sv`) |
| MLP-009 | Ignore `start` while busy | Reassert start during hidden MAC | Check original transaction result and latency are unchanged | Implemented (`tb_mlp_inference.sv`) |
| MLP-010 | Inference cannot start with partial weights | Incomplete scan load | Add protocol/status and negative test | Planned |

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
| ARG-006 | End-to-end predicted class matches golden model | 40 generated vectors | SPI class comparison | Implemented |

### 4.5 Packet formatter

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| FMT-001 | Byte 0 is `0xAA` | 40 generated vectors | Check received sync byte | Implemented |
| FMT-002 | Class byte upper six bits are zero | 40 generated vectors | Check reserved bits | Implemented |
| FMT-003 | Class byte lower two bits contain prediction | 40 generated vectors | Compare expected class | Implemented |
| FMT-004 | Pack score `[31:16]` fields in class order | 40 generated vectors | Compare all four packet fields | Implemented |
| FMT-005 | Hold packet stable while valid and not ready | Delay ready while changing formatter inputs | Check packet and valid over multiple cycles | Implemented (`tb_output_formatter.sv`) |
| FMT-006 | Clear valid on ready | Pulse ready, including simultaneous new decision | Check clear and documented ready priority | Implemented (`tb_output_formatter.sv`) |

### 4.6 SPI transmitter

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| SPI-001 | Transmit 80 bits MSB-first in Mode 0 | Normal read | Compare complete received packet | Implemented |
| SPI-002 | Present first MISO bit before first sample edge | Normal read with timing check | Protocol assertion/waveform check | Planned |
| SPI-003 | Pulse `packet_ready` after exactly 80 sampled bits | Normal read | Edge counter and assertion | Planned |
| SPI-004 | Abort safely when CS deasserts early | Transfers of 1–79 bits | Return to idle; no ready pulse | Planned |
| SPI-005 | Ignore clocks while CS is inactive | Toggle SCLK with CS high | Check state/output behavior | Planned |
| SPI-006 | Support documented SCLK/system-clock ratio | Sweep SCLK frequency and phase | Establish passing boundary and specification | Planned |
| SPI-007 | Reset during transfer returns interface to idle | Assert reset at several bit positions | Check state, MISO, ready | Planned |

### 4.7 Top-level pipeline and reset

| ID | Requirement | Stimulus | Checker / coverage goal | Status |
|---|---|---|---|---|
| TOP-001 | Process 40 vectors without deadlock | Existing regression | Timeout plus 40 completed packets | Partial |
| TOP-002 | Match all golden intermediate/final values | Existing regression | SBP, scores, class, packet checks | Implemented |
| TOP-003 | Do not overwrite a pending packet | Delay SPI read | Stable packet and no new collection | Planned |
| TOP-004 | Restart collection only after packet completion | Normal and delayed SPI | Handshake assertion | Planned |
| TOP-005 | Meet documented stage latencies | One transaction | Cycle counters per stage | Planned |
| TOP-006 | Recover from reset in every pipeline stage | Reset injection | Directed tests at each stage | Planned |
| TOP-007 | No X/Z values reach valid architectural outputs | Normal and reset tests | Assertions on valid events | Planned |

### 4.8 Static and backend verification

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| STA-001 | RTL compiles and runs with an open-source simulator | Icarus 40-vector regression | Implemented |
| STA-002 | RTL is clean under lint or all waivers are documented | Verilator lint report | Planned |
| STA-003 | Clock-domain crossings have documented protocols | CDC review of scan and SPI interfaces | Planned |
| STA-004 | All functional paths are correctly constrained | SDC review and unconstrained-path report | Planned |
| STA-005 | Post-route setup and hold timing meet constraints | Existing Innovus reports | Partial |
| STA-006 | Post-route electrical design rules are clean | Transition/fanout reports | Planned |
| GLS-001 | Gate-level outputs match golden vectors | Existing GLS testbench | Partial |

## 5. Coverage Model

### 5.1 Functional coverage goals

Planned coverpoints include:

- Every ADC channel selected.
- First and final sample index captured for every channel.
- SBP values at zero, near maximum, and ordinary ranges.
- Hidden ReLU both passing and clamping values.
- Each output class winning.
- All-negative class-score vector.
- Pairwise and four-way argmax ties.
- Packet-valid wait lengths, including delayed host reads.
- SPI abort at early, middle, and final bit positions.
- Reset observed in each pipeline stage.

Cross coverage should include winning class versus score-sign pattern and reset stage versus recovery result.

### 5.2 Code coverage goals

When supported by the simulator, collect:

- Statement coverage
- Branch coverage
- FSM state and transition coverage
- Toggle coverage for control signals

Code coverage is a gap-finding metric, not the definition of correctness. Uncovered code must be tested, justified as unreachable, or waived with rationale.

## 6. Regression Suites

| Suite | Contents | Intended use |
|---|---|---|
| Smoke | Compile plus one end-to-end vector | Fast edit feedback |
| Block | Directed block-level tests | Arithmetic and corner-case verification |
| Integration | All 40 generated vectors | Golden-model equivalence |
| Robustness | Reset, backpressure, malformed/aborted SPI | Protocol and recovery testing |
| GLS | All supported gate-level vectors | Netlist equivalence |
| Static | Lint, CDC review, synthesis/STA checks | Structural sign-off |

## 7. Pass/Fail Policy

A test passes only when every enabled checker passes and the simulator exits successfully. Warnings are not substitutes for protocol checks. Any mismatch, timeout, unexpected X/Z, assertion failure, or missing expected transaction must return a nonzero process status.

An expected failure used for fault injection must never be committed as part of a passing regression.

## 8. Sign-off Criteria

Frontend sign-off requires:

1. Every applicable vPlan entry is Implemented, or has a written waiver.
2. All block, integration, robustness, and GLS regressions pass.
3. Functional coverage goals are met or waived with rationale.
4. Code coverage holes are reviewed.
5. Lint has no unwaived errors or high-severity warnings.
6. CDC assumptions and reset behavior are documented and checked.
7. Fixed-point range and overflow analysis is complete.
8. Test failures return nonzero status and regressions include timeouts.

Backend sign-off additionally requires correct constraints, no unconstrained functional paths, setup/hold closure, and resolution or explicit waiver of electrical design-rule violations.

## 9. Current Baseline

As of 2026-08-26:

- RTL integration regression: **40/40 vectors pass**.
- Each vector checks 8 SBP values, 4 full scores, class/framing fields, and 4 transmitted score slices.
- The vector set demonstrates RTL/Python integer equivalence.
- The 40-vector subset has 50% prediction accuracy against biological labels; this is an ML-quality metric, not an RTL regression failure.
- Directed block tests, assertions, functional coverage, timeouts, and CDC verification remain open.

## 10. Maintenance Rules

- Give every new requirement a stable ID.
- Link each implemented entry to a test, assertion, coverage point, or report.
- Do not mark an entry Implemented based only on visual waveform inspection.
- Update status whenever RTL behavior, interfaces, parameters, or constraints change.
- Preserve failing seeds and vectors as regression tests after fixing a defect.
