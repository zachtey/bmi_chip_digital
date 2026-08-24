# BMI ASIC

Real-time brain-machine interface classifier. Eight electrode channels are read through a time-multiplexed ADC, spiking band power (SBP) features are extracted per channel, and a small MLP classifies the neural activity into one of four hand-gesture classes. Results are reported to a host over SPI.

Designed by Zachary Tey, Christopher Leung, Rohit Seshadri, Ben Rivera Flores
Full project report attached in repository under "392-final-report.pdf"
## Output Classes

| Index | Label | Description |
|-------|-------|-------------|
| 0 | PG-LF | Power Grasp, Low Force |
| 1 | PG-HF | Power Grasp, High Force |
| 2 | SG-LF | Spherical Grasp, Low Force |
| 3 | SG-HF | Spherical Grasp, High Force |

## Architecture

```
Electrodes (8 ch)
     │
   [ADC] ──── adc_channel MUX select
     │
sample_collection          fills 8 × 250 sample window
     │  window_ready
     ▼
sbp_feature_extraction     SBP[ch] = Σ|x−128| >> 8   (2000 cycles)
     │  sbp_done
     ▼
mlp_inference              8→8(ReLU)→4 MLP, single MAC (~121 cycles)
     │  mlp_done
     ▼
argmax                     tournament comparator        (1 cycle)
     │  decision_valid
     ▼
output_formatter           packs 10-byte SPI packet    (1 cycle)
     │  packet_valid
     ▼
spi_slave                  SPI Mode 0 shift-out        (80 SCLK cycles)
     │  packet_ready ──────────────────────────────────────────────┐
     ▼                                                             │
  SPI master                                          sample_collection.resume
                                                       (pipeline restarts)
```

The pipeline is self-throttling: collection does not restart until the SPI transmission completes.

## Block Summary

| File | Module | Purpose |
|------|--------|---------|
| `bmi_chip_top.sv` | `bmi_chip_top` | Top-level structural wrapper |
| `sample_collection.sv` | `sample_collection` | ADC MUX control, window filling |
| `sram_sample_win.sv` | `sram_sample_win` | Behavioral SRAM model for synthesis (not instantiated in top — alternative to the reg array in sample_collection) |
| `sbp_feature_extractor.sv` | `sbp_feature_extraction` | SBP feature extraction |
| `mlp_inference.sv` | `mlp_inference` | MLP inference engine + scan chain |
| `argmax.sv` | `argmax` | Class decision |
| `output_formatter.sv` | `output_formatter` | SPI packet assembly |
| `spi_slave.sv` | `spi_slave` | SPI Mode 0 transmitter |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CH` | 8 | Electrode channels |
| `N_SAMPLES` | 250 | Samples per channel per inference window |
| `ADC_WIDTH` | 8 | ADC resolution (bits) |
| `N_HIDDEN` | 8 | Hidden neurons in MLP |
| `N_OUT` | 4 | Output classes |
| `HIDDEN_BIAS_SCALE` | 85 | Quantization correction for hidden layer biases |
| `OUTPUT_BIAS_SCALE` | 109 | Quantization correction for output layer biases |
| `PKT_BYTES` | 10 | SPI packet size |

`HIDDEN_BIAS_SCALE` and `OUTPUT_BIAS_SCALE` are printed in the `weights.hex` header by the Python training pipeline. Update them whenever weights are retrained.

## Weight Loading (Scan Chain)

MLP weights are loaded once at power-up via a serial scan chain before any inference.

Total: **108 bytes = 864 bits** in the following order, LSB-first per byte:

```
hw[0][0..7]  hw[1][0..7]  ...  hw[7][0..7]   — 64 bytes, hidden weights
hb[0..7]                                       —  8 bytes, hidden biases
ow[0][0..7]  ow[1][0..7]  ...  ow[3][0..7]   — 32 bytes, output weights
ob[0..3]                                       —  4 bytes, output biases
```

Procedure:
1. Assert `scan_en`.
2. Clock in all 864 bits on `scan_clk` (separate from system clock).
3. Deassert `scan_en`.
4. Begin normal operation — assert `rst_n` then let the pipeline run.

Do not assert `start` (via `window_ready`) while the scan chain is still loading.

## SPI Packet Format

10-byte packet, MSB-first (byte 0 transmitted first). SPI Mode 0 (CPOL=0, CPHA=0).

```
Byte 0      0xAA                    sync / framing byte
Byte 1      {6'b0, class[1:0]}      predicted class index (0–3)
Bytes 2–3   score[0][31:16]         PG-LF raw score (upper 16 bits)
Bytes 4–5   score[1][31:16]         PG-HF raw score
Bytes 6–7   score[2][31:16]         SG-LF raw score
Bytes 8–9   score[3][31:16]         SG-HF raw score
```

Scores are signed 32-bit integers; only the upper 16 bits are transmitted. The predicted class is the index of the largest score. The 0xAA sync byte can be used by the host for frame alignment recovery.

## Pipeline Latency

Measured in system clock cycles from `window_ready` to `packet_ready`:

| Stage | Cycles |
|-------|--------|
| SBP extraction | 2 000 |
| MLP inference | 121 |
| Argmax + formatter | 2 |
| **Total (excl. SPI)** | **~2 123** |

At 10 MHz system clock this is ~212 µs, well within the 50 ms collection window.

SPI transmission adds 80 SCLK cycles; duration depends on the master's clock rate.

## Directory Structure

```
digital/
├── rtl/               RTL source (SystemVerilog)
├── hdl/
│   ├── sv/            Alternate RTL source tree (SystemVerilog)
│   └── sim/           Block-level simulation scripts and test vectors
├── sim/               Top-level simulation (Xcelium); RTL and GLS runs
│   └── vectors/       Input stimulus and expected-output vectors
├── ml/                Python training pipeline, quantized weights, and
│   └── bmi_pipeline/    generated test vectors for RTL verification
├── syn/               Synthesis (Cadence Genus): scripts, netlists, SDC, reports
│   └── fv/            Formal verification scripts
├── pnr/               Place-and-route (Cadence Innovus) checkpoint saves
│   ├── clock_report/
│   ├── digital_final_reports/
│   └── timingReports/
├── results/           Post-implementation summary reports (timing, area, specs)
└── README.md
```
