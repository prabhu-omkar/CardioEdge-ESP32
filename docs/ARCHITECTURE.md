# Architecture and runtime behavior

This project is an educational, single-channel ECG processing prototype. It does not provide a clinical diagnosis. The deployed device runs locally: neither inference nor the dashboard needs a cloud service.

## Components and ownership

| Component | Source | Responsibility |
|---|---|---|
| Portable processing core | `firmware/lib/ecg_core/` | Filters, QRS detection, signal quality, features, HRV and rhythm decisions |
| Device application | `firmware/src/main.cpp` | Acquisition, task scheduling, source changes, live state, alarms and logging |
| Classifiers | `firmware/src/classifiers.cpp`, `rf_predict.h` | TFLite Micro CNN and hard-voting Random Forest |
| User interfaces | `display.cpp`, `webui.cpp`, `firmware/web/dashboard.html` | OLED, HTTP and WebSocket dashboard |
| Host binding | `ml/core.py`, `ecg_capi.cpp` | Compile the portable core and call it through ctypes |
| Data and models | `ml/dataset.py`, `train_cnn.py`, `train_rf.py` | Prepare splits, select training settings, fit and export models |
| Evaluation | `ml/evaluate.py`, `eval_qrs.py`, `sqa_eval.py`, `ad8232_sim.py` | Separate detector, classifier, rhythm and robustness measurements |

## Data path

```text
AD8232 ADC or demo samples (250 Hz, electrode-equivalent mV)
  -> source/sequence-tagged sample queue
  -> high-pass 0.5 Hz -> notch 50 Hz -> low-pass 40 Hz
  -> adaptive Pan-Tompkins QRS detector
  -> quality assessment and delayed beat extraction
  -> CNN + Random Forest
  -> AF rhythm rule, independently for each classifier
  -> rolling rhythm verdicts
  -> CNN verdict controls alarm; both verdicts are displayed
```

Acquisition and DSP run on ESP32 core 1; UI runs on core 0. A hardware timer wakes acquisition every 4 ms. The queue holds 128 samples (about 512 ms). Acquisition never blocks on a full queue.

Each queued sample carries its actual source and a sequence number. DSP changes sources when it reaches the first sample from that source, preserving FIFO ordering. A source change or missing sequence resets processing, HRV, verdicts and displayed clinical state. Timer notifications that accumulate before acquisition wakes are counted as missed samples; no fabricated samples are inserted. The drop counter is cumulative. Hardware scheduling and queue behavior still require board testing.

The waveform UI has its own rolling history, so an old trace can remain visible briefly after a source change; the clinical pipeline is reset independently.

## Processing and latency

The core uses fixed-size buffers with no Arduino dependency. Python compiles the same implementation into `build/ecg_core.dll` on Windows. Struct sizes are checked when the binding loads. Raw sample and lead-flag arrays are made contiguous before crossing the C boundary, and malformed input shapes are rejected. Compiler caches default to `build/zig-global-cache` and `build/zig-local-cache`; environment overrides are respected.

- QRS learning takes approximately 2.5 seconds.
- Signal quality examines a five-second window every 0.5 seconds. Two consecutive good evaluations are required to recover.
- Quality checks cover amplitude, noise/SNR and kurtosis, baseline movement, and similarity between recent beat shapes.
- Beats are emitted at least two seconds after their R peak. This allows quality assessment around the beat and use of its following RR interval; this is delayed streaming inference, not instantaneous detection.
- Classification also requires an eight-beat template and at least four usable RR intervals. Startup is commonly around 15 seconds and depends on rhythm and quality.

A beat has a 160-sample normalized waveform: 60 samples before R and 100 from R onward. Thirteen context features describe relative RR timing, irregularity and template similarity. Thirty-two morphology features summarize waveform regions, QRS width, amplitude, slopes and variability.

The CNN consumes waveform plus context. The forest consumes all 45 context/morphology features. Feature dimensions and class order are a cross-language contract; update and test Python and C++ together if they change.

## Decisions and failure behavior

Class order is `Normal, PVC, AFib, LBBB, RBBB, Other` (IDs 0 through 5).

The AF rule uses relative RR irregularity. It can promote Normal/Other to AFib or demote AFib to Normal. Wide/ectopic predictions are otherwise preserved. A changed CNN class is displayed as **rhythm rule**, without assigning it the original class probability. WebSocket `conf = -1` marks that case. Other confidence values are model scores, not calibrated diagnostic probabilities.

The rhythm tracker keeps ten classified beats and starts reporting after five. AFib, LBBB, RBBB and Other use the code's 60% threshold (integer truncation during startup); at least three PVCs trigger a frequent-PVC verdict when a higher-priority condition does not win. Eight consecutive abnormal verdict updates enable the alarm. These are beat counts, not fixed durations.

Poor signal suppresses alarms and blocks classification. Existing rhythm/HRV history is retained across quality pauses; the next usable beat must satisfy core recovery rules. Source changes and sample losses reset that history. Invalid classifier output clears pending verdicts and clinical state; invalid values are also rejected defensively by `RhythmTracker` before indexing its class table.

Heart rate uses up to eight usable RR intervals. HRV uses up to 60 accepted intervals and excludes intervals touching PVC or Other predictions. AFib and bundle-branch classes remain eligible: do not interpret this as a strict normal-sinus-only HRV analysis. `ml.eval_capture` uses the same interval eligibility and breaks adjacency at rejected beats.

## Interfaces

The device creates the access point configured in `firmware/src/config.h` (default `ECG-Monitor`, password `ecgmonitor`). Open `http://192.168.4.1`.

| Interface | Contract |
|---|---|
| `GET /` | Embedded dashboard HTML |
| `GET /log.csv` | Download event log; empty response if absent |
| `POST /log/clear` | Delete stored event log |
| WebSocket `/ws`, text from browser | Numeric source ID: 0 live, 1-6 demo clips |
| WebSocket binary from device | Little-endian signed int16 microvolt samples, 125 Hz, 25 samples/frame |
| WebSocket text from device | JSON status every 500 ms: source, HR/HRV, verdicts, latest classes, confidence, timing, quality and agreement |

The access point password is the access boundary; there is no separate HTTP login or TLS. Anyone connected can switch sources or clear the log.

Serial uses 921600 baud. `n` cycles sources, `b` reports inference timing and drops, and `r` toggles raw output. Raw rows are `R,raw_mv,filtered_mv,leads_off`. Beat/status/benchmark rows begin with `B`, `S`, and `BENCH`. Raw capture has no timestamps or sequence IDs, so a CSV alone cannot prove that acquisition was uninterrupted.

The LittleFS event log records changes in rhythm, capped at 256 KiB without rotation. It has no header: columns are uptime seconds, CNN rhythm, source name, heart rate, SDNN, RMSSD and SNR. It records transitions, not a continuous ECG or wall-clock timestamps.

## Generated artifacts

| Artifact | Generator |
|---|---|
| `data/beats/*.npz` | `python -m ml.dataset` |
| `models/rf.pkl`, `firmware/src/model_rf.h` | `python -m ml.train_rf` |
| `models/cnn.keras`, `models/cnn_int8.tflite`, `firmware/src/model_cnn.h` | `python -m ml.train_cnn` |
| `firmware/src/demo_clips.h` | `python -m ml.export_demo` |
| `firmware/src/dashboard_html.h` | `python tools/embed_web.py` |
| `firmware/src/selftest_vectors.h` | `python tools/run_qemu_selftest.py` |
| `docs/figures/*.png` | `python -m ml.figures` |

Edit generators/source files rather than generated headers. Changes to preprocessing require rebuilding datasets and retraining; existing models are tied to the old feature distribution. Runtime handling improvements do not themselves establish improved classification accuracy.
