# CardioEdge-ESP32

**Signal-quality-aware, real-time ECG arrhythmia detection on an ESP32.**

B.Tech project: *AI-Based Real-Time ECG Signal Analysis and Multi-Class Cardiac Abnormality Detection Using an Embedded System.*

An ESP32 reads a single-lead ECG from an AD8232 front end. It filters the signal, checks signal quality, detects heartbeats and classifies each clean beat with two on-device models running side by side: a **TensorFlow Lite Micro 1D CNN** and a **Random Forest**. Results go to an OLED, a phone dashboard over Wi-Fi and a buzzer for abnormal rhythms. Everything runs on the device, with no cloud service.

## Highlights

- **One DSP codebase for device and host.** `firmware/lib/ecg_core` is plain C++ with no Arduino dependency. The firmware runs it live, and Python compiles the same source into a shared library (through `ziglang`) to build training data and run tests. Training features and on-device features come from the same code.
- **Signal quality gating.** Every 0.5 s, a 5 s window is scored on amplitude, noise (SNR and kurtosis), baseline wander and beat-shape consistency. Bad signal shows *"Please reposition electrodes"* and stops classification and alarms. At -6 dB muscle noise, false alarms fall from **78.5 to 0.7 per hour**.
- **Two classifiers side by side.** An int8 1D CNN (17 KB, 9,574 parameters) and a 30-tree Random Forest (105 KB) both run on every beat, so they can be compared directly on the device.
- **Rhythm-level decisions.** A rule based on RR irregularity detects atrial fibrillation, and a rolling 10-beat verdict smooths single-beat errors before any alarm.
- **HRV metrics.** SDNN, RMSSD and pNN50, tested against their definitions.
- **Phone dashboard.** The ESP32 runs its own Wi-Fi access point and serves a live waveform, verdicts and a downloadable event log over WebSocket.
- **Built-in demo clips.** Abnormal rhythms can be shown with no patient attached.
- **Emulator verification.** A QEMU (Xtensa) self-test boots the real firmware build and compares its outputs with host inference on 240 test vectors.
- **Evaluated on held-out patients**, with every reported number traceable to a JSON file in `results/`.

## Project status

| Area | State |
|---|---|
| DSP, R-peaks, SQA, HRV, alert logic | Tested on the firmware's own code, on real ECG databases |
| CNN and Random Forest | Trained and evaluated on patients not used in training |
| ESP32 firmware | Builds. Models and pipeline verified on an emulated ESP32 (QEMU). |
| AD8232, OLED, buzzer, Wi-Fi on a real board | **Not tested yet.** No hardware was available. |

Normal, PVC and AFib work well enough to demonstrate. LBBB, RBBB and Other are unreliable on new patients. See [Known limitations](#known-limitations).

## How it works

```text
Electrodes -> AD8232 -> ESP32 ADC1 (GPIO 34, 250 Hz)
  -> Filters: 0.5 Hz high-pass, 50 Hz notch, 40 Hz low-pass
  -> Adaptive Pan-Tompkins R-peak detection
  -> Signal quality: amplitude, noise (SNR + kurtosis), baseline, morphology
        bad  -> "Please reposition electrodes", no classification, no alarm
        good -> beat extraction (delayed about 2 s, so the next RR interval is known)
             -> 1D CNN (TFLite Micro)  +  Random Forest
             -> AF rhythm rule (per classifier)
             -> rolling 10-beat verdict
  -> OLED, buzzer/LED, phone dashboard, event log
```

**Tasks and timing (dual core):**

| Core | Task | Role |
|---|---|---|
| 1 | Acquisition | A hardware timer fires every 4 ms (250 Hz) and pushes source-tagged, sequence-numbered samples into a 128-sample queue (about 512 ms) |
| 1 | DSP | Filtering, QRS, SQA, beat extraction, both classifiers, rhythm logic |
| 0 | UI | OLED refresh, HTTP and WebSocket dashboard, serial commands |

- QRS learning takes about 2.5 s after start or after a source change.
- Recovering from bad signal takes two consecutive good quality checks.
- A source change or a missing sequence number resets the pipeline, HRV and verdicts. The firmware never inserts made-up samples.
- The **CNN verdict controls the alarm**. Both verdicts are displayed.

Full contracts, latency budget and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Classes detected

| Class | Source annotations |
|---|---|
| Normal | N outside AF |
| PVC | V |
| AFib | N during an AFIB rhythm |
| LBBB | L |
| RBBB | R |
| Other | A, a, J, S, e, E, j, F, f, and N during flutter |

## Results at a glance

These figures come from held-out test patients. Source data is in `results/*.json`, and the full analysis is in [docs/RESULTS.md](docs/RESULTS.md).

| Capability | Result | Verdict |
|---|---|---|
| R-peak detection (44 MIT-BIH records, 100,460 beats) | Sensitivity 99.11%, PPV 99.59% | Works well |
| PVC detection | Beat F1 0.87 on MIT-BIH test patients | Works well |
| AFib vs normal rhythm | Recall 77%, precision 82% on rhythm windows | Usable |
| Signal quality gating | False alarms 78.5 → 0.7 per hour at -6 dB muscle noise | Works well |
| LBBB, RBBB | Precision 0.85–0.91, recall 0.48–0.64 on PTB-XL; much worse on MIT-BIH | **Unreliable** |
| "Other" class | Recall 3% | **Does not work** |
| ESP32 fit | 17 KB model, 3.7 KB tensor arena, 27% RAM, 41% flash | Fits comfortably |

**Classifier comparison** (123 test patients, 100,522 beats, each classifier followed by the AF rule, as on the device):

| Metric | 1D CNN (int8) | Random Forest |
|---|---:|---:|
| Accuracy | 73.4% | 68.2% |
| Macro precision | 0.474 | 0.463 |
| Macro recall | 0.491 | 0.499 |
| Macro F1 | 0.473 | 0.455 |
| Model size | 17 KB, 9,574 parameters | 105 KB, 30 trees, 11,970 nodes |
| Inference time on ESP32 | Not measured yet (estimate 40–60 ms) | Not measured yet |

| Confusion matrix | SQA false alarms |
|---|---|
| ![Confusion matrix](docs/figures/confusion.png) | ![SQA false alarms](docs/figures/sqa_false_alarms.png) |

## Hardware

| Part | Notes |
|---|---|
| ESP32 DevKit (ESP32-WROOM-32, 4 MB flash) | The `esp32dev` build targets this board |
| AD8232 ECG module with 3-lead cable | |
| Disposable Ag/AgCl gel electrodes | Use fresh ones. Dried gel is the most common cause of noise. |
| SSD1306 OLED, 128×64, I2C (0x3C) | |
| Active 3.3 V buzzer | Use an NPN transistor if it draws over 12 mA |
| LiPo battery or power bank | **Not charging while in use** |

**Wiring:**

| AD8232 / peripheral | ESP32 |
|---|---|
| AD8232 3.3V / GND | 3V3 / GND (do not use 5 V) |
| AD8232 OUTPUT | GPIO 34 (must be ADC1, because ADC2 stops working with Wi-Fi on) |
| AD8232 LO+ / LO- | GPIO 32 / GPIO 33 (lead-off detect) |
| OLED SDA / SCL | GPIO 21 / GPIO 22 |
| Buzzer + | GPIO 25 |
| Onboard LED | GPIO 2 (mirrors the alert) |

All pins are defined in `firmware/src/config.h`. Electrode placement matters: the models were trained mostly on lead II, so use the modified lead II placement in [docs/HARDWARE.md](docs/HARDWARE.md), which also has the first power-up checklist.

## Quick start: flash the firmware

The trained models and generated firmware headers are already in the repo, so flashing needs **no datasets and no training**. You need Python 3.12 (TensorFlow does not support 3.14 yet). The commands below are for Windows PowerShell.

```powershell
git clone https://github.com/prabhu-omkar/CardioEdge-ESP32.git
cd CardioEdge-ESP32

py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt   # or requirements.txt for looser pins

$env:PLATFORMIO_CORE_DIR = Join-Path $PWD '.tools/pio'               # optional: keep PlatformIO local to the repo
.venv\Scripts\pio.exe run -d firmware -e esp32dev                    # build
.venv\Scripts\pio.exe run -d firmware -e esp32dev -t upload          # flash
.venv\Scripts\pio.exe device monitor -d firmware -e esp32dev         # serial monitor (921600 baud)
```

## Using the device

- **BOOT button:** steps through live AD8232 input and the built-in demo clips (Normal, PVC, AFib and others), so abnormal rhythms can be shown without a patient.
- **OLED:** live trace, heart rate, signal quality status and the current verdict.
- **Phone dashboard:** connect to the Wi-Fi access point `ECG-Monitor` (default password `ecgmonitor`, which you can change in `firmware/src/config.h`) and open the address printed on the serial monitor. It shows a live waveform, CNN and RF verdicts, HRV and a downloadable event log.
- **Buzzer and LED:** sound when the CNN rhythm verdict is abnormal and signal quality is good.
- **Serial commands** (921600 baud):

  | Key | Action |
  |---|---|
  | `n` | Next input source |
  | `b` | Report measured inference timing for the CNN and RF |
  | `r` | Toggle the raw sample stream |

## Reproduce training and evaluation

This needs about 2 GB of public PhysioNet data. Training runs on a CPU, so no GPU is required.

```powershell
.venv\Scripts\python tools\download_data.py     # MIT-BIH, AFDB, PTB-XL, INCART, NSTDB
.venv\Scripts\python -m ml.dataset              # beat extraction with the C++ core
.venv\Scripts\python -m ml.train_rf             # Random Forest -> models/rf.pkl, firmware/src/model_rf.h
.venv\Scripts\python -m ml.train_cnn            # 1D CNN -> models/cnn_int8.tflite, firmware/src/model_cnn.h
.venv\Scripts\python -m ml.export_demo          # demo clips -> firmware/src/demo_clips.h
.venv\Scripts\python -m ml.eval_qrs             # R-peak detector benchmark
.venv\Scripts\python -m ml.sqa_eval test        # signal quality robustness
.venv\Scripts\python -m ml.evaluate             # classifier and rhythm metrics
.venv\Scripts\python -m ml.ad8232_sim           # simulated AD8232 front-end robustness
.venv\Scripts\python -m ml.figures              # docs/figures/*.png
```

If you change preprocessing, rebuild the dataset and models, because the exported headers depend on them. To check real captures from the board (record them on a laptop running on battery):

```powershell
python tools\record_serial.py COM5 --seconds 120 --out data\captures\rest_01.csv
python -m ml.eval_capture data\captures\rest_01.csv
```

## Testing

```powershell
.venv\Scripts\python.exe -m pytest tests -q      # C++ core, filters, QRS, SQA, features, HRV, rhythm, model export parity
node tests\dashboard_check.js                    # dashboard logic with a fake DOM and WebSocket
```

After editing `firmware/web/dashboard.html`, regenerate the embedded header with `python tools/embed_web.py`.

**Emulator self-test:** place [Espressif QEMU](https://github.com/espressif/qemu/releases) so that `.tools/qemu/bin/qemu-system-xtensa.exe` exists, then run:

```powershell
$env:PLATFORMIO_CORE_DIR = Join-Path $PWD '.tools/pio'
.venv\Scripts\python.exe tools\run_qemu_selftest.py
```

This builds a dedicated Xtensa firmware, boots it in QEMU and checks that its beat counts, classifications and rhythms match host inference. The result is written to `results/qemu_selftest.json`. QEMU is not cycle accurate, so it does not give real inference timing.

## Project layout

| Path | Contents |
|---|---|
| `firmware/lib/ecg_core/` | Portable C++ DSP: filters, QRS, SQA, features, HRV, rhythm logic, C API |
| `firmware/src/` | ESP32 application, classifiers, OLED, Wi-Fi dashboard, generated model and demo headers |
| `firmware/web/dashboard.html` | Phone dashboard source |
| `firmware/platformio.ini` | PlatformIO build configuration |
| `ml/` | Dataset building, training, evaluation, figures, ctypes binding to the core |
| `models/` | Trained models (`cnn.keras`, `cnn_int8.tflite`, `rf.pkl`) |
| `results/` | Every measured number, as JSON |
| `tools/` | Data download, QEMU self-test, serial capture, dashboard embedding |
| `tests/` | Automated tests |
| `docs/` | Architecture, development, hardware, datasets, results, maintenance |

Not committed (see `.gitignore`): `data/` (downloaded databases and extracted beats), `build/`, `logs/`, `.venv/`, `.tools/`, `firmware/.pio/`.

## Mapping to the proposal objectives

| Objective | Where |
|---|---|
| 1, 2. AD8232 acquisition on ESP32 | `firmware/src/main.cpp`, [HARDWARE](docs/HARDWARE.md). Not yet run on hardware. |
| 3. Noise reduction | `firmware/lib/ecg_core/ecg_core.cpp` |
| 4. R-peaks and RR intervals | Pan-Tompkins in `ecg_core.cpp`, 99.11% sensitivity |
| 5. Heart rate and HRV | `HrvTracker`: SDNN, RMSSD, pNN50 |
| 6, 7. AI classification, TinyML | `ml/train_cnn.py`, `ml/train_rf.py`, `firmware/src/classifiers.cpp` |
| 8. Real-time display | `firmware/src/display.cpp`, `firmware/web/dashboard.html` |
| 9. Alerts | `RhythmTracker`, buzzer logic in `main.cpp` |
| 10. Evaluation | Database evaluation done in [RESULTS](docs/RESULTS.md). Experimental signals still to do. |
| Extension: mobile and logging | Wi-Fi dashboard and downloadable event log |

## Known limitations

- **No real-hardware testing yet.** The AD8232, OLED, buzzer, Wi-Fi and the real ADC have not been run on a physical board, and ESP32 inference time has not been measured.
- **LBBB, RBBB and Other are unreliable** on unseen patients. Only Normal, PVC and AFib are fit to demonstrate.
- **Single lead.** The models were trained mostly on lead II. A different electrode placement may reduce accuracy.
- **Delayed decisions.** Beats are classified about 2 s after their R peak, and verdicts need several beats.
- **Label noise.** PTB-XL labels cover whole records, and AFDB has rhythm labels only. See [docs/DATASETS.md](docs/DATASETS.md).
- The tooling targets **Windows and Python 3.12**. Other operating systems have not been validated.

## Next steps

1. Build the circuit and run the checklist in [docs/HARDWARE.md](docs/HARDWARE.md).
2. Press `b` in the serial monitor to get real inference times for the comparison table.
3. Record real captures (from a battery-powered laptop) and evaluate them with `ml.eval_capture`.
4. Compare the captures with [docs/RESULTS.md](docs/RESULTS.md) and report any drop in accuracy.
5. Improve the minority classes (LBBB, RBBB, Other) with more patients or patient-specific calibration.

## Datasets and citation

All databases are public on [PhysioNet](https://physionet.org/). Cite each database and PhysioNet itself in any report.

| Database | Used for |
|---|---|
| MIT-BIH Arrhythmia | All classes. DS1 train, DS2 test (de Chazal 2004 split). Paced records excluded. |
| MIT-BIH Atrial Fibrillation | AFib and Normal |
| PTB-XL | LBBB, RBBB, Normal |
| St Petersburg INCART | PVC, Other, RBBB, Normal (training only) |
| MIT-BIH Noise Stress Test | Signal quality evaluation |

Details, splits and label caveats: [docs/DATASETS.md](docs/DATASETS.md).

## Documentation

- [Architecture and interfaces](docs/ARCHITECTURE.md): processing, tasks, latency, decisions and generated artifacts
- [Development and validation](docs/DEVELOPMENT.md): setup, builds, tests, emulator, reproduction and hardware acceptance
- [Hardware](docs/HARDWARE.md): parts, wiring, electrode placement, safety and power-up checklist
- [Datasets](docs/DATASETS.md): sources, splits and class mapping
- [Results](docs/RESULTS.md): every metric, with its limits
- [Maintenance verification](docs/MAINTENANCE.md): changes and checks from the latest improvement pass
