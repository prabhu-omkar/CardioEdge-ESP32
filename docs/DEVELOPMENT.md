# Development and validation guide

Read [Architecture](ARCHITECTURE.md) for implementation contracts, [Datasets](DATASETS.md) for labeling/splits, [Results](RESULTS.md) for measured limitations, and [Hardware](HARDWARE.md) before attaching electrodes.

## Windows setup

Run commands from the project root in PowerShell unless indicated otherwise. The included tools primarily target Windows and Python 3.12. The C export declarations and QEMU paths currently assume Windows; other operating systems have not been validated.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
$env:PLATFORMIO_CORE_DIR = Join-Path $PWD '.tools/pio'
.venv\Scripts\pio.exe run -d firmware -e esp32dev
```

The lock file captures the existing Python environment. `requirements.txt` is the shorter dependency list with fewer version constraints. PlatformIO pins the Espressif platform; several firmware libraries use compatible version ranges, so a clean dependency resolution may differ from the existing installation.

Pretrained models and generated firmware headers are included. Flashing does not require downloading training databases or retraining. The full Python environment is for development, tests and ML work.

```powershell
.venv\Scripts\pio.exe run -d firmware -e esp32dev -t upload
.venv\Scripts\pio.exe device monitor -d firmware -e esp32dev
```

## Routine checks

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
node tests/dashboard_check.js
$env:PLATFORMIO_CORE_DIR = Join-Path $PWD '.tools/pio'
.venv\Scripts\pio.exe run -d firmware -e esp32dev
```

Python tests exercise the compiled C++ core, filters, detection, quality gating, features, HRV, rhythm rules, model export parity and tooling regressions. Some model/data tests depend on the included artifacts. The dashboard test uses a fake DOM and WebSocket; it is not a browser rendering or real-device network test.

After editing the dashboard, regenerate its header before building or running the freshness check:

```powershell
.venv\Scripts\python.exe tools/embed_web.py
```

## Emulator validation

Place Espressif QEMU under `.tools/qemu` so `.tools/qemu/bin/qemu-system-xtensa.exe` exists, then run:

```powershell
.venv\Scripts\python.exe tools/run_qemu_selftest.py
```

The runner generates 240 test vectors, builds a dedicated Xtensa firmware, boots it, and compares output with host inference. It writes `results/qemu_selftest.json` and exits nonzero on validation failure. Its timeout also handles a silent emulator and terminates the child process.

Acceptance criteria:

- Boot and completion markers must both be present.
- All 240 CNN and RF vector classes must match.
- Maximum CNN score difference is six int8 units.
- Six distinct demo clips must be present, with exact detected/analysed beat counts and CNN final verdicts.
- RF class counts must match exactly; each CNN class count may differ by at most one beat, and total counts must match.

The numerical tolerances preserve the documented host/Xtensa baseline. This verifies compiled model/core behavior, not acquisition tasks, peripherals, real Wi-Fi, real inference timing, or clinical validity.

## Rebuild the research pipeline

Downloading and training are explicit operations; normal tests do not retrain models. Downloads need network access and roughly 2 GB of raw data, plus space for beat arrays and tooling.

```powershell
.venv\Scripts\python.exe tools/download_data.py
.venv\Scripts\python.exe -m ml.dataset
.venv\Scripts\python.exe -m ml.train_rf
.venv\Scripts\python.exe -m ml.train_cnn
.venv\Scripts\python.exe -m ml.export_demo
```

Training selects hyperparameters/epoch count on validation data, then fits final models on training plus validation. The forest uses hard tree votes to match deployment. CNN conversion uses full int8 quantization and disables per-channel Dense quantization for the bundled TFLite Micro kernel.

After freezing the processing and model choices:

```powershell
.venv\Scripts\python.exe -m ml.eval_qrs
.venv\Scripts\python.exe -m ml.sqa_eval test
.venv\Scripts\python.exe -m ml.evaluate
.venv\Scripts\python.exe -m ml.ad8232_sim
.venv\Scripts\python.exe -m ml.figures
```

These commands overwrite result artifacts. Preserve a dated copy before running another experiment. Record dependency versions, data selection, source changes and the reason for every test-set access. Existing results already include two evaluation rounds, as disclosed in RESULTS.md; do not describe them as a pristine first-look test.

Beat accuracy excludes rejected/unmatched beats. Rhythm evaluation selects windows with sufficiently clear ground truth. Noise acceptance, analysed-beat coverage, false-alarm rate and class-specific recall must accompany accuracy when interpreting results. Record-name disjointness tests are useful but do not independently establish patient identity across every database.

## Hardware acceptance checklist

1. Follow the battery-power and wiring requirements in HARDWARE.md.
2. Confirm all six demos, source selection, OLED, dashboard and buzzer on a board.
3. Run with Wi-Fi connected and record `b` timing/drop counters. Stress source changes and check fresh acquisition after each transition.
4. Disconnect/reconnect leads and introduce motion. Confirm classification pauses and alarms are suppressed during poor quality.
5. Measure inference timing on the board; QEMU timing is not a substitute.
6. Capture data on a battery-powered laptop, with the device already booted and raw streaming initially off:

```powershell
.venv\Scripts\python.exe tools/record_serial.py COM5 --seconds 120 --out data/captures/rest_01.csv
.venv\Scripts\python.exe -m ml.eval_capture data/captures/rest_01.csv
```

The recorder toggles raw mode and stops after the requested number of samples, not a wall-clock deadline. If the device never streams, stop with Ctrl+C and inspect the connection. Captures must contain three finite numeric columns and at least 125 samples for analysis. The analysis reports quality, predictions and selected-interval HRV; it cannot measure classification accuracy without independent labels.

## Maintenance boundaries

- Do not edit generated model arrays by hand.
- Keep C++ and Python class order, dimensions and struct layout synchronized.
- Do not silently replace stored evaluation numbers after a runtime fix.
- A firmware build proves compilation, not real-time scheduling or peripheral operation.
- Use the regression suite plus emulator for DSP/model changes, and board tests for acquisition/UI changes.
