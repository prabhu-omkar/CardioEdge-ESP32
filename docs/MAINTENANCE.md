# Maintenance verification ? 2026-11-10

## Changes

- Guard invalid rhythm classes before array indexing. Failed device inference clears pending verdicts, HRV and displayed clinical state.
- Attach source and sequence metadata to queued samples. Reset DSP and clinical history at source boundaries or detected sample loss; count missed timer notifications as well as queue drops.
- Show `rhythm rule` instead of a misleading CNN confidence percentage when the AF rule changes the class. The embedded dashboard header was regenerated.
- Align offline capture HRV interval selection with the live device, including rejected-beat gaps and PVC/Other neighbours. Reject malformed/short captures.
- Make raw sample and lead-flag arrays contiguous at the Python/C boundary and reject mismatched flag lengths or nonfinite/non-vector sample input.
- Place default Zig compiler caches inside `build/`, respecting explicit environment overrides. This permits rebuilding in a workspace with restricted writes outside the project.
- Enforce emulator comparison results through a nonzero exit status, and terminate a silent emulator at the timeout.
- Add architecture, interface, development and validation documentation; clarify that the QRS aggregate includes training records.

## Validation scope

The model weights and feature extraction algorithms were not retrained or redesigned. Historical classification/robustness figures remain in RESULTS.md and the original JSON artifacts. The prior emulator report is preserved as `results/qemu_selftest_baseline_20261105.json`.

The firmware build covers the real application. The emulator covers the portable pipeline and deployed models, not the FreeRTOS acquisition/UI tasks. Tests cover invalid class recovery, capture interval selection, strided input, malformed lead flags, dashboard rule labels and emulator failure/timeout handling, in addition to the original suite.

## Results of this pass

| Check | Outcome |
|---|---|
| Python regression suite | 50 passed, four dependency deprecation warnings; 48.93 seconds |
| Dashboard fake-device check | Passed, including rule-adjusted confidence text |
| ESP32 production build | Passed; 87,140 bytes static RAM, 1,289,009 bytes application flash |
| Fresh QEMU run | Passed; 240/240 CNN and 240/240 RF vector classes match |
| Demo replay parity | All six clips passed the documented count/verdict tolerances |
| Tensor arena actually used | 3,652 bytes (the firmware reserves 24 KiB) |

The production build reports an existing `send_P` deprecation warning. Python reports existing WFDB/pandas and TFLite interpreter deprecations. The first host rebuild failed because Zig attempted to write its default cache outside the writable workspace; the project-local cache change resolved it, and the full suite was rerun successfully.

Verification evidence: `logs/tests_improvements.log`, `logs/qemu_improvements.log`, and `results/qemu_selftest.json`. Classification accuracy was not re-evaluated in this maintenance pass.

## Outstanding hardware work

Source transitions under load, sample loss recovery, physical lead-off behavior, Wi-Fi/OLED/buzzer integration and inference latency still require a real ESP32. The CSV format has no timestamps or sequence IDs, so offline captures cannot reconstruct lost time. Existing weak LBBB/RBBB/Other performance and signal-quality limitations are unchanged.
