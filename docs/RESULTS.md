# Results

The classification and robustness figures below describe the existing trained models and evaluation artifacts from 2026-11-05. Runtime maintenance does not constitute a new accuracy evaluation; see [Maintenance verification](MAINTENANCE.md) for subsequent checks.

The source measurements are in `results/*.json`. Classifier test metrics use held-out records/patient identifiers; the QRS detector summary uses all 44 non-paced MIT-BIH records, including training records. Resource and emulator measurements are engineering checks, not patient evaluations. Nothing was measured on real AD8232 hardware. See [What has not been tested](#what-has-not-been-tested).

## Summary

| Capability | Result | Verdict |
|---|---|---|
| R-peak detection (44 records, 100,460 beats) | Sensitivity 99.11%, PPV 99.59% | Works well |
| AFib vs normal rhythm | Recall 77%, precision 82% on rhythm windows | Usable |
| PVC detection | Beat F1 0.87 on MIT-BIH test patients | Works well |
| Signal quality gating | False alarms 78.5 → 0.7 per hour at -6 dB muscle noise | Works well |
| LBBB, RBBB | Precision 0.85-0.91, recall 0.48-0.64 on 95 PTB-XL patients. Much worse on MIT-BIH. | **Unreliable** |
| "Other" class | Recall 3% | **Does not work** |
| ESP32 fit | 17 KB model, 3.7 KB arena, 27% RAM, 41% flash | Fits comfortably |

## Classifier comparison

This fills in the table from the project diagram. Scores use each classifier followed by the AF rhythm rule, the way the device runs them.

**All test patients** (123 patients, 100,522 beats):

| Metric | 1D CNN (int8) | Random Forest |
|---|---:|---:|
| Accuracy | 73.4% | 68.2% |
| Precision (macro) | 0.474 | 0.463 |
| Recall (macro) | 0.491 | 0.499 |
| F1-score (macro) | 0.473 | 0.455 |
| Inference time on ESP32 | **Not measured** (estimate 40-60 ms) | **Not measured** |
| Model size | 17 KB, 9,574 parameters | 105 KB, 30 trees, 11,970 nodes |

To complete the inference row, flash a real ESP32, let it run on a demo clip, then type `b` in the serial monitor. The estimate scales Espressif's published person-detection time to this model's roughly 75,000 multiply-accumulates. The library in use runs reference kernels, without ESP-NN acceleration. QEMU is not cycle accurate, so it cannot provide this number.

**By test set:**

| Test set | Patients | Beats | CNN accuracy | CNN macro-F1 | RF accuracy | RF macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MIT-BIH DS2 | 22 | 47,137 | 69.1% | 0.471 | 65.0% | 0.480 |
| MIT-BIH AF database | 6 | 52,310 | 77.1% | 0.814 | 70.9% | 0.777 |
| PTB-XL fold 10 | 95 | 1,075 | 79.6% | 0.745 | 78.9% | 0.734 |

**Per class, CNN, precision / recall / F1:**

| Class | MIT-BIH DS2 | AF database | PTB-XL |
|---|---|---|---|
| Normal | 0.77 / 0.86 / 0.82 | 0.84 / 0.77 / 0.80 | 0.78 / 0.99 / 0.87 |
| PVC | 0.89 / 0.86 / 0.87 | - | - |
| AFib | 0.53 / 0.76 / 0.62 | 0.89 / 0.77 / 0.82 | - |
| LBBB | 0.33 / 0.04 / 0.08 | - | 0.91 / 0.64 / 0.75 |
| RBBB | 0.38 / 0.39 / 0.38 | - | 0.85 / 0.48 / 0.61 |
| Other | 0.28 / 0.03 / 0.05 | - | - |

![Confusion matrices](figures/confusion.png)

**Which classifier is better?** The CNN is more accurate overall and much more precise on PVCs (0.79 vs 0.59 across all test patients). It also needs about a sixth of the flash. The Random Forest catches more LBBB beats. On macro-F1 the two are close. The CNN is the device's primary classifier and the Random Forest runs beside it for comparison.

### Why these numbers look lower than published papers

Many papers report 95-99%. Most of those use random beat splits, so beats from the same person land in both training and testing. This project keeps every patient on one side of the split. In this project's own development, random-style results would have hidden a complete failure: with MIT-BIH alone, LBBB and RBBB scored F1 = 0 on a new patient.

Inter-patient results are the honest estimate of what happens with a new person, which is the situation at a demo.

## Rhythm verdict on full recordings

The device smooths per-beat classes into a 10-beat verdict. Scored on 296,001 windows from full test recordings in time order:

| Verdict | CNN recall | CNN precision | RF recall | RF precision |
|---|---:|---:|---:|---:|
| Normal | 0.89 | 0.85 | 0.81 | 0.83 |
| AFib | 0.77 | 0.82 | 0.77 | 0.83 |
| PVC (frequent) | 0.48 | 0.68 | 0.69 | 0.51 |
| RBBB | 0.41 | 0.20 | 0.10 | 0.02 |
| LBBB | 0.00 | 0.02 | 0.13 | 0.10 |

Overall verdict accuracy is 82.3% for the CNN and 77.7% for the Random Forest. LBBB is effectively never reported by the CNN on these recordings.

## Does signal quality checking help?

This tests the project's key improvement. The middle third of each MIT-BIH test record was corrupted with real recorded noise from the Noise Stress Test Database, then processed with SQA on and off.

| Noise | SNR | Accuracy, SQA off | Accuracy, SQA on | False alarms/h, off | False alarms/h, on |
|---|---:|---:|---:|---:|---:|
| Electrode motion | 12 dB | 67.5% | 68.3% | 0.7 | 0.0 |
| Electrode motion | 6 dB | 59.0% | 65.4% | 12.7 | 6.7 |
| Electrode motion | 0 dB | 34.1% | 60.6% | 42.6 | 3.0 |
| Electrode motion | -6 dB | 17.6% | 67.9%* | 38.9 | 0.0 |
| Muscle artefact | 12 dB | 67.9% | 68.3% | 0.0 | 0.0 |
| Muscle artefact | 6 dB | 66.6% | 67.1% | 0.0 | 0.0 |
| Muscle artefact | 0 dB | 54.2% | 62.8% | 32.2 | 6.0 |
| Muscle artefact | -6 dB | 33.9% | 66.3% | 78.5 | 0.7 |

\* Almost no beats pass at this level (under 1%), so this accuracy rests on very few beats. The false-alarm figure is the meaningful one.

![False alarms with SQA on and off](figures/sqa_false_alarms.png)

**Acceptance rate on test patients:** 98.0% of clean signal is accepted (worst record 90.0%).

| Noise | 12 dB | 6 dB | 0 dB | -6 dB |
|---|---:|---:|---:|---:|
| Electrode motion | 97% | 84% | 30% | 2% |
| Muscle artefact | 97% | 90% | 55% | 20% |
| Baseline wander | 97% | 95% | 79% | 41% |

**Known limitation:** ventricular flutter and fibrillation have a low-kurtosis waveform that resembles noise, so SQA can report "poor signal" during those rhythms. This prototype must not be relied on to detect them.

## AD8232 front-end simulation

Test patients' ECGs were passed through a software model of the module's filters, mains pickup, and the ESP32 ADC's noise and non-linearity (`ml/ad8232_sim.py`). This estimates robustness only.

| Condition | R-peak sensitivity | PPV | Beats analysed | CNN accuracy |
|---|---:|---:|---:|---:|
| Ideal | 99.28% | 99.91% | 95.6% | 69.1% |
| Typical AD8232 + ESP32 | 99.30% | 99.91% | 91.7% | 68.6% |
| Noisy ADC, strong mains | 99.29% | 99.91% | 91.6% | 68.5% |
| Gain 30% lower than assumed | 99.27% | 99.91% | 96.6% | 66.9% |

## Emulated ESP32 self-test

`tools/run_qemu_selftest.py` boots the real Xtensa firmware binary in Espressif's QEMU.

| Check | Result |
|---|---|
| Boot, TFLM load | Pass, 3,652-byte tensor arena |
| Random Forest vs Python, 240 test beats | 240/240 identical |
| CNN class vs Python, 240 test beats | 240/240 identical, largest score difference 6/255 |
| Full pipeline on 6 demo clips | Detected and analysed beat counts identical to the PC, CNN class counts differ by at most 1 beat |

Emulation found two bugs that compiling could not: the demo samples lost their sign, and TF 2.20's per-channel Dense weights were silently misread by this TFLite Micro version.

## Demo clip behaviour

Demo clips are 60 s stretches from test patients. Their verdicts are honest and not all correct:

| Clip | CNN verdict | RF verdict | Note |
|---|---|---|---|
| Normal | Normal | Normal | Correct, no alert |
| PVC | PVC | AFib | CNN correct. Only 39% of beats in this stretch are PVCs. |
| AFib | AFib | AFib | Correct at the end of the clip, but the alert barely fires: most beats read Normal |
| LBBB | **Normal** | **Normal** | Wrong |
| RBBB | **Normal** | RBBB | CNN wrong, RF correct |
| Noise | Normal outside the burst | Normal | Noise burst is gated, 25 of 79 beats analysed |

## How the numbers were obtained, including two evaluation rounds

The test set was evaluated twice, and that should be disclosed in any report.

1. **First run:** MIT-BIH DS2 accuracy 61.2% (CNN) and 60.4% (RF), macro-F1 0.474 and 0.473. AFib recall 18% (CNN). About 50 false alarms per hour.
2. **Diagnosis, no further test-set access:** the confusion matrices showed inflated class priors and database confounding. Further work used patient-grouped cross-validation on the training pool only.
3. **Second run, reported above:** DS2 accuracy 69.1% and 65.0%, macro-F1 0.471 and 0.480.

Because the first run informed the redesign, the final figures are not a pure first-look estimate and may be slightly optimistic. The second round was also a trade-off rather than a pure gain:

| Measure (CNN) | First run | Final |
|---|---:|---:|
| DS2 accuracy | 61.2% | 69.1% |
| DS2 AFib recall | 18% | 76% |
| DS2 macro-F1 | 0.474 | 0.471 |
| DS2 LBBB recall | 21% | 4% |
| PTB-XL accuracy | 84.3% | 79.6% |
| PTB-XL LBBB recall | 79% | 64% |

First-run results are preserved in `results/evaluation_first_run.json`.

## Why LBBB, RBBB and Other are weak

- **Lead:** bundle branch blocks are told apart mainly in chest lead V1. A 3-electrode AD8232 approximates lead II, where they look alike, and differ a lot from person to person.
- **Data:** MIT-BIH has very few patients with these conditions. PTB-XL adds hundreds, but its recordings last only 10 s, in a different hospital setting.
- **Other** mixes many unrelated rare beats, each with little data. Atrial premature beats look like normal beats with early timing, which the AF rule can also mistake for irregularity.

Suggested wording for the viva: AFib, PVC and Normal are the supported patterns. LBBB and RBBB are experimental, as the proposal allowed ("depending on dataset availability").

## What has not been tested

- **No real AD8232, electrodes, OLED, buzzer, button or Wi-Fi.** Firmware for those parts compiles, but has never run. QEMU does not emulate them.
- **No real inference timing** on an ESP32.
- **No real human ECG from this hardware.** Objective 10 (experimental signals) is still open. Domain shift between hospital recordings and a dry-lab AD8232 may lower accuracy further than the simulation suggests.
- **The dashboard** was tested against a fake device in Node (`tests/dashboard_check.js`), not in a browser on a phone.
