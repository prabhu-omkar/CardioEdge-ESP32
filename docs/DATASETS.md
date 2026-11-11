# Datasets

All five databases are public on PhysioNet. Cite them, and PhysioNet itself, in the project report.

| Database | Used for | Split | Why it is needed |
|---|---|---|---|
| MIT-BIH Arrhythmia | All classes | DS1 train and validation, DS2 test (de Chazal 2004) | Standard benchmark, named in the proposal |
| MIT-BIH Atrial Fibrillation | AFib, Normal | 14 records train, 3 validation, 6 test | MIT-BIH alone has too few AF patients |
| PTB-XL | LBBB, RBBB, Normal, Other (training only) | Folds 1-8 train, 9 validation, 10 test | Hundreds of LBBB and RBBB patients |
| St Petersburg INCART | PVC, Other, RBBB, Normal | Train only | 20,000 PVCs. A patient can have several records, so it never enters a test set. |
| MIT-BIH Noise Stress Test | Noise only | SQA evaluation | Real electrode motion, muscle and baseline noise |

Paced records 102, 104, 107 and 217 are excluded, following AAMI guidance.

## Class mapping

| Annotation | Class |
|---|---|
| N outside AF | Normal |
| V | PVC |
| N during AFIB rhythm | AFib |
| L | LBBB |
| R | RBBB |
| A, a, J, S (supraventricular premature beats) | Other |
| F, f, e, E, j, Q, and N during flutter | Not used (no label) |

The proposal listed Normal, PVC, AFib, LBBB and RBBB, and the diagram listed Normal, PVC, AFib and Other. This project combines them into six classes.

"Other" means supraventricular premature beats only. Until 2026-11 it also held fusion, escape and flutter beats. Those types have little in common, and the class scored F1 0.03-0.10 in patient-grouped cross-validation. Only 36% of the old Other beats were premature. The excluded beats are still matched to detected beats, so their neighbours are not mislabelled. They are dropped from training and scoring.

## PTB-XL selection

`ml/ptbxl_selection.csv` lists the 1,236 records used. A record was chosen if it had a high-confidence single label (CLBBB, CRBBB, or NORM) and no recorded quality problems: no electrode problems, baseline drift, static noise or burst noise. Records with PVCs, pacing or tachyarrhythmias were excluded. All qualifying LBBB and RBBB records are included, with Normal capped at 600.

The selection also has 260 PAC records (254 patients, folds 1-9 only). These are records with a PAC or SVARR statement, no quality problems, and no PVC, AF, flutter, pacing, bundle-branch, WPW, tachycardia or bigeminy statement. PTB-XL labels the whole record, but only one or two beats in it are premature. So only beats whose preceding RR is under 0.85 of the median are labelled Other, and the rest are dropped. These are **weak labels**, used for training and validation only. `ml/dataset.py` keeps them out of the test set, so Other is scored on MIT-BIH beat annotations only. The 28 clean AF records are too few to use and are ignored by `ml/dataset.py`. PTB-XL records last 10 s, so each is looped 4 times and beats come from the third pass, away from start-up and the loop joins.

## Label limitations

- INCART rhythm markers are sparse, so its N beats inside a non-normal rhythm are dropped.
- The AF database has rhythm labels and no beat types, so any PVCs inside it are labelled by rhythm alone.
- PTB-XL labels describe the whole 10 s record and apply to every beat in it.
