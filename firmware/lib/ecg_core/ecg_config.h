// Shared constants for the ECG processing core.
// This code is the single source of truth: the ESP32 firmware runs it live and
// the Python training pipeline calls the very same code through a shared library,
// so the models are trained on exactly what the device will compute.
#pragma once
#include <stdint.h>

namespace ecg {

constexpr int FS = 250;              // Sampling rate, Hz

// Beat window: 0.24 s before the R peak to 0.40 s after it.
constexpr int WAVE_PRE = 60;
constexpr int WAVE_POST = 100;
constexpr int WAVE_LEN = WAVE_PRE + WAVE_POST;

constexpr int N_CTX = 13;            // Rhythm / context features (CNN second input, RF)
constexpr int N_MORPH = 32;          // Hand-crafted morphology features (RF only)
constexpr int N_FEAT = N_CTX + N_MORPH;

constexpr int EMIT_DELAY = 2 * FS;   // Hold a beat 2 s so SQA can see the signal around it
constexpr int RR_TIMEOUT = 5 * FS / 2;  // Give up waiting for the next beat after 2.5 s
constexpr int MIN_RR_HISTORY = 4;    // RR intervals needed before a beat may be classified
constexpr int RR_WINDOW = 16;        // RR intervals used for the rhythm statistics

constexpr int SQA_WINDOW = 5 * FS;   // SQA looks at the last 5 s ...
constexpr int SQA_HOP = FS / 2;      // ... and re-evaluates every 0.5 s

enum Class : uint8_t { NORMAL = 0, PVC, AFIB, LBBB, RBBB, OTHER, N_CLASSES };

inline const char* className(int c) {
  static const char* const names[] = {"Normal", "PVC", "AFib", "LBBB", "RBBB", "Other"};
  return (c >= 0 && c < N_CLASSES) ? names[c] : "?";
}

}  // namespace ecg
