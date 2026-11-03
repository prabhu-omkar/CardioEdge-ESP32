// C interface used by the Python pipeline (ctypes). Not compiled into the firmware.
#ifdef ECG_BUILD_SHARED
#include "ecg_core.h"
#include "../../src/rf_predict.h"
using namespace ecg;

extern "C" {

__declspec(dllexport) int ecg_beat_size() { return (int)sizeof(Beat); }
__declspec(dllexport) int ecg_sqa_size() { return (int)sizeof(SqaStatus); }

// Runs a whole recording through a fresh core. Returns the number of beats written.
// sq receives one SqaStatus per SQA_HOP samples when it is not null.
__declspec(dllexport) int ecg_run(const float* mv, int n, const uint8_t* leads_off, Beat* out, int max_out,
                                  SqaStatus* sq, int max_sq, float* filtered, int bypass_sqa) {
  EcgCore* core = new EcgCore();
  core->bypassSqa(bypass_sqa != 0);
  int nb = 0, ns = 0;
  for (int i = 0; i < n; i++) {
    const bool ready = core->push(mv[i], leads_off ? leads_off[i] != 0 : false);
    if (filtered) filtered[i] = core->filtered();
    if (core->sqaUpdated() && sq && ns < max_sq) sq[ns++] = core->sqa();
    if (ready && nb < max_out) out[nb++] = core->beat();
  }
  delete core;
  return nb;
}

// hr, mean NN, SDNN, RMSSD, pNN50 after pushing n RR intervals
__declspec(dllexport) void ecg_hrv(const float* rr, int n, float* out) {
  HrvTracker t;
  for (int i = 0; i < n; i++) t.push(rr[i]);
  const HrvStats s = t.stats();
  out[0] = s.hr_bpm; out[1] = s.mean_nn_ms; out[2] = s.sdnn_ms; out[3] = s.rmssd_ms; out[4] = s.pnn50;
}

// Stable verdict and alert after each beat class
__declspec(dllexport) void ecg_rhythm(const int8_t* cls, int n, int8_t* label, uint8_t* alert) {
  RhythmTracker t;
  t.reset();
  for (int i = 0; i < n; i++) {
    const RhythmStatus s = t.push(cls[i]);
    label[i] = (int8_t)s.label;
    alert[i] = s.alert;
  }
}

__declspec(dllexport) void ecg_rhythm_rule(int8_t* cls, const float* ctx, int n) {
  for (int i = 0; i < n; i++) cls[i] = (int8_t)rhythmRule(cls[i], ctx + i * N_CTX);
}

__declspec(dllexport) int ecg_rf_nodes() { return RF_N_NODES; }

__declspec(dllexport) void ecg_rf_predict(const float* features, int n, int8_t* out) {
  for (int i = 0; i < n; i++) out[i] = (int8_t)rfVote(features + i * N_FEAT);
}

}
#endif
