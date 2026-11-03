// Streaming ECG pipeline: filtering -> QRS detection -> signal quality assessment
// -> per-beat feature extraction. Portable C++ with no heap use and no Arduino dependency.
#pragma once
#include <stdint.h>
#include "ecg_config.h"

namespace ecg {

struct Biquad {
  float b0, b1, b2, a1, a2;
  float z1 = 0, z2 = 0;
  float step(float x) {  // Transposed direct form II
    const float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};

// Pan-Tompkins QRS detector with adaptive thresholds, T-wave rejection and search-back.
class QrsDetector {
 public:
  void reset();
  // Feed one filtered sample with absolute index n.
  // Returns true when a QRS was accepted; qrsIndex() is the integrator peak.
  bool step(float x, int32_t n);
  int32_t qrsIndex() const { return accepted_idx_; }

 private:
  static constexpr int MWI_LEN = 38;      // 150 ms
  static constexpr int REFRACTORY = 50;   // 200 ms
  static constexpr int TWAVE_LIMIT = 90;  // 360 ms
  static constexpr int LEARN = 2 * FS;

  struct Candidate { float val; int32_t idx; float slope; };
  void evalPeak(const Candidate& c);
  void accept(const Candidate& c, bool searchback);
  void noisePeak(const Candidate& c);
  void updateThresholds();

  Biquad bp_[2];
  float dly_[4];
  float sq_[MWI_LEN];
  float deriv_[64];
  int32_t count_;
  float cand_val_; int32_t cand_idx_; float cand_slope_;
  float spki_, npki_, thr1_, thr2_;
  float learn_max_, learn_sum_; int32_t learn_n_;
  bool learned_;
  int32_t last_qrs_; float last_slope_;
  Candidate best_;
  int32_t rr_[8]; int rr_n_, rr_i_;
  int32_t accepted_idx_; bool accepted_;
  int32_t last_decay_;
};

struct SqaStatus {
  bool good = false;       // Overall verdict (with hysteresis)
  bool leads_off = false;
  bool amp_ok = false;     // Amplitude check
  bool snr_ok = false;     // Noise level (SNR)
  bool base_ok = false;    // Baseline stability
  bool morph_ok = false;   // Morphology consistency
  float amp_mv = 0;        // Median QRS peak-to-peak, mV
  float snr_db = 0;
  float base_mv = 0;       // Largest baseline step, mV
  float morph = 0;         // Fraction of beats resembling another beat
  float kurtosis = 0;      // Peakiness of the signal, about 3 for pure noise
};

// Beat flag bits
enum BeatFlags : uint8_t {
  BEAT_READY = 1,      // Enough rhythm history and a template exist
  BEAT_GOOD_SQ = 2,    // Signal quality was good around this beat
};

struct Beat {
  int32_t r_index;          // Sample index of the R peak in the filtered signal
  uint8_t flags;
  float rr_pre_ms;
  float rr_post_ms;
  float ctx[N_CTX];
  float morph[N_MORPH];
  float wave[WAVE_LEN];     // Normalised beat window, R at index WAVE_PRE
  bool classifiable() const { return (flags & (BEAT_READY | BEAT_GOOD_SQ)) == (BEAT_READY | BEAT_GOOD_SQ); }
};

class EcgCore {
 public:
  EcgCore() { reset(); }
  void reset();
  // For the A/B evaluation only: treat every signal as good, as if SQA did not exist.
  void bypassSqa(bool on) { bypass_sqa_ = on; }

  // Push one sample in mV. Returns true if a beat is ready, then read beat().
  bool push(float mv, bool leads_off);

  const Beat& beat() const { return beat_; }
  const SqaStatus& sqa() const { return sqa_; }
  float filtered() const { return last_x_; }
  int32_t samples() const { return n_; }
  // True when SQA was re-evaluated by the last push()
  bool sqaUpdated() const { return sqa_updated_; }
  // Milliseconds since the last detected QRS (for no-signal warnings)
  int32_t msSinceBeat() const;

 private:
  static constexpr int RING = 2048;     // 8.2 s of filtered signal
  static constexpr int R_RING = 64;
  static constexpr int HIST = 8;

  float at(int32_t i) const { return ring_[i & (RING - 1)]; }
  void onQrs(int32_t mwiIdx);
  void evalSqa();
  bool tryEmit();
  void buildBeat(int32_t idx);
  void updateTemplate(const float* w, float amp, float* corr, float* ampRel);
  void rhythmFeatures(int32_t idx, float* ctx);
  void morphFeatures(const float* w, float* m);

  Biquad hp_, notch_, lp_[2];
  QrsDetector qrs_;
  float dc_; bool seeded_;
  float ring_[RING];
  float last_x_, last_mv_;
  int32_t n_;
  int32_t r_[R_RING]; int32_t r_count_; int32_t emit_;
  int32_t good_since_;     // RR statistics may only use beats after this sample
  int32_t last_bad_;       // Last sample at which quality was bad
  // Baseline: mean of the removed low-frequency part in 1 s blocks
  float base_acc_; int base_n_; float base_hist_[6]; int base_count_;
  int good_streak_;
  SqaStatus sqa_;
  bool sqa_updated_;
  bool bypass_sqa_ = false;
  Beat beat_;
  // Dominant beat template (patient-adaptive reference)
  float tpl_[WAVE_LEN]; float tpl_amp_; bool have_tpl_; int lost_;
  float hist_[HIST][WAVE_LEN]; float hist_amp_[HIST]; int hist_n_, hist_i_;
};

// AF is a property of the rhythm rather than of one beat's shape, so the final AF decision
// comes from RR irregularity. Wide or ectopic beats keep the class the classifier gave them.
int rhythmRule(int cls, const float* ctx);

// HR and HRV from normal-to-normal intervals over the last 60 intervals
struct HrvStats {
  float hr_bpm = 0, mean_nn_ms = 0, sdnn_ms = 0, rmssd_ms = 0, pnn50 = 0;
  int n = 0;
};

class HrvTracker {
 public:
  void reset() { n_ = i_ = 0; }
  void push(float rr_ms);
  HrvStats stats() const;
 private:
  static constexpr int CAP = 60;
  float rr_[CAP]; int n_ = 0, i_ = 0;
};

// Turns per-beat classes into a stable rhythm verdict and an alert decision.
struct RhythmStatus {
  int label = -1;       // Class index, -1 until enough beats were seen
  bool alert = false;
  int pvc_count = 0;    // PVCs in the last 10 beats
};

class RhythmTracker {
 public:
  void reset();
  RhythmStatus push(int cls);
  RhythmStatus current() const { return status_; }
 private:
  static constexpr int LEN = 10;
  static constexpr int ALERT_HOLD = 8;  // Beats an abnormal verdict must last before alerting
  uint8_t c_[LEN]; int n_ = 0, i_ = 0;
  int abnormal_run_ = 0;
  RhythmStatus status_;
};

}  // namespace ecg
