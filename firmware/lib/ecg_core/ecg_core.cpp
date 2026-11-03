#include "ecg_core.h"
#include <math.h>
#include <string.h>
#include <algorithm>

namespace ecg {

// Filter designs at 250 Hz (scipy.signal.butter / iirnotch)
static const Biquad HP_05 = {0.9911535951f, -1.9823071902f, 0.9911535951f, -1.9822289298f, 0.9823854506f};
static const Biquad NOTCH_50 = {0.9794827610f, -0.6053536377f, 0.9794827610f, -0.6053536377f, 0.9589655220f};
static const Biquad LP_40[2] = {
    {0.0228702077f, 0.0457404154f, 0.0228702077f, -0.6020332023f, 0.1235593440f},
    {1.0000000000f, 2.0000000000f, 1.0000000000f, -0.8099502989f, 0.5115897647f}};
static const Biquad BP_5_15[2] = {
    {0.0133592000f, 0.0267184001f, 0.0133592000f, -1.6813538570f, 0.7798921440f},
    {1.0000000000f, -2.0000000000f, 1.0000000000f, -1.8795935959f, 0.8987098878f}};

// SQA decision thresholds, calibrated by ml/sqa_eval.py
static constexpr float SQA_AMP_MIN_MV = 0.12f;
static constexpr float SQA_AMP_MAX_MV = 6.0f;
static constexpr float SQA_SNR_MIN_DB = 12.0f;
static constexpr float SQA_KURTOSIS_MIN = 3.6f;
static constexpr float SQA_BASE_MAX_MV = 1.2f;
static constexpr float SQA_MORPH_MIN = 0.50f;
static constexpr float SQA_MORPH_CORR = 0.70f;
static constexpr int SQA_MIN_BEATS = 2;
static constexpr int SQA_MAX_BEATS = 20;   // 240 bpm over the 5 s window
static constexpr int SQA_GOOD_HOLD = 2;    // Good evaluations needed to leave the bad state

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static float pearson(const float* a, const float* b, int n) {
  float ma = 0, mb = 0;
  for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  float sab = 0, saa = 0, sbb = 0;
  for (int i = 0; i < n; i++) {
    const float da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  const float d = sqrtf(saa * sbb);
  return d > 1e-12f ? sab / d : 0.0f;
}

// ---------------------------------------------------------------- QRS detector

void QrsDetector::reset() {
  bp_[0] = BP_5_15[0]; bp_[1] = BP_5_15[1];
  memset(dly_, 0, sizeof(dly_));
  memset(sq_, 0, sizeof(sq_));
  memset(deriv_, 0, sizeof(deriv_));
  count_ = 0;
  cand_val_ = 0; cand_idx_ = 0; cand_slope_ = 0;
  spki_ = npki_ = thr1_ = thr2_ = 0;
  learn_max_ = learn_sum_ = 0; learn_n_ = 0;
  learned_ = false;
  last_qrs_ = -1; last_slope_ = 0;
  best_ = {0, 0, 0};
  rr_n_ = rr_i_ = 0;
  accepted_idx_ = 0; accepted_ = false;
  last_decay_ = 0;
}

void QrsDetector::updateThresholds() {
  thr1_ = npki_ + 0.25f * (spki_ - npki_);
  thr2_ = 0.5f * thr1_;
}

void QrsDetector::noisePeak(const Candidate& c) {
  npki_ = 0.125f * c.val + 0.875f * npki_;
  updateThresholds();
}

void QrsDetector::accept(const Candidate& c, bool searchback) {
  const float w = searchback ? 0.25f : 0.125f;
  // Limit how far one tall ectopic beat can raise the threshold above the smaller beats
  spki_ = w * std::min(c.val, 2.0f * spki_) + (1.0f - w) * spki_;
  updateThresholds();
  if (last_qrs_ >= 0) {
    rr_[rr_i_] = c.idx - last_qrs_;
    rr_i_ = (rr_i_ + 1) % 8;
    if (rr_n_ < 8) rr_n_++;
  }
  last_qrs_ = c.idx;
  last_slope_ = c.slope;
  best_ = {0, 0, 0};
  accepted_ = true;
  accepted_idx_ = c.idx;
}

void QrsDetector::evalPeak(const Candidate& c) {
  const int32_t since = last_qrs_ >= 0 ? c.idx - last_qrs_ : INT32_MAX;
  if (since < REFRACTORY) { noisePeak(c); return; }
  if (c.val > thr1_) {
    if (since < TWAVE_LIMIT && c.slope < 0.5f * last_slope_) { noisePeak(c); return; }  // T wave
    accept(c, false);
  } else {
    noisePeak(c);
    if (c.val > best_.val) best_ = c;
  }
}

bool QrsDetector::step(float x, int32_t n) {
  accepted_ = false;
  const float d = bp_[1].step(bp_[0].step(x));
  const float y = (2 * d + dly_[0] - dly_[2] - 2 * dly_[3]) * 0.125f;
  dly_[3] = dly_[2]; dly_[2] = dly_[1]; dly_[1] = dly_[0]; dly_[0] = d;

  deriv_[n & 63] = fabsf(y);
  sq_[n % MWI_LEN] = y * y;
  float m = 0;
  for (int i = 0; i < MWI_LEN; i++) m += sq_[i];
  m /= MWI_LEN;
  count_++;

  if (!learned_) {
    if (count_ > FS / 2) {  // Skip the filter start-up transient
      learn_max_ = std::max(learn_max_, m);
      learn_sum_ += m; learn_n_++;
    }
    if (count_ >= FS / 2 + LEARN) {
      spki_ = 0.5f * learn_max_;
      npki_ = 0.5f * learn_sum_ / learn_n_;
      updateThresholds();
      learned_ = true;
      last_decay_ = n;
    }
    return false;
  }

  // Peak tracking on the integrator output: a peak is complete once the signal has
  // dropped to half its height, or has not risen for 200 ms.
  if (m > cand_val_) {
    cand_val_ = m; cand_idx_ = n;
    float s = 0;
    for (int i = 0; i < MWI_LEN; i++) s = std::max(s, deriv_[(n - i) & 63]);
    cand_slope_ = s;
  } else if (cand_val_ > 0 && (m < 0.5f * cand_val_ || n - cand_idx_ > REFRACTORY)) {
    evalPeak({cand_val_, cand_idx_, cand_slope_});
    cand_val_ = 0;
  }

  // Search back for a missed beat after 1.66 average RR intervals
  if (!accepted_ && last_qrs_ >= 0 && rr_n_ > 0) {
    int32_t sum = 0;
    for (int i = 0; i < rr_n_; i++) sum += rr_[i];
    const int32_t limit = (sum * 166) / (100 * rr_n_);
    if (n - last_qrs_ > limit && best_.val > thr2_ && best_.idx - last_qrs_ >= REFRACTORY)
      accept(best_, true);
  }

  // No beat for a while: the signal level has probably fallen, so relax the thresholds
  if (!accepted_ && n - std::max(last_qrs_, last_decay_) > 2 * FS) {
    spki_ *= 0.5f; npki_ *= 0.5f;
    updateThresholds();
    last_decay_ = n;
  }
  return accepted_;
}

// ---------------------------------------------------------------- Core

void EcgCore::reset() {
  hp_ = HP_05; notch_ = NOTCH_50; lp_[0] = LP_40[0]; lp_[1] = LP_40[1];
  qrs_.reset();
  dc_ = 0; seeded_ = false;
  memset(ring_, 0, sizeof(ring_));
  last_x_ = last_mv_ = 0;
  n_ = 0;
  r_count_ = 0; emit_ = 0;
  good_since_ = 0; last_bad_ = 0;
  base_acc_ = 0; base_n_ = 0; base_count_ = 0;
  memset(base_hist_, 0, sizeof(base_hist_));
  good_streak_ = 0;
  sqa_ = SqaStatus();
  sqa_updated_ = false;
  have_tpl_ = false; lost_ = 0; tpl_amp_ = 1;
  hist_n_ = hist_i_ = 0;
}

int32_t EcgCore::msSinceBeat() const {
  if (r_count_ == 0) return n_ * 4;
  return (n_ - r_[(r_count_ - 1) % R_RING]) * 4;
}

bool EcgCore::push(float mv, bool leads_off) {
  if (!seeded_) { dc_ = mv; seeded_ = true; last_mv_ = mv; }
  // Hold the last good value while the leads are off, so the rail-to-rail step
  // does not ring through the high-pass filter for several seconds.
  if (leads_off) mv = last_mv_; else last_mv_ = mv;
  sqa_.leads_off = leads_off;

  const float in = mv - dc_;
  const float h = hp_.step(in);
  const float x = lp_[1].step(lp_[0].step(notch_.step(h)));
  last_x_ = x;
  const int32_t n = n_;
  ring_[n & (RING - 1)] = x;

  base_acc_ += in - h;
  if (++base_n_ == FS) {
    memmove(base_hist_, base_hist_ + 1, sizeof(float) * 5);
    base_hist_[5] = base_acc_ / FS;
    if (base_count_ < 6) base_count_++;
    base_acc_ = 0; base_n_ = 0;
  }

  if (qrs_.step(x, n)) onQrs(qrs_.qrsIndex());

  sqa_updated_ = false;
  n_ = n + 1;
  if (n_ % SQA_HOP == 0) { evalSqa(); sqa_updated_ = true; }
  if (!sqa_.good || leads_off) last_bad_ = n_;
  return tryEmit();
}

void EcgCore::onQrs(int32_t mwiIdx) {
  // The integrator peak lags the R wave; find the true extremum of the filtered signal.
  const int32_t lo = mwiIdx - 50, hi = mwiIdx - 8;
  if (lo < 0) return;
  float mean = 0;
  for (int32_t i = lo; i <= hi; i++) mean += at(i);
  mean /= (hi - lo + 1);
  float best = -1; int32_t r = lo;
  for (int32_t i = lo; i <= hi; i++) {
    const float a = fabsf(at(i) - mean);
    if (a > best) { best = a; r = i; }
  }
  if (r_count_ > 0 && r - r_[(r_count_ - 1) % R_RING] < 40) return;  // Same beat
  r_[r_count_ % R_RING] = r;
  r_count_++;
}

void EcgCore::evalSqa() {
  const int32_t end = n_;
  const int32_t start = std::max<int32_t>(0, end - SQA_WINDOW);
  SqaStatus s;
  s.leads_off = sqa_.leads_off;

  // Collect beats inside the window that have their whole QRS available
  int32_t rs[SQA_MAX_BEATS + 2];
  int nb = 0;
  bool too_many = false;
  for (int32_t k = r_count_ - 1; k >= 0 && k > r_count_ - R_RING; k--) {
    const int32_t r = r_[k % R_RING];
    if (r < start + 12) break;
    if (r + 22 >= end) continue;
    if (nb >= SQA_MAX_BEATS + 1) { too_many = true; break; }
    rs[nb++] = r;
  }
  std::reverse(rs, rs + nb);

  // Amplitude: median QRS peak-to-peak
  float amps[SQA_MAX_BEATS + 2];
  for (int i = 0; i < nb; i++) {
    float lo = 1e9f, hi = -1e9f;
    for (int32_t j = rs[i] - 12; j <= rs[i] + 15; j++) { lo = std::min(lo, at(j)); hi = std::max(hi, at(j)); }
    amps[i] = hi - lo;
  }
  if (nb > 0) {
    std::nth_element(amps, amps + nb / 2, amps + nb);
    s.amp_mv = amps[nb / 2];
  }
  s.amp_ok = !too_many && nb >= SQA_MIN_BEATS && s.amp_mv >= SQA_AMP_MIN_MV && s.amp_mv <= SQA_AMP_MAX_MV;

  // Noise: RMS level of the TP segments, the electrically quiet gaps between the end of the
  // T wave and the next QRS. A clean ECG is nearly flat here, so whatever remains is noise.
  // Using the second-worst gap instead of the average means a burst is noticed after two
  // gaps, rather than being diluted across the whole window.
  float gap_rms[SQA_MAX_BEATS + 2];
  int gaps = 0;
  for (int i = 0; i + 1 < nb; i++) {
    const int32_t rr = rs[i + 1] - rs[i];
    const int32_t from = rs[i] + std::min<int32_t>(88, rr * 6 / 10);
    const int32_t to = rs[i + 1] - 15;
    if (to - from < 5) continue;
    float acc = 0;
    for (int32_t j = from; j < to; j++) acc += at(j) * at(j);
    gap_rms[gaps++] = sqrtf(acc / (to - from));
  }
  float noise = 0;
  if (gaps >= 2) {
    std::nth_element(gap_rms, gap_rms + gaps - 2, gap_rms + gaps);
    noise = gap_rms[gaps - 2];
  } else if (gaps == 1) {
    noise = gap_rms[0];
  }
  s.snr_db = (noise > 1e-6f && s.amp_mv > 0) ? 20.0f * log10f(s.amp_mv / noise) : 0.0f;

  // Kurtosis: an ECG concentrates its energy in sharp QRS peaks, while noise that looks
  // like beats after filtering stays close to the Gaussian value of 3.
  float m1 = 0;
  for (int32_t i = start; i < end; i++) m1 += at(i);
  m1 /= (end - start);
  float m2 = 0, m4 = 0;
  for (int32_t i = start; i < end; i++) {
    const float d = at(i) - m1, d2 = d * d;
    m2 += d2; m4 += d2 * d2;
  }
  m2 /= (end - start); m4 /= (end - start);
  s.kurtosis = m2 > 1e-9f ? m4 / (m2 * m2) : 0.0f;
  s.snr_ok = s.snr_db >= SQA_SNR_MIN_DB && s.kurtosis >= SQA_KURTOSIS_MIN;

  // Baseline: largest step between consecutive 1 s means
  float step = 0;
  for (int i = 6 - base_count_ + 1; i < 6; i++) step = std::max(step, fabsf(base_hist_[i] - base_hist_[i - 1]));
  s.base_mv = step;
  s.base_ok = step <= SQA_BASE_MAX_MV;

  // Morphology: a genuine beat resembles some other beat in the window. Comparing against
  // every beat, not just the last few, keeps isolated or recurring ectopic beats from being
  // mistaken for noise, while artefacts detected as beats match nothing. Each comparison
  // allows a small timing shift between the two R-peak positions.
  int match = 0, tested = 0;
  static constexpr int M_LEN = 36, M_SHIFT = 2;
  for (int i = 0; i < nb && nb >= 2; i++) {
    float a[M_LEN], b[M_LEN];
    for (int j = 0; j < M_LEN; j++) a[j] = at(rs[i] - 12 + j);
    bool found = false;
    for (int k = 0; k < nb && !found; k++) {
      if (k == i) continue;
      for (int shift = -M_SHIFT; shift <= M_SHIFT && !found; shift++) {
        for (int j = 0; j < M_LEN; j++) b[j] = at(rs[k] - 12 + shift + j);
        found = pearson(a, b, M_LEN) >= SQA_MORPH_CORR;
      }
    }
    tested++;
    if (found) match++;
  }
  s.morph = tested > 0 ? (float)match / tested : 0.0f;
  s.morph_ok = s.morph >= SQA_MORPH_MIN;

  const bool now_good = !s.leads_off && s.amp_ok && s.snr_ok && s.base_ok && s.morph_ok;
  if (now_good) {
    if (good_streak_ < SQA_GOOD_HOLD) good_streak_++;
  } else {
    good_streak_ = 0;
  }
  s.good = bypass_sqa_ || good_streak_ >= SQA_GOOD_HOLD;
  if (!s.good) good_since_ = end;
  sqa_ = s;
}

bool EcgCore::tryEmit() {
  if (emit_ >= r_count_) return false;
  if (r_count_ - emit_ >= R_RING - 4) emit_ = r_count_ - R_RING + 4;  // Unreachable at physiological rates
  const int32_t r = r_[emit_ % R_RING];
  if (r - WAVE_PRE < 0) { emit_++; return false; }
  const int32_t age = n_ - 1 - r;
  const bool have_next = emit_ + 1 < r_count_;
  if (age < EMIT_DELAY || age < WAVE_POST + 1) return false;
  if (!have_next && age < RR_TIMEOUT) return false;
  buildBeat(emit_);
  emit_++;
  return true;
}

void EcgCore::rhythmFeatures(int32_t idx, float* ctx) {
  const int32_t r = r_[idx % R_RING];
  const bool has_prev = idx > 0 && idx > r_count_ - R_RING;
  const bool has_next = idx + 1 < r_count_;
  const float rr_pre = has_prev ? (float)(r - r_[(idx - 1) % R_RING]) / FS : 2.5f;
  const float rr_post = has_next ? (float)(r_[(idx + 1) % R_RING] - r) / FS : 2.5f;

  // RR intervals up to and including the one after this beat, inside the good segment
  float rr[RR_WINDOW]; int nr = 0;
  const int32_t last = has_next ? idx + 1 : idx;
  for (int32_t k = last; k > 0 && nr < RR_WINDOW && k > r_count_ - R_RING + 1; k--) {
    const int32_t a = r_[(k - 1) % R_RING], b = r_[k % R_RING];
    if (a < good_since_) break;
    rr[nr++] = (float)(b - a) / FS;
  }

  float mean = rr_pre, cv = 0, rmssd = 0, prr = 0, tpr = 0, ent = 0, mad = 0, med = rr_pre;
  if (nr >= 2) {
    mean = 0;
    for (int i = 0; i < nr; i++) mean += rr[i];
    mean /= nr;
    float var = 0, dsum = 0; int big = 0;
    for (int i = 0; i < nr; i++) var += (rr[i] - mean) * (rr[i] - mean);
    for (int i = 1; i < nr; i++) {
      const float d = rr[i] - rr[i - 1];
      dsum += d * d;
      if (fabsf(d) > 0.125f * mean) big++;
    }
    cv = sqrtf(var / nr) / mean;
    rmssd = sqrtf(dsum / (nr - 1)) / mean;
    prr = (float)big / (nr - 1);

    int turns = 0;
    for (int i = 1; i + 1 < nr; i++)
      if ((rr[i] - rr[i - 1]) * (rr[i + 1] - rr[i]) < 0) turns++;
    tpr = nr > 2 ? (float)turns / (nr - 2) : 0;

    float tmp[RR_WINDOW];
    memcpy(tmp, rr, sizeof(float) * nr);
    std::nth_element(tmp, tmp + nr / 2, tmp + nr);
    med = tmp[nr / 2];
    float dev[RR_WINDOW];
    for (int i = 0; i < nr; i++) dev[i] = fabsf(rr[i] - med);
    std::nth_element(dev, dev + nr / 2, dev + nr);
    mad = dev[nr / 2] / med;

    int hist[8] = {0};  // Spread of RR/median across 8 bins from 0.5 to 1.5
    for (int i = 0; i < nr; i++) {
      const int bin = (int)clampf((rr[i] / med - 0.5f) * 8.0f, 0.0f, 7.0f);
      hist[bin]++;
    }
    for (int i = 0; i < 8; i++)
      if (hist[i] > 0) { const float p = (float)hist[i] / nr; ent -= p * log2f(p); }
    ent /= 3.0f;
  }
  // Everything here is relative to this person's own rhythm. Absolute RR values are left
  // out on purpose: AF patients tend to be faster, so a model given the raw heart rate
  // learns that a fast normal rhythm means AF.
  ctx[0] = clampf(rr_pre / med, 0, 3);
  ctx[1] = clampf(rr_post / med, 0, 3);
  ctx[2] = clampf(rr_pre / mean, 0, 3);
  ctx[3] = clampf(rr_post / mean, 0, 3);
  ctx[4] = clampf(mean / med, 0, 3);
  ctx[5] = clampf(cv, 0, 1);
  ctx[6] = clampf(rmssd, 0, 2);
  ctx[7] = prr;
  ctx[8] = tpr;
  ctx[9] = ent;
  ctx[10] = clampf(mad, 0, 1);
}

void EcgCore::updateTemplate(const float* w, float amp, float* corr, float* ampRel) {
  *corr = 1.0f; *ampRel = 1.0f;
  if (have_tpl_) {
    *corr = pearson(w, tpl_, WAVE_LEN);
    *ampRel = clampf(amp / std::max(tpl_amp_, 1e-3f), 0, 3);
    if (*corr > 0.8f) {
      for (int i = 0; i < WAVE_LEN; i++) tpl_[i] += 0.03f * (w[i] - tpl_[i]);
      tpl_amp_ += 0.03f * (amp - tpl_amp_);
      lost_ = 0;
    } else {
      lost_++;
    }
  }
  memcpy(hist_[hist_i_], w, sizeof(float) * WAVE_LEN);
  hist_amp_[hist_i_] = amp;
  hist_i_ = (hist_i_ + 1) % HIST;
  if (hist_n_ < HIST) hist_n_++;

  // The dominant beat is the one most similar to all the others in the recent history
  if ((!have_tpl_ && hist_n_ == HIST) || lost_ >= 24) {
    int medoid = 0; float best = -1e9f;
    for (int i = 0; i < HIST; i++) {
      float sum = 0;
      for (int j = 0; j < HIST; j++)
        if (j != i) sum += pearson(hist_[i], hist_[j], WAVE_LEN);
      if (sum > best) { best = sum; medoid = i; }
    }
    memcpy(tpl_, hist_[medoid], sizeof(tpl_));
    tpl_amp_ = hist_amp_[medoid];
    have_tpl_ = true;
    lost_ = 0;
  }
}

void EcgCore::morphFeatures(const float* w, float* m) {
  int k = 0;
  const auto meanOf = [&](int a, int b) {
    float s = 0;
    for (int i = a; i < b; i++) s += w[i];
    return s / (b - a);
  };
  for (int i = 0; i < 6; i++) m[k++] = meanOf(i * 8, (i + 1) * 8);       // P wave region
  for (int i = 0; i < 10; i++) m[k++] = meanOf(48 + i * 3, 51 + i * 3);  // QRS region
  for (int i = 0; i < 8; i++) m[k++] = meanOf(78 + i * 10, 88 + i * 10); // ST-T region

  // QRS width: contiguous samples around R above 30 % of full scale
  int a = WAVE_PRE, b = WAVE_PRE;
  while (a > 0 && fabsf(w[a - 1]) > 0.3f) a--;
  while (b < WAVE_LEN - 1 && fabsf(w[b + 1]) > 0.3f) b++;
  float lo = 1e9f, hi = -1e9f, pos = 0, neg = 0, slope = 0;
  for (int i = 40; i < 90; i++) {
    lo = std::min(lo, w[i]); hi = std::max(hi, w[i]);
    if (w[i] > 0) pos += w[i]; else neg -= w[i];
    slope = std::max(slope, fabsf(w[i + 1] - w[i]));
  }
  float pm = 0, pv = 0;
  for (int i = 10; i < 45; i++) pm += w[i];
  pm /= 35;
  for (int i = 10; i < 45; i++) pv += (w[i] - pm) * (w[i] - pm);
  m[k++] = (float)(b - a + 1) / FS;
  m[k++] = w[WAVE_PRE];
  m[k++] = lo;
  m[k++] = hi;
  m[k++] = pos / 50;
  m[k++] = neg / 50;
  m[k++] = slope;
  m[k++] = sqrtf(pv / 35);
}

void EcgCore::buildBeat(int32_t idx) {
  const int32_t r = r_[idx % R_RING];
  Beat& b = beat_;
  b.r_index = r;
  b.flags = 0;
  const bool has_prev = idx > 0 && idx > r_count_ - R_RING;
  const bool has_next = idx + 1 < r_count_;
  b.rr_pre_ms = has_prev ? (r - r_[(idx - 1) % R_RING]) * 4.0f : 0;
  b.rr_post_ms = has_next ? (r_[(idx + 1) % R_RING] - r) * 4.0f : 0;

  // Normalise out amplitude and offset so the device gain does not matter
  float tmp[WAVE_LEN];
  for (int i = 0; i < WAVE_LEN; i++) { b.wave[i] = at(r - WAVE_PRE + i); tmp[i] = b.wave[i]; }
  std::nth_element(tmp, tmp + WAVE_LEN / 2, tmp + WAVE_LEN);
  const float med = tmp[WAVE_LEN / 2];
  float peak = 1e-6f;
  for (int i = 0; i < WAVE_LEN; i++) { b.wave[i] -= med; peak = std::max(peak, fabsf(b.wave[i])); }
  for (int i = 0; i < WAVE_LEN; i++) b.wave[i] /= peak;

  rhythmFeatures(idx, b.ctx);
  morphFeatures(b.wave, b.morph);

  const bool sq_good = sqa_.good && r - WAVE_PRE > last_bad_;
  if (sq_good) updateTemplate(b.wave, peak, &b.ctx[11], &b.ctx[12]);
  else { b.ctx[11] = 1.0f; b.ctx[12] = 1.0f; }

  // Count the RR intervals available inside the current good segment
  int rr_good = 0;
  for (int32_t k = idx; k > 0 && rr_good < MIN_RR_HISTORY && k > r_count_ - R_RING + 1; k--) {
    if (r_[(k - 1) % R_RING] < good_since_) break;
    rr_good++;
  }
  if (have_tpl_ && rr_good >= MIN_RR_HISTORY && has_next) b.flags |= BEAT_READY;
  if (sq_good) b.flags |= BEAT_GOOD_SQ;
}

// ---------------------------------------------------------------- HRV and rhythm

void HrvTracker::push(float rr_ms) {
  if (rr_ms < 250 || rr_ms > 2500) return;
  rr_[i_] = rr_ms;
  i_ = (i_ + 1) % CAP;
  if (n_ < CAP) n_++;
}

HrvStats HrvTracker::stats() const {
  HrvStats s;
  s.n = n_;
  if (n_ < 2) return s;
  // Walk oldest to newest so successive differences are real neighbours
  const int first = n_ < CAP ? 0 : i_;
  float sum = 0, prev = 0, dsum = 0; int nn50 = 0;
  for (int k = 0; k < n_; k++) {
    const float v = rr_[(first + k) % CAP];
    sum += v;
    if (k > 0) {
      const float d = v - prev;
      dsum += d * d;
      if (fabsf(d) > 50.0f) nn50++;
    }
    prev = v;
  }
  const float mean = sum / n_;
  float var = 0;
  for (int k = 0; k < n_; k++) { const float d = rr_[k] - mean; var += d * d; }
  s.mean_nn_ms = mean;
  s.sdnn_ms = sqrtf(var / (n_ - 1));
  s.rmssd_ms = sqrtf(dsum / (n_ - 1));
  s.pnn50 = 100.0f * nn50 / (n_ - 1);
  s.hr_bpm = 60000.0f / mean;
  return s;
}

int rhythmRule(int cls, const float* ctx) {
  // Thresholds from patient-grouped cross-validation on the training pool, see docs/RESULTS.md
  const float prr = ctx[7], mad = ctx[10];
  const bool irregular = mad > 0.08f && prr > 0.30f && prr <= 0.97f;
  if (irregular && (cls == NORMAL || cls == OTHER)) return AFIB;
  if (!irregular && cls == AFIB) return NORMAL;
  return cls;
}

void RhythmTracker::reset() { n_ = i_ = 0; abnormal_run_ = 0; status_ = RhythmStatus(); }

RhythmStatus RhythmTracker::push(int cls) {
  // A failed classifier must clear any pending alarm, never index the class table.
  if (cls < 0 || cls >= N_CLASSES) { reset(); return status_; }
  c_[i_] = (uint8_t)cls;
  i_ = (i_ + 1) % LEN;
  if (n_ < LEN) n_++;
  int counts[N_CLASSES] = {0};
  for (int i = 0; i < n_; i++) counts[c_[i]]++;

  RhythmStatus s;
  s.pvc_count = counts[PVC];
  if (n_ < 5) { status_ = s; return s; }
  const int majority = n_ * 6 / 10;
  if (counts[AFIB] >= majority) s.label = AFIB;
  else if (counts[LBBB] >= majority) s.label = LBBB;
  else if (counts[RBBB] >= majority) s.label = RBBB;
  else if (counts[PVC] >= 3) s.label = PVC;
  else if (counts[OTHER] >= majority) s.label = OTHER;
  else s.label = NORMAL;
  // A buzzer that cries wolf gets ignored, so the verdict has to persist first
  abnormal_run_ = s.label != NORMAL ? abnormal_run_ + 1 : 0;
  s.alert = abnormal_run_ >= ALERT_HOLD;
  status_ = s;
  return s;
}

}  // namespace ecg
