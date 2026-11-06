// Snapshot of everything the user interfaces show.
#pragma once
#include "ecg_core.h"

struct LiveState {
  int source = 0;
  float hr = 0, sdnn = 0, rmssd = 0, pnn50 = 0, mean_nn = 0;
  int rhythm = -1, rhythm_rf = -1;   // Stable verdicts from CNN and RF
  int cnn_class = -1, rf_class = -1; // Latest beat
  float cnn_conf = 0;
  uint32_t cnn_us = 0, rf_us = 0;
  int pvc_count = 0;
  bool alert = false;
  uint32_t beats = 0, agreements = 0, dropped_samples = 0;
  ecg::SqaStatus sqa;

  void resetClinical() {
    hr = sdnn = rmssd = pnn50 = mean_nn = 0;
    rhythm = rhythm_rf = cnn_class = rf_class = -1;
    cnn_conf = 0;
    pvc_count = 0;
    alert = false;
    beats = agreements = 0;
    sqa = ecg::SqaStatus();
  }
};

void selectSource(int source);
const char* sourceName(int source);
int sourceCount();
