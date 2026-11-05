// Random Forest inference: majority vote of the trees in model_rf.h.
#pragma once
#include <string.h>
#include "ecg_config.h"
#include "model_rf.h"

inline int rfVote(const float* f) {
  int votes[ecg::N_CLASSES];
  memset(votes, 0, sizeof(votes));
  for (int t = 0; t < RF_N_TREES; t++) {
    uint16_t n = rf_roots[t];
    while (rf_feature[n] >= 0)
      n = f[rf_feature[n]] <= rf_threshold[n] ? rf_left[n] : rf_right[n];
    votes[rf_left[n]]++;
  }
  int best = 0;
  for (int c = 1; c < ecg::N_CLASSES; c++)
    if (votes[c] > votes[best]) best = c;
  return best;
}
