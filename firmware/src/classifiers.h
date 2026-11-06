// Runs both classifiers on a beat and reports their inference times.
#pragma once
#include <stdint.h>
#include "ecg_core.h"

namespace classifiers {

struct Result {
  int cnn_class = -1;
  int rf_class = -1;
  float cnn_conf = 0;
  uint32_t cnn_us = 0;
  uint32_t rf_us = 0;
};

// Returns false if the TensorFlow Lite model could not be loaded.
bool begin();
Result classify(const ecg::Beat& beat);
// Random Forest alone, also used by the host tests.
int rfPredict(const float* features);
int arenaUsedBytes();
// int8 CNN output of the last classify() call, for the self-test.
const int8_t* rawScores();

}  // namespace classifiers
