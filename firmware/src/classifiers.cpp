#include "classifiers.h"
#include <TensorFlowLite_ESP32.h>
#include <esp_timer.h>
#include <math.h>
#include "model_cnn.h"
#include "rf_predict.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace classifiers {

static constexpr int ARENA_SIZE = 24 * 1024;
alignas(16) static uint8_t arena[ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* waveIn = nullptr;
static TfLiteTensor* ctxIn = nullptr;
static TfLiteTensor* out = nullptr;

bool begin() {
  static tflite::MicroErrorReporter reporter;
  const tflite::Model* model = tflite::GetModel(cnn_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;

  static tflite::MicroMutableOpResolver<9> ops;
  ops.AddConv2D();
  ops.AddMaxPool2D();
  ops.AddFullyConnected();
  ops.AddConcatenation();
  ops.AddReshape();
  ops.AddExpandDims();
  ops.AddSoftmax();
  ops.AddQuantize();
  ops.AddDequantize();

  static tflite::MicroInterpreter instance(model, ops, arena, ARENA_SIZE, &reporter);
  interpreter = &instance;
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;

  // The converter does not guarantee input order, so identify them by shape
  for (size_t i = 0; i < interpreter->inputs_size(); i++) {
    TfLiteTensor* t = interpreter->input(i);
    if (t->dims->data[1] == ecg::WAVE_LEN) waveIn = t; else ctxIn = t;
  }
  out = interpreter->output(0);
  return waveIn && ctxIn && out;
}

static inline int8_t quantise(float v, const TfLiteTensor* t) {
  const int q = (int)lroundf(v / t->params.scale) + t->params.zero_point;
  return (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}

int rfPredict(const float* features) { return rfVote(features); }

Result classify(const ecg::Beat& b) {
  Result r;
  float features[ecg::N_FEAT];
  memcpy(features, b.ctx, sizeof(b.ctx));
  memcpy(features + ecg::N_CTX, b.morph, sizeof(b.morph));
  int64_t t0 = esp_timer_get_time();
  r.rf_class = rfVote(features);
  r.rf_us = (uint32_t)(esp_timer_get_time() - t0);

  t0 = esp_timer_get_time();
  for (int i = 0; i < ecg::WAVE_LEN; i++) waveIn->data.int8[i] = quantise(b.wave[i], waveIn);
  for (int i = 0; i < ecg::N_CTX; i++) ctxIn->data.int8[i] = quantise(b.ctx[i], ctxIn);
  if (interpreter->Invoke() != kTfLiteOk) return r;
  int best = 0;
  for (int c = 1; c < ecg::N_CLASSES; c++)
    if (out->data.int8[c] > out->data.int8[best]) best = c;
  r.cnn_us = (uint32_t)(esp_timer_get_time() - t0);
  r.cnn_class = best;
  r.cnn_conf = (out->data.int8[best] - out->params.zero_point) * out->params.scale;
  return r;
}

int arenaUsedBytes() { return interpreter ? (int)interpreter->arena_used_bytes() : 0; }

const int8_t* rawScores() { return out->data.int8; }

}  // namespace classifiers
