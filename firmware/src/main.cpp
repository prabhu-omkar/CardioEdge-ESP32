// AI-Based Real-Time ECG Signal Analysis and Multi-Class Cardiac Abnormality Detection
// ESP32 application. Educational prototype only: not a medical device.
//
// Tasks:
//   acqTask (core 1, highest)  250 Hz sampling from the AD8232 or a demo clip
//   dspTask (core 1)           filtering, QRS, SQA, both classifiers, alert decision
//   uiTask  (core 0)           OLED, buzzer, button, serial and Wi-Fi dashboard
#include <Arduino.h>
#include <LittleFS.h>
#include "classifiers.h"
#include "config.h"
#include "demo_clips.h"
#include "display.h"
#include "ecg_core.h"
#include "state.h"
#include "webui.h"

using namespace ecg;

struct Sample { float mv; bool leads_off; int source = 0; uint32_t sequence = 0; };

static QueueHandle_t sampleQueue;
static TaskHandle_t acqHandle;
static hw_timer_t* timer;
static EcgCore core;
static HrvTracker hrv;
static RhythmTracker rhythmCnn, rhythmRf;

static portMUX_TYPE stateLock = portMUX_INITIALIZER_UNLOCKED;
static LiveState live;
static volatile int requestedSource = 0;
static volatile bool rawStream = false;

// Filtered samples waiting for the dashboard and OLED, in microvolts at 125 Hz
static int16_t traceBuf[256];
static volatile uint32_t traceHead = 0;

// Inference timing, reported by the serial 'b' command
static uint32_t cnnSumUs, rfSumUs, cnnMaxUs, rfMaxUs, timedBeats;

void IRAM_ATTR onTimer() {
  BaseType_t woken = pdFALSE;
  vTaskNotifyGiveFromISR(acqHandle, &woken);
  if (woken) portYIELD_FROM_ISR();
}

static Sample readAd8232() {
  uint32_t sum = 0;
  for (int i = 0; i < ADC_OVERSAMPLE; i++) sum += analogRead(PIN_ECG_OUTPUT);
  const float raw = (float)sum / ADC_OVERSAMPLE;
  const bool railed = raw <= ADC_RAIL_LOW || raw >= ADC_RAIL_HIGH;
  const bool off = digitalRead(PIN_LO_PLUS) || digitalRead(PIN_LO_MINUS) || railed;
  // 12-bit reading at 11 dB attenuation covers about 3.1 V
  return {raw * 3100.0f / 4095.0f / AD8232_GAIN, off};
}

static void acqTask(void*) {
  int source = 0;
  uint32_t pos = 0;
  uint32_t sequence = 0;
  for (;;) {
    const uint32_t ticks = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    sequence += ticks;
    const int nextSource = requestedSource;
    if (source != nextSource) { source = nextSource; pos = 0; }
    Sample s;
    if (source == 0) {
      s = readAd8232();
    } else {
      const DemoClip& clip = DEMO_CLIPS[source - 1];
      s = {clip.data[pos] * clip.scale_mv, false};  // Signed data: pgm_read_word would lose the sign
      if (++pos >= clip.length) pos = 0;
    }
    s.source = source;
    s.sequence = sequence;
    // Count both missed timer ticks and full-queue losses. Sequence gaps reset DSP.
    const uint32_t dropped = ticks - 1 + (xQueueSend(sampleQueue, &s, 0) != pdTRUE);
    if (dropped) {
      portENTER_CRITICAL(&stateLock);
      live.dropped_samples += dropped;
      portEXIT_CRITICAL(&stateLock);
    }
  }
}

static void resetPipeline() {
  core.reset();
  hrv.reset();
  rhythmCnn.reset();
  rhythmRf.reset();
}

static void handleBeat(const Beat& b, int& prevClass, float* rrRing, int& rrCount) {
  // Rolling heart rate uses every beat with usable signal, whatever its class
  if ((b.flags & BEAT_GOOD_SQ) && b.rr_pre_ms > 0) {
    rrRing[rrCount++ % 8] = b.rr_pre_ms;
  }
  if (!b.classifiable()) { prevClass = -1; return; }

  classifiers::Result r = classifiers::classify(b);
  if (r.cnn_class < 0 || r.cnn_class >= N_CLASSES || r.rf_class < 0 || r.rf_class >= N_CLASSES) {
    rhythmCnn.reset(); rhythmRf.reset(); hrv.reset(); prevClass = -1;
    portENTER_CRITICAL(&stateLock);
    live.resetClinical();
    portEXIT_CRITICAL(&stateLock);
    return;
  }
  const int rawCnnClass = r.cnn_class;
  r.cnn_class = rhythmRule(r.cnn_class, b.ctx);
  // A rhythm-rule decision has no CNN probability of its own.
  if (r.cnn_class != rawCnnClass) r.cnn_conf = -1;
  r.rf_class = rhythmRule(r.rf_class, b.ctx);
  const RhythmStatus rc = rhythmCnn.push(r.cnn_class);
  const RhythmStatus rr = rhythmRf.push(r.rf_class);

  // HRV is defined on normal-to-normal intervals: skip intervals touching an ectopic beat
  const auto ectopic = [](int c) { return c == PVC || c == OTHER; };
  if (prevClass >= 0 && !ectopic(prevClass) && !ectopic(r.cnn_class)) hrv.push(b.rr_pre_ms);
  prevClass = r.cnn_class;

  cnnSumUs += r.cnn_us; rfSumUs += r.rf_us;
  cnnMaxUs = max(cnnMaxUs, r.cnn_us); rfMaxUs = max(rfMaxUs, r.rf_us);
  timedBeats++;

  const HrvStats h = hrv.stats();
  portENTER_CRITICAL(&stateLock);
  live.cnn_class = r.cnn_class;
  live.rf_class = r.rf_class;
  live.cnn_conf = r.cnn_conf;
  live.cnn_us = r.cnn_us;
  live.rf_us = r.rf_us;
  live.rhythm = rc.label;
  live.rhythm_rf = rr.label;
  live.pvc_count = rc.pvc_count;
  live.alert = rc.alert;
  live.sdnn = h.sdnn_ms;
  live.rmssd = h.rmssd_ms;
  live.pnn50 = h.pnn50;
  live.mean_nn = h.mean_nn_ms;
  live.beats++;
  if (r.cnn_class == r.rf_class) live.agreements++;
  portEXIT_CRITICAL(&stateLock);

  if (Serial && !rawStream)
    Serial.printf("B,%lu,%s,%s,%.0f,%lu,%lu\n", (unsigned long)millis(), className(r.cnn_class), className(r.rf_class),
                  b.rr_pre_ms, (unsigned long)r.cnn_us, (unsigned long)r.rf_us);
}

static void dspTask(void*) {
  int source = 0, prevClass = -1, rrCount = 0, decimate = 0;
  float rrRing[8] = {0};
  uint32_t lastSequence = 0;
  Sample s;
  for (;;) {
    if (xQueueReceive(sampleQueue, &s, portMAX_DELAY) != pdTRUE) continue;
    if (source != s.source || s.sequence != lastSequence + 1) {
      source = s.source;
      resetPipeline();
      prevClass = -1; rrCount = 0; decimate = 0;
      portENTER_CRITICAL(&stateLock);
      live.resetClinical();
      live.source = source;
      portEXIT_CRITICAL(&stateLock);
    }
    lastSequence = s.sequence;
    const bool beatReady = core.push(s.mv, s.leads_off);
    if (rawStream && Serial) Serial.printf("R,%.4f,%.4f,%d\n", s.mv, core.filtered(), s.leads_off);
    if (beatReady) handleBeat(core.beat(), prevClass, rrRing, rrCount);

    if (++decimate == 2) {
      decimate = 0;
      traceBuf[traceHead % 256] = (int16_t)constrain(core.filtered() * 1000.0f, -32000.0f, 32000.0f);
      traceHead = traceHead + 1;
    }

    if (core.sqaUpdated()) {
      const SqaStatus q = core.sqa();
      float sum = 0; const int n = min(rrCount, 8);
      for (int i = 0; i < n; i++) sum += rrRing[i];
      portENTER_CRITICAL(&stateLock);
      live.sqa = q;
      live.hr = (n >= 3 && q.good) ? 60000.0f * n / sum : 0;
      if (!q.good) live.alert = false;  // Never raise an alarm from a signal we do not trust
      portEXIT_CRITICAL(&stateLock);
    }
  }
}

static void snapshot(LiveState& out) {
  portENTER_CRITICAL(&stateLock);
  out = live;
  portEXIT_CRITICAL(&stateLock);
}

void selectSource(int source) {
  if (source >= 0 && source <= DEMO_CLIP_COUNT) requestedSource = source;
}

const char* sourceName(int source) { return source == 0 ? "Live AD8232" : DEMO_CLIPS[source - 1].name; }
int sourceCount() { return DEMO_CLIP_COUNT + 1; }

static void updateBuzzer(bool alert) {
  // 200 ms beep every second while an alert is active
  const bool on = alert && (millis() % 1000) < 200;
  digitalWrite(PIN_BUZZER, on);
  digitalWrite(PIN_ALERT_LED, alert);
}

static void logEvent(const LiveState& st) {
  static int lastRhythm = -2;
  if (st.rhythm == lastRhythm) return;
  lastRhythm = st.rhythm;
  if (st.rhythm < 0) return;
  File f = LittleFS.open("/log.csv", "a");
  if (!f) return;
  if (f.size() < 256 * 1024)
    f.printf("%lu,%s,%s,%.0f,%.1f,%.1f,%.1f\n", (unsigned long)(millis() / 1000), className(st.rhythm),
             sourceName(st.source), st.hr, st.sdnn, st.rmssd, st.sqa.snr_db);
  f.close();
}

static void handleSerial() {
  if (!Serial.available()) return;
  const char c = Serial.read();
  if (c == 'n') selectSource((requestedSource + 1) % sourceCount());
  else if (c == 'r') rawStream = !rawStream;
  else if (c == 'b') {
    const uint32_t n = max<uint32_t>(timedBeats, 1);
    Serial.printf("BENCH,beats=%lu,cnn_avg_us=%lu,cnn_max_us=%lu,rf_avg_us=%lu,rf_max_us=%lu,arena=%d,dropped=%lu\n",
                  (unsigned long)timedBeats, (unsigned long)(cnnSumUs / n), (unsigned long)cnnMaxUs,
                  (unsigned long)(rfSumUs / n), (unsigned long)rfMaxUs, classifiers::arenaUsedBytes(),
                  (unsigned long)live.dropped_samples);
  }
}

static void uiTask(void*) {
  uint32_t tracePos = 0, lastDraw = 0, lastPush = 0, lastStatus = 0;
  bool lastButton = true;
  uint32_t buttonAt = 0;
  LiveState st;
  int16_t chunk[64];
  for (;;) {
    const uint32_t now = millis();
    snapshot(st);
    updateBuzzer(st.alert);
    handleSerial();

    const bool button = digitalRead(PIN_BUTTON);
    if (lastButton && !button && now - buttonAt > 250) {
      buttonAt = now;
      selectSource((requestedSource + 1) % sourceCount());
    }
    lastButton = button;

    int count = 0;
    while (tracePos != traceHead && count < 64) {
      chunk[count] = traceBuf[tracePos % 256];
      display::addSample(chunk[count]);
      count++;
      tracePos++;
    }
    if (count > 0) webui::pushSamples(chunk, count);

    if (now - lastDraw >= 100) { lastDraw = now; display::draw(st, sourceName(st.source)); }
    if (now - lastPush >= 500) { lastPush = now; webui::pushStatus(st, sourceName(st.source)); logEvent(st); }
    if (now - lastStatus >= 2000 && Serial && !rawStream) {
      lastStatus = now;
      Serial.printf("S,%lu,%s,hr=%.0f,sqa=%d,snr=%.1f,rhythm=%s\n", (unsigned long)now, sourceName(st.source), st.hr,
                    st.sqa.good, st.sqa.snr_db, st.rhythm >= 0 ? className(st.rhythm) : "-");
    }
    webui::loop();
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

void setup() {
  Serial.begin(921600);
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_ALERT_LED, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_ECG_OUTPUT, ADC_11db);

  display::begin();
  if (!LittleFS.begin(true)) Serial.println("LittleFS unavailable, event log disabled");
  if (!classifiers::begin()) {
    Serial.println("Model failed to load");
    display::fatal("Model load failed");
    for (;;) delay(1000);
  }
  webui::begin();

  sampleQueue = xQueueCreate(128, sizeof(Sample));
  xTaskCreatePinnedToCore(dspTask, "dsp", 10000, nullptr, 5, nullptr, 1);
  xTaskCreatePinnedToCore(acqTask, "acq", 4096, nullptr, configMAX_PRIORITIES - 2, &acqHandle, 1);
  xTaskCreatePinnedToCore(uiTask, "ui", 8192, nullptr, 2, nullptr, 0);

  timer = timerBegin(0, 80, true);  // 1 MHz tick
  timerAttachInterrupt(timer, onTimer, true);
  timerAlarmWrite(timer, 1000000 / ecg::FS, true);
  timerAlarmEnable(timer);
  Serial.printf("ECG monitor ready. Wi-Fi '%s', http://192.168.4.1\n", WIFI_SSID);
}

void loop() { vTaskDelete(nullptr); }
