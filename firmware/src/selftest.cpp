// Self-test build (env:qemu_selftest), for the ESP32 emulator or a bare board.
// Checks the deployed models against vectors produced by Python, then replays the demo
// clips through the full pipeline at top speed. Results go to the serial port.
#include <Arduino.h>
#include <esp_timer.h>
#include "classifiers.h"
#include "demo_clips.h"
#include "ecg_core.h"
#include "selftest_vectors.h"

using namespace ecg;

static EcgCore core;
static RhythmTracker rhythmCnn, rhythmRf;
static Beat beat;

static void checkVectors() {
  int cnnOk = 0, rfOk = 0, worst = 0;
  for (int i = 0; i < ST_COUNT; i++) {
    memcpy(beat.wave, ST_WAVE[i], sizeof(beat.wave));
    memcpy(beat.ctx, ST_CTX[i], sizeof(beat.ctx));
    memcpy(beat.morph, ST_MORPH[i], sizeof(beat.morph));
    const classifiers::Result r = classifiers::classify(beat);
    if (r.cnn_class == ST_CNN[i]) cnnOk++;
    if (r.rf_class == ST_RF[i]) rfOk++;
    const int8_t* s = classifiers::rawScores();
    for (int c = 0; c < N_CLASSES; c++) worst = max(worst, abs((int)s[c] - (int)ST_SCORES[i][c]));
  }
  Serial.printf("VECTORS,count=%d,cnn_match=%d,rf_match=%d,max_score_diff=%d\n", ST_COUNT, cnnOk, rfOk, worst);
}

static void replay(const DemoClip& clip) {
  core.reset();
  rhythmCnn.reset();
  rhythmRf.reset();
  int detected = 0, analysed = 0, cnnCounts[N_CLASSES] = {0}, rfCounts[N_CLASSES] = {0};
  int alertBeats = 0;
  RhythmStatus rc, rr;
  for (uint32_t i = 0; i < clip.length; i++) {
    const bool ready = core.push(clip.data[i] * clip.scale_mv, false);
    if (ready) {
      detected++;
      const Beat& b = core.beat();
      if (b.classifiable()) {
        classifiers::Result r = classifiers::classify(b);
        r.cnn_class = rhythmRule(r.cnn_class, b.ctx);
        r.rf_class = rhythmRule(r.rf_class, b.ctx);
        analysed++;
        cnnCounts[r.cnn_class]++;
        rfCounts[r.rf_class]++;
        rc = rhythmCnn.push(r.cnn_class);
        rr = rhythmRf.push(r.rf_class);
        if (rc.alert) alertBeats++;
      }
    }
    if (i % 2000 == 0) delay(1);
  }
  Serial.printf("CLIP,%s,detected=%d,analysed=%d,alert_beats=%d,rhythm=%s,rhythm_rf=%s,cnn=", clip.name, detected, analysed,
                alertBeats, rc.label >= 0 ? className(rc.label) : "-", rr.label >= 0 ? className(rr.label) : "-");
  for (int c = 0; c < N_CLASSES; c++) Serial.printf("%d%c", cnnCounts[c], c + 1 < N_CLASSES ? ' ' : ',');
  Serial.print("rf=");
  for (int c = 0; c < N_CLASSES; c++) Serial.printf("%d%c", rfCounts[c], c + 1 < N_CLASSES ? ' ' : '\n');
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("SELFTEST,boot");
  if (!classifiers::begin()) {
    Serial.println("SELFTEST,model_load_failed");
    return;
  }
  Serial.printf("MODEL,arena_bytes=%d,free_heap=%lu\n", classifiers::arenaUsedBytes(), (unsigned long)ESP.getFreeHeap());
  checkVectors();
  for (int i = 0; i < DEMO_CLIP_COUNT; i++) replay(DEMO_CLIPS[i]);
  Serial.println("SELFTEST,done");
}

void loop() { delay(1000); }
