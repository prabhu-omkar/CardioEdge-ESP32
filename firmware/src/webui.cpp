#include "webui.h"
#include <ESPAsyncWebServer.h>
#include <LittleFS.h>
#include <WiFi.h>
#include "config.h"
#include "dashboard_html.h"

namespace webui {

static AsyncWebServer server(80);
static AsyncWebSocket ws("/ws");
static int16_t pending[64];
static int pendingCount = 0;

static void onEvent(AsyncWebSocket*, AsyncWebSocketClient*, AwsEventType type, void* arg, uint8_t* data, size_t len) {
  if (type != WS_EVT_DATA) return;
  const AwsFrameInfo* info = static_cast<AwsFrameInfo*>(arg);
  if (!info->final || info->index != 0 || info->opcode != WS_TEXT || len == 0 || len > 8) return;
  char buf[9];
  memcpy(buf, data, len);
  buf[len] = 0;
  selectSource(atoi(buf));
}

void begin() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASSWORD);
  ws.onEvent(onEvent);
  server.addHandler(&ws);
  server.on("/", HTTP_GET, [](AsyncWebServerRequest* r) {
    r->send_P(200, "text/html", DASHBOARD_HTML);
  });
  server.on("/log.csv", HTTP_GET, [](AsyncWebServerRequest* r) {
    if (LittleFS.exists("/log.csv")) r->send(LittleFS, "/log.csv", "text/csv", true);
    else r->send(200, "text/csv", "");
  });
  server.on("/log/clear", HTTP_POST, [](AsyncWebServerRequest* r) {
    LittleFS.remove("/log.csv");
    r->send(204);
  });
  server.begin();
}

void loop() { ws.cleanupClients(); }

void pushSamples(const int16_t* uv, int count) {
  if (ws.count() == 0) { pendingCount = 0; return; }
  for (int i = 0; i < count; i++) {
    pending[pendingCount++] = uv[i];
    if (pendingCount == 25) {  // Five frames per second at 125 samples/s
      if (ws.availableForWriteAll()) ws.binaryAll(reinterpret_cast<uint8_t*>(pending), pendingCount * sizeof(int16_t));
      pendingCount = 0;
    }
  }
}

void pushStatus(const LiveState& s, const char* source) {
  if (ws.count() == 0 || !ws.availableForWriteAll()) return;
  char buf[640];
  const int n = snprintf(
      buf, sizeof(buf),
      "{\"src\":%d,\"srcName\":\"%s\",\"nsrc\":%d,\"hr\":%.1f,\"sdnn\":%.1f,\"rmssd\":%.1f,\"pnn50\":%.1f,"
      "\"rhythm\":\"%s\",\"rhythmRf\":\"%s\",\"beat\":\"%s\",\"beatRf\":\"%s\",\"conf\":%.2f,\"pvc\":%d,"
      "\"alert\":%d,\"beats\":%lu,\"agree\":%lu,\"cnnUs\":%lu,\"rfUs\":%lu,\"good\":%d,\"off\":%d,"
      "\"ampOk\":%d,\"snrOk\":%d,\"baseOk\":%d,\"morphOk\":%d,\"amp\":%.2f,\"snr\":%.1f,\"base\":%.2f,\"morph\":%.2f}",
      s.source, source, sourceCount(), s.hr, s.sdnn, s.rmssd, s.pnn50,
      s.rhythm >= 0 ? ecg::className(s.rhythm) : "", s.rhythm_rf >= 0 ? ecg::className(s.rhythm_rf) : "",
      s.cnn_class >= 0 ? ecg::className(s.cnn_class) : "", s.rf_class >= 0 ? ecg::className(s.rf_class) : "",
      s.cnn_conf, s.pvc_count, s.alert, (unsigned long)s.beats, (unsigned long)s.agreements,
      (unsigned long)s.cnn_us, (unsigned long)s.rf_us, s.sqa.good, s.sqa.leads_off, s.sqa.amp_ok, s.sqa.snr_ok,
      s.sqa.base_ok, s.sqa.morph_ok, s.sqa.amp_mv, s.sqa.snr_db, s.sqa.base_mv, s.sqa.morph);
  if (n > 0 && n < (int)sizeof(buf)) ws.textAll(buf, n);
}

}  // namespace webui
