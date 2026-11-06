#include "display.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>
#include "config.h"

namespace display {

static constexpr int W = 128, H = 64, TRACE_TOP = 12, TRACE_H = 38;
static Adafruit_SSD1306 oled(W, H, &Wire, -1);
static bool present = false;
static int16_t trace[W];
static int pos = 0, decim = 0;
static int32_t acc = 0;

void begin() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  present = oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS);
  if (!present) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 20);
  oled.println("ECG Monitor");
  oled.println("Research prototype");
  oled.println("Not a medical device");
  oled.display();
}

void addSample(int16_t uv) {
  // 128 pixels shows about 4 s of the 125 Hz trace
  acc += uv;
  if (++decim < 4) return;
  trace[pos] = acc / 4;
  acc = 0; decim = 0;
  pos = (pos + 1) % W;
}

static void drawTrace() {
  int lo = 32767, hi = -32768;
  for (int i = 0; i < W; i++) { lo = min<int>(lo, trace[i]); hi = max<int>(hi, trace[i]); }
  const int span = max(hi - lo, 300);
  int prev = -1;
  for (int x = 0; x < W; x++) {
    const int v = trace[(pos + x) % W];
    const int y = TRACE_TOP + TRACE_H - 1 - (long)(v - lo) * (TRACE_H - 1) / span;
    if (prev >= 0) oled.drawLine(x - 1, prev, x, y, SSD1306_WHITE);
    prev = y;
  }
}

static void drawWarning(const LiveState& st) {
  oled.fillTriangle(10, 2, 0, 20, 20, 20, SSD1306_WHITE);
  oled.fillRect(9, 8, 3, 7, SSD1306_BLACK);
  oled.fillRect(9, 16, 3, 2, SSD1306_BLACK);
  oled.setCursor(28, 2);
  oled.print(st.sqa.leads_off ? "Leads off" : "Poor signal");
  oled.setCursor(28, 12);
  if (st.sqa.leads_off) oled.print("Check electrodes");
  else if (!st.sqa.amp_ok) oled.print("Amplitude");
  else if (!st.sqa.snr_ok) oled.print("Noise");
  else if (!st.sqa.base_ok) oled.print("Baseline drift");
  else oled.print("Irregular shape");
  oled.setCursor(0, 30);
  oled.print("Please reposition");
  oled.setCursor(0, 40);
  oled.print("electrodes");
  oled.setCursor(0, 54);
  oled.printf("SNR %.0f dB", st.sqa.snr_db);
}

void draw(const LiveState& st, const char* source) {
  if (!present) return;
  oled.clearDisplay();
  if (!st.sqa.good && st.sqa.leads_off) {
    drawWarning(st);
  } else if (!st.sqa.good && st.beats > 0) {
    drawWarning(st);
  } else {
    oled.setCursor(0, 0);
    if (st.hr > 0) oled.printf("HR %3.0f", st.hr); else oled.print("HR ---");
    oled.setCursor(52, 0);
    oled.print(st.source == 0 ? "LIVE" : "DEMO");
    oled.setCursor(84, 0);
    oled.printf("SNR%3.0f", st.sqa.snr_db);
    drawTrace();
    oled.setCursor(0, 54);
    if (st.rhythm >= 0) {
      if (st.alert) oled.print("! ");
      oled.print(ecg::className(st.rhythm));
      oled.setCursor(78, 54);
      oled.printf("RF:%s", ecg::className(st.rhythm_rf >= 0 ? st.rhythm_rf : 0));
    } else {
      oled.print(st.sqa.good ? "Analysing..." : "Acquiring signal...");
    }
  }
  oled.display();
}

void fatal(const char* msg) {
  if (!present) return;
  oled.clearDisplay();
  oled.setCursor(0, 20);
  oled.print(msg);
  oled.display();
}

}  // namespace display
