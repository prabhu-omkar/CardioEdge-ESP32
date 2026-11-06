// SSD1306 128x64 OLED: waveform, heart rate, verdict and electrode warnings.
#pragma once
#include <stdint.h>
#include "state.h"

namespace display {
void begin();
void addSample(int16_t uv);
void draw(const LiveState& st, const char* source);
void fatal(const char* msg);
}
