// Wi-Fi access point and phone dashboard (Mobile/IoT extension and data logging).
#pragma once
#include <stdint.h>
#include "state.h"

namespace webui {
void begin();
void loop();
void pushSamples(const int16_t* uv, int count);
void pushStatus(const LiveState& st, const char* source);
}
