// Board wiring and device settings.
#pragma once

// AD8232 (see docs/HARDWARE.md). The output must be on an ADC1 pin: ADC2 cannot be
// used while Wi-Fi is running.
#define PIN_ECG_OUTPUT 34
#define PIN_LO_PLUS 32
#define PIN_LO_MINUS 33

// Outputs
#define PIN_BUZZER 25
#define PIN_ALERT_LED 2
#define PIN_I2C_SDA 21
#define PIN_I2C_SCL 22

// BOOT button: short press selects the next signal source
#define PIN_BUTTON 0

#define OLED_ADDRESS 0x3C

// AD8232 module gain from electrodes to OUTPUT. The DSP core expects mV at the
// electrodes. Heart-rate boards ship at about 1100 V/V; check yours with
// docs/HARDWARE.md if SQA reports the wrong QRS amplitude.
#define AD8232_GAIN 1100.0f

// Wi-Fi access point for the phone dashboard. Open http://192.168.4.1
#define WIFI_SSID "ECG-Monitor"
#define WIFI_PASSWORD "ecgmonitor"

// ADC readings near the rails mean the amplifier has saturated
#define ADC_RAIL_LOW 8
#define ADC_RAIL_HIGH 4087

#define ADC_OVERSAMPLE 4
