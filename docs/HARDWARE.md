# Hardware build guide

## Safety first

This is an educational prototype, not a medical device.

**Never connect electrodes to a person while the circuit is powered from mains.** That includes USB from a laptop on its charger, a USB wall adapter, or a desktop PC. A mains fault could reach the body through the electrodes. Run the device from a battery. For serial capture, use a laptop running on battery with its charger unplugged.

Do not use it on anyone with a pacemaker or other implanted device, on broken skin, or to make any health decision. If the device shows an alert on a real person, treat it as a demo result. Anyone with symptoms should see a doctor regardless of what the prototype shows.

## Parts

| Part | Notes |
|---|---|
| ESP32 DevKit (ESP32-WROOM-32, 4 MB flash) | The plain ESP32. The `esp32dev` build targets this. |
| AD8232 ECG module with 3.5 mm lead cable | Includes the 3 snap leads |
| Disposable Ag/AgCl gel electrodes | Fresh ones. Dried gel is the most common cause of noise. |
| SSD1306 OLED, 128x64, I2C | 0.96 inch, address 0x3C |
| Active 3.3 V buzzer | With an NPN transistor if it draws over 12 mA |
| LiPo battery with 3.3 V or 5 V regulated output | Or a USB power bank, not plugged in to charge |

## Wiring

| From | To (ESP32) | Why |
|---|---|---|
| AD8232 3.3V | 3V3 | Do not use 5 V |
| AD8232 GND | GND | |
| AD8232 OUTPUT | GPIO 34 | Must be ADC1. ADC2 stops working when Wi-Fi is on. |
| AD8232 LO+ | GPIO 32 | Lead-off detect |
| AD8232 LO- | GPIO 33 | Lead-off detect |
| AD8232 SDN | Not connected | Internally pulled high |
| OLED VCC / GND | 3V3 / GND | |
| OLED SDA | GPIO 21 | |
| OLED SCL | GPIO 22 | |
| Buzzer + | GPIO 25 | Through a transistor if needed |
| Buzzer - | GND | |

The onboard LED on GPIO 2 mirrors the alert. All pin numbers are in `firmware/src/config.h`.

Keep the AD8232 wires short and away from the ESP32 antenna. Wi-Fi bursts can couple into the analog line. If the trace shows periodic noise, connect a 100 nF capacitor between AD8232 3.3V and GND, right at the module.

## Electrode placement

Placement matters more than it looks. The models were trained mostly on lead II, where the R wave points upward and the trace looks like a textbook ECG. The placement in the project's block diagram (chest, three pads across the top) gives a different lead and may reduce accuracy.

Use this placement to approximate modified lead II:

| Lead | Usual colour | Position |
|---|---|---|
| RA | Red | Right chest, just below the collarbone |
| LA | Yellow | **Left lower rib cage**, above the hip (this stands in for the left leg) |
| RL | Green | Right lower rib cage (reference) |

Colours vary between cables, so check the labels printed on your snaps. Clean the skin with an alcohol wipe and let it dry. Sit still and relaxed.

If the QRS spikes point downward, swap the RA and LA leads.

## First power-up checklist

1. Build and flash (see the README).
2. With no electrodes, press BOOT to step through the demo clips. This checks the display, buzzer and models with no analog hardware.
3. Open the serial monitor at 921600 baud. Type `b` and record the `BENCH` line, which gives real inference times for the comparison table.
4. Connect to the Wi-Fi network `ECG-Monitor` (password `ecgmonitor`) and open `http://192.168.4.1`.
5. Press BOOT until the screen says LIVE. Disconnect USB power from mains and run on battery. Attach the electrodes.
6. Wait about 15 s. The core needs 2.5 s to learn thresholds, 8 beats to find the dominant beat shape, and 2 s of quality buffering.

## Checking the gain

The firmware assumes the module's gain is 1100 (`AD8232_GAIN` in `config.h`). The classifiers do not depend on this because each beat is normalised, but the SQA amplitude check does. On a healthy adult at rest, the dashboard's Amplitude chip should read roughly 0.5 to 2 mV. If it is far outside that with a clean-looking trace, check your module's schematic and adjust the value.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Leads off" stays on | Electrode not snapped, or LO pins miswired | Check all 3 snaps and GPIO 32, 33 |
| "Poor signal: Noise" | Movement, dry gel, cable over the body | Sit still, fresh electrodes, tape the cable down |
| "Poor signal: Baseline drift" | Breathing or moving, loose pad | Press pads firmly, breathe normally |
| "Poor signal: Amplitude" | Very small QRS, or wrong gain | Check placement and `AD8232_GAIN` |
| Regular 50 Hz fuzz | Mains pickup | Move away from chargers, shorten wires, run on battery |
| Trace upside down | RA and LA swapped | Swap them |
| Noise appears only when a phone connects | Wi-Fi coupling | Add the capacitor above, or route AD8232 wires away from the antenna |
| No OLED | Wrong address or SDA/SCL swapped | Serial still works. Check the wiring and `OLED_ADDRESS`. |
