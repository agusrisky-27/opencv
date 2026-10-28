/*
 * ============================================================================
 * CONVEYOR IR SENSOR CONTROLLER (Arduino / ESP32)
 * Proyek: Robotik Sortir Berbasis Machine Vision YOLOv8
 * ============================================================================
 * 
 * CARA KERJA SISTEM STOP-AND-GO:
 * 1. Konveyor berjalan membawa objek (Koin / Tutup Botol / Charger).
 * 2. Sensor Infrared (IR) mendeteksi objek yang tiba di zona sortir.
 * 3. Arduino mematikan Relay Konveyor seketika (STOP).
 * 4. Arduino mengirim string "IR_TRIGGER\n" via USB Serial ke laptop Python.
 * 5. Python menunggu jeda inersia (Settling Delay ~0.6s) agar getaran & selip reda total,
 *    mengambil 1 snapshot foto statis, dan mengunci koordinat fisik (X, Y, Z).
 * 6. Jika objek termasuk kelas target sortir:
 *    - Python mengirim instruksi "PICK:<class>:<x>:<y>:<z>\n" ke robot.
 *    - Setelah robot selesai mengambil, kirim "DONE\n" untuk menghidupkan konveyor lagi.
 * 7. Jika objek bukan target sortir:
 *    - Python mengirim "RESUME\n" sehingga konveyor langsung jalan membawa objek lewat.
 * ============================================================================
 */

#define PIN_IR_SENSOR    2   // Pin digital sensor Infrared (Aktif LOW / HIGH sesuai modul)
#define PIN_RELAY_CONV   7   // Pin kontrol Relay Konveyor Motor (Aktif LOW)
#define PIN_LED_INDICATOR 13 // LED indikator status

// Konfigurasi tipe sensor IR (kebanyakan modul IR TCRT5000 / Proximity = LOW saat deteksi)
const int IR_DETECTED_STATE = LOW; 
const int RELAY_ON_STATE    = LOW;   // Relay aktif menyalakan konveyor
const int RELAY_OFF_STATE   = HIGH;  // Relay mati menghentikan konveyor

// Parameter Debounce & Timing
const unsigned long DEBOUNCE_DELAY_MS = 200; // Mencegah noise bouncing sensor IR
unsigned long lastDebounceTime = 0;
bool conveyorRunning = true;
bool waitingForPython = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_IR_SENSOR, INPUT_PULLUP);
  pinMode(PIN_RELAY_CONV, OUTPUT);
  pinMode(PIN_LED_INDICATOR, OUTPUT);

  // Nyalakan konveyor awal
  startConveyor();
  Serial.println("ARDUINO_READY: Conveyor IR Controller Active");
}

void loop() {
  // 1. Baca Sinyal dari Python / Laptop Robotik
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("RESUME") || command.startsWith("DONE")) {
      // Perintah melanjutkan konveyor
      startConveyor();
      waitingForPython = false;
      delay(300); // Waktu agar objek yang lewat tidak langsung mentrigger sensor lagi
    }
  }

  // 2. Baca Sensor Infrared saat konveyor sedang berjalan
  if (conveyorRunning && !waitingForPython) {
    int sensorVal = digitalRead(PIN_IR_SENSOR);

    if (sensorVal == IR_DETECTED_STATE) {
      if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY_MS) {
        lastDebounceTime = millis();

        // Objek tiba! Matikan konveyor seketika
        stopConveyor();
        waitingForPython = true;

        // Kirim notifikasi trigger ke Python
        Serial.println("IR_TRIGGER");
      }
    }
  }
}

void startConveyor() {
  digitalWrite(PIN_RELAY_CONV, RELAY_ON_STATE);
  digitalWrite(PIN_LED_INDICATOR, HIGH);
  conveyorRunning = true;
}

void stopConveyor() {
  digitalWrite(PIN_RELAY_CONV, RELAY_OFF_STATE);
  digitalWrite(PIN_LED_INDICATOR, LOW);
  conveyorRunning = false;
}
