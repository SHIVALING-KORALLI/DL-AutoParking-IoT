//Presnt code needed to correct.
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>

// ---------------------------
// OLED Configuration
// ---------------------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ---------------------------
// WiFi & MQTT Configuration
// ---------------------------
const char* ssid = "svlng"; //change this
const char* password = "12345678"; //change this
const char* mqtt_server = "172.20.10.12"; //change this  
const int mqtt_port = 1883;
const char* mqtt_topic = "parking/slots/status";

// ---------------------------
// Shift Register Configuration
// ---------------------------
#define DATA_PIN   D7
#define CLOCK_PIN  D6
#define LATCH_PIN  D5
#define NUM_SLOTS  8

WiFiClient espClient;
PubSubClient client(espClient);

// ---------------------------
// Global variables for parking status
// ---------------------------
int g_freeSpaces = 0;
int g_closestFreeSlot = -1;

// ---------------------------
// Placeholder for Car Bitmap
// ---------------------------
const unsigned char Car_logo [] PROGMEM = {
 // 'pic_logo', 81x60px
0xff, 0xff, 0xff, 0xff, 0xc0, 0x00, 0xff, 0xff, 0xff, 0xff, 0x80, 0xff, 0xff, 0xff, 0xf8, 0x00, 
0x00, 0x07, 0xff, 0xff, 0xff, 0x80, 0xff, 0xff, 0xff, 0x80, 0x00, 0x1f, 0x80, 0x7f, 0xff, 0xff, 
0x80, 0xff, 0xff, 0xfe, 0x00, 0x0f, 0xff, 0xff, 0x1f, 0xff, 0xff, 0x80, 0xff, 0xff, 0xf0, 0x00, 
0xfe, 0x3f, 0xff, 0xf3, 0xff, 0xff, 0x80, 0xff, 0xff, 0xc0, 0x03, 0xc3, 0xfc, 0x00, 0xff, 0xff, 
0xff, 0x80, 0xff, 0xff, 0x00, 0x1e, 0x1f, 0xff, 0xc0, 0x07, 0xff, 0xff, 0x80, 0xff, 0xfe, 0x00, 
0x78, 0x7f, 0xff, 0xf8, 0x00, 0xff, 0xff, 0x80, 0xff, 0xf8, 0x01, 0xe1, 0xff, 0xff, 0xfe, 0x00, 
0x3f, 0xff, 0x80, 0xff, 0xf0, 0x03, 0x87, 0xff, 0xff, 0xff, 0x80, 0x0f, 0xff, 0x80, 0xff, 0xe0, 
0x0e, 0x0f, 0xff, 0xff, 0xff, 0xc0, 0x07, 0xff, 0x80, 0xff, 0xc0, 0x1c, 0x3f, 0xff, 0xff, 0xff, 
0xf0, 0x01, 0xff, 0x80, 0xff, 0x80, 0x30, 0x7f, 0xff, 0xff, 0xff, 0xf8, 0x00, 0xff, 0x80, 0xff, 
0x00, 0x60, 0xff, 0xff, 0xff, 0xff, 0xfc, 0x00, 0x7f, 0x80, 0xfe, 0x00, 0xc1, 0xff, 0xff, 0xff, 
0xff, 0xfe, 0x00, 0x3f, 0x80, 0xfc, 0x01, 0x83, 0xff, 0xff, 0xfe, 0x00, 0x1f, 0x00, 0x19, 0x80, 
0xfc, 0x03, 0x07, 0xff, 0xff, 0xff, 0xe3, 0xf1, 0x00, 0x07, 0x80, 0xf8, 0x06, 0x0f, 0xff, 0xff, 
0xff, 0xef, 0xbf, 0x00, 0x1f, 0x80, 0xf0, 0x06, 0x1f, 0xff, 0x3f, 0xff, 0xd8, 0x01, 0xc0, 0x7f, 
0x80, 0xf0, 0x0c, 0x1f, 0xfc, 0xff, 0xff, 0xb0, 0x00, 0x71, 0xe7, 0x80, 0xe0, 0x08, 0x3f, 0xf3, 
0xff, 0xff, 0x67, 0x00, 0x1f, 0xc3, 0x80, 0xe0, 0x18, 0x7f, 0xef, 0xff, 0xfc, 0xcf, 0x00, 0x1f, 
0x83, 0x80, 0xe0, 0x10, 0x7f, 0x9f, 0xc0, 0x01, 0xc8, 0x07, 0xff, 0xe3, 0x80, 0xc0, 0x30, 0xf0, 
0x00, 0x00, 0x01, 0x01, 0xfe, 0x0c, 0x01, 0x80, 0xc0, 0x20, 0x03, 0xff, 0xe0, 0x00, 0x00, 0x00, 
0x00, 0xc1, 0x80, 0xc0, 0x60, 0x3f, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x01, 0xe1, 0x80, 0xc0, 0x73, 
0xff, 0xf0, 0x00, 0x3c, 0x00, 0x00, 0x03, 0x21, 0x80, 0x80, 0xff, 0xff, 0x80, 0xc0, 0xfe, 0x00, 
0x00, 0x02, 0x00, 0x80, 0x81, 0xff, 0xfc, 0x07, 0x01, 0xc3, 0x00, 0x00, 0x06, 0x00, 0x80, 0x87, 
0xff, 0xe0, 0x3e, 0x03, 0x88, 0x80, 0x00, 0x04, 0x00, 0x80, 0x8f, 0xff, 0x00, 0xf8, 0x03, 0x3e, 
0x80, 0x00, 0x04, 0x00, 0x80, 0x98, 0x00, 0x07, 0xff, 0x06, 0x62, 0x40, 0x00, 0x04, 0x00, 0x80, 
0x80, 0x00, 0x1f, 0xf8, 0x06, 0x41, 0x00, 0x00, 0x1c, 0x01, 0x80, 0xc0, 0x3c, 0x00, 0x00, 0x0e, 
0x81, 0x00, 0x00, 0xfc, 0x27, 0x80, 0xc0, 0x03, 0x00, 0x00, 0x0c, 0x81, 0x80, 0x07, 0xc0, 0x7d, 
0x80, 0xc0, 0x00, 0xc0, 0x00, 0x0c, 0x80, 0x80, 0x1c, 0x01, 0xe1, 0x80, 0xd4, 0x00, 0xe0, 0x00, 
0x0c, 0x80, 0x80, 0x40, 0x0f, 0x81, 0x80, 0xf0, 0x00, 0x70, 0x00, 0x0c, 0x81, 0x80, 0x00, 0x7c, 
0x01, 0x80, 0xfa, 0x00, 0x3c, 0x00, 0x0c, 0x80, 0x80, 0x07, 0xcc, 0x03, 0x80, 0xfc, 0x00, 0x1e, 
0x00, 0x0c, 0x00, 0x80, 0x3e, 0x18, 0x03, 0x80, 0xff, 0xc0, 0x1f, 0xc0, 0x04, 0x41, 0x03, 0xfc, 
0x18, 0x07, 0x80, 0xdf, 0xff, 0xff, 0xff, 0x04, 0x63, 0x3f, 0xf8, 0x30, 0x07, 0x80, 0xc0, 0x7f, 
0xfc, 0x00, 0x00, 0x37, 0xff, 0xf0, 0x60, 0x0f, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 
0xf0, 0x60, 0x0f, 0x80, 0xfc, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xe0, 0xc0, 0x1f, 0x80, 0xff, 
0xff, 0xc7, 0xff, 0xff, 0xff, 0xff, 0xc1, 0x80, 0x3f, 0x80, 0xff, 0x07, 0xff, 0xff, 0xff, 0xff, 
0xff, 0x83, 0x00, 0x7f, 0x80, 0xff, 0x80, 0x0f, 0xff, 0xff, 0xff, 0xff, 0x0e, 0x00, 0xff, 0x80, 
0xff, 0xc0, 0x03, 0xff, 0xff, 0xff, 0xfc, 0x1c, 0x01, 0xff, 0x80, 0xff, 0xe0, 0x01, 0xff, 0xff, 
0xff, 0xf8, 0x30, 0x03, 0xff, 0x80, 0xff, 0xf8, 0x00, 0xff, 0xff, 0xff, 0xe0, 0xe0, 0x07, 0xff, 
0x80, 0xff, 0xfe, 0x00, 0x3f, 0xff, 0xff, 0xc3, 0x80, 0x0f, 0xff, 0x80, 0xff, 0xff, 0x80, 0x0f, 
0xff, 0xff, 0x0f, 0x00, 0x3f, 0xff, 0x80, 0xff, 0xff, 0xf0, 0x01, 0xff, 0xf8, 0x7c, 0x00, 0x7f, 
0xff, 0x80, 0xff, 0xff, 0xff, 0x00, 0x3f, 0xc3, 0xe0, 0x01, 0xff, 0xff, 0x80, 0xff, 0xff, 0xe7, 
0xff, 0xfe, 0x7f, 0x80, 0x07, 0xff, 0xff, 0x80, 0xff, 0xff, 0xfc, 0x7f, 0xff, 0xf8, 0x00, 0x1f, 
0xff, 0xff, 0x80, 0xff, 0xff, 0xff, 0x00, 0xfc, 0x00, 0x00, 0xff, 0xff, 0xff, 0x80, 0xff, 0xff, 
0xff, 0xf0, 0x00, 0x00, 0x07, 0xff, 0xff, 0xff, 0x80, 0xff, 0xff, 0xff, 0xff, 0x80, 0x00, 0xff, 
0xff, 0xff, 0xff, 0x80
};

// ---------------------------
// Helper: Draw Multi-Line Text
// ---------------------------
void drawMultilineText(const String lines[], int numLines, int startY = 0, int textSize = 1, int spacing = 4) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(textSize);

  int lineHeight = 8 * textSize;
  for (int i = 0; i < numLines; i++) {
    display.setCursor(0, startY + i * (lineHeight + spacing));
    display.println(lines[i]);
  }

  display.display();
}

// ---------------------------
// Show Startup Info on OLED
// ---------------------------
void showStartupScreen(const char* ssid, IPAddress ip) {
  display.clearDisplay();
  display.drawBitmap((SCREEN_WIDTH - 81) / 2, 2, Car_logo, 81, 60, SSD1306_WHITE);
  display.display();
  delay(3000);

  String lines[] = {
    "WiFi: Connected",
    "SSID: " + String(ssid),
    "IP: " + ip.toString(),
    "MQTT: " + String(mqtt_server),
    "Happy parking"
  };
  drawMultilineText(lines, 5, 0, 1, 2);
  delay(3000);
}

// ---------------------------
// Shift Register Update
// ---------------------------
void updateShiftRegister(uint8_t status) {
  digitalWrite(LATCH_PIN, LOW);
  shiftOut(DATA_PIN, CLOCK_PIN, MSBFIRST, status);
  digitalWrite(LATCH_PIN, HIGH);
}

// ---------------------------
// MQTT Callback
// ---------------------------
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.println("Message received on topic: " + String(topic));
  
  if (length >= 512) {
    Serial.println("Message too large");
    return;
  }

  // Print raw payload for debugging
  char message[512];
  memcpy(message, payload, length);
  message[length] = '\0';
  
  Serial.println("Raw payload: " + String(message));

  // Parse the JSON payload
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, message);
  
  if (error) {
    Serial.println("Failed to parse JSON: " + String(error.c_str()));
    return;
  }

  // Debug: print parsed JSON
  String jsonStr;
  serializeJsonPretty(doc, jsonStr);
  Serial.println("Parsed JSON: " + jsonStr);

  uint8_t parkingStatus = 0;
  int freeSpaces = 0;
  int firstFreeRow1 = -1;
  int firstFreeRow2 = -1;
  
  // Expected format based on previous working code:
  // {"slot1": 1, "slot2": 1, "slot3": 0, ...} where 1=occupied, 0=free

  // Process all slots (1-8)
  for (int i = 0; i < NUM_SLOTS; i++) {
    String slotKey = "slot" + String(i + 1);
    
    // Check if slot exists in the JSON
    if (doc.containsKey(slotKey)) {
      int status = doc[slotKey]; // 1 = occupied, 0 = free
      
      if (status == 0) {  // Free slot
        freeSpaces++;
        // Set the bit for this slot (RED LED for free slot)
        parkingStatus &= ~(1 << i);
        
        // Track first free slots by row
        if (i < 4 && firstFreeRow1 == -1) {
          firstFreeRow1 = i;  // First free slot in Row 1 (0-indexed)
        } else if (i >= 4 && firstFreeRow2 == -1) {
          firstFreeRow2 = i;  // First free slot in Row 2 (0-indexed)
        }
        
        Serial.println("Slot " + slotKey + " is FREE");
      } else {  // Occupied slot
        // Clear the bit for this slot (GREEN LED for occupied slot)
        parkingStatus |= (1 << i);
        Serial.println("Slot " + slotKey + " is OCCUPIED");
      }
    }
  }
  
  // Calculate closest free slot (using logic from previous working code)
  int closestFreeSlot = -1;
  if (firstFreeRow1 != -1 || firstFreeRow2 != -1) {
    int distanceRow1 = (firstFreeRow1 != -1) ? abs(0 - firstFreeRow1) : INT_MAX;
    int distanceRow2 = (firstFreeRow2 != -1) ? abs(4 - firstFreeRow2) : INT_MAX;

    if (distanceRow1 <= distanceRow2) {
      closestFreeSlot = firstFreeRow1 + 1;  // Convert to 1-indexed
    } else {
      closestFreeSlot = firstFreeRow2 + 1;  // Convert to 1-indexed
    }
  }

  int occupiedSpaces = NUM_SLOTS - freeSpaces;
  
  Serial.println("Free spaces: " + String(freeSpaces));
  Serial.println("Occupied spaces: " + String(occupiedSpaces));
  Serial.println("Closest free slot: " + String(closestFreeSlot));
  Serial.println("LED status byte: " + String(parkingStatus, BIN));

  // Update global variables to be used elsewhere in the program
  g_freeSpaces = freeSpaces;
  g_closestFreeSlot = closestFreeSlot;

  // Update LEDs via shift register
  updateShiftRegister(parkingStatus);

  // Display info on OLED
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  
  // Show status summary
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.print("No. of free");
  display.setCursor(0, 16);
  display.print("slots: ");
  display.println(g_freeSpaces);

  // Show closest free slot if available
  display.setCursor(0, 36);
  display.print("Nearest free");
  display.setCursor(0, 52);
  if (g_freeSpaces > 0) {
    display.print("slot: ");
    display.print(g_closestFreeSlot);
  } else {
    display.print("slot: None");
  }
  display.display();
}

// ---------------------------
// MQTT Reconnection
// ---------------------------
void reconnectMQTT() {
  // Try to reconnect to MQTT broker
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting to MQTT...");
  display.println(mqtt_server);
  display.display();
  
  int attempt = 0;
  while (!client.connected() && attempt < 5) {
    Serial.println("Attempting MQTT connection...");
    String clientId = "ESP8266Client-" + String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("Connected to MQTT broker");
      
      // Subscribe to the parking status topic
      if (client.subscribe(mqtt_topic, 1)) {  // QoS 1
        Serial.println("Subscribed to topic: " + String(mqtt_topic));
      } else {
        Serial.println("Failed to subscribe to topic");
      }
      
      display.clearDisplay();
      display.setCursor(0, 0);
      display.println("MQTT Connected!");
      display.println("Topic: " + String(mqtt_topic));
      display.display();
      delay(1000);
      return;
    } else {
      Serial.print("MQTT connection failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      
      display.setCursor(0, 30);
      display.println("Failed! Retry " + String(attempt+1) + "/5");
      display.display();
      
      delay(5000);
      attempt++;
    }
  }
  
  if (!client.connected()) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("MQTT connection failed");
    display.println("Will retry later...");
    display.display();
    delay(1000);
  }
}

// ---------------------------
// WiFi Setup
// ---------------------------
void connectWiFi() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting to WiFi");
  display.println(ssid);
  display.display();
  
  WiFi.begin(ssid, password);
  
  int dots = 0;
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    
    display.setCursor(0, 20);
    String progress = "Progress: ";
    for (int i = 0; i < dots % 10; i++) {
      progress += ".";
    }
    display.println(progress);
    display.display();
    dots++;
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
    
    showStartupScreen(ssid, WiFi.localIP());
  } else {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi connection failed!");
    display.println("Check credentials");
    display.println("or network status.");
    display.display();
    delay(3000);
  }
}

// ---------------------------
// Setup
// ---------------------------
void setup() {
  Serial.begin(115200);
  Serial.println("\n\n===== Starting Smart Parking System =====");

  // Initialize shift register pins
  pinMode(DATA_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);
  pinMode(LATCH_PIN, OUTPUT);
  
  // Set all LEDs to on initially (test LEDs)
  updateShiftRegister(0xFF);

  // Initialize I2C for OLED
  Wire.begin(D2, D1);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Smart Parking System");
  display.println("Initializing...");
  display.display();
  delay(2000);
  
  // Turn off all LEDs after test
  updateShiftRegister(0x00);
  delay(500);

  // Connect to WiFi
  connectWiFi();
  
  // Setup MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);
  client.setKeepAlive(30);  // Set keepalive to 30 seconds
  
  // Connect to MQTT broker
  reconnectMQTT();
  
  Serial.println("Setup complete");
}

// ---------------------------
// Main Loop
// ---------------------------
void loop() {
  // Check MQTT connection
  if (!client.connected()) {
    Serial.println("MQTT disconnected");
    reconnectMQTT();
  }
  
  // Process incoming messages
  client.loop();
  
  // Show logo every 9 seconds
  static unsigned long lastLogoTime = 0;
  if (millis() - lastLogoTime > 9000) {  // Every 5 seconds
    lastLogoTime = millis();
    
    // Show logo
    display.clearDisplay();
    display.drawBitmap((SCREEN_WIDTH - 81) / 2, 2, Car_logo, 81, 60, SSD1306_WHITE);
    display.display();
    delay(2000);  // Show for 2 seconds
    
    // Restore the parking information display
    // We need to trigger a "fake" update to refresh the display
    // This uses the global variables to repopulate the screen
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    
    display.setTextSize(2); //change the font size according to your OLED display screen size
    display.setCursor(0, 0);
    display.print("No. of free");
    display.setCursor(0, 16);
    display.print("slots: ");
    display.println(g_freeSpaces);
    
    display.setCursor(0, 36);
    display.print("Nearest free");
    display.setCursor(0, 52);
    if (g_freeSpaces > 0) {
      display.print("slot: ");
      display.print(g_closestFreeSlot);
    } else {
      display.print("slot: None");
    }
    display.display();
  }
  
  // Check WiFi connection periodically
  static unsigned long lastWifiCheck = 0;
  if (millis() - lastWifiCheck > 30000) {  // Check every 30 seconds
    lastWifiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected, reconnecting...");
      connectWiFi();
      reconnectMQTT();
    }
  }
  
  // Add a small delay to prevent hogging CPU
  delay(100);
}