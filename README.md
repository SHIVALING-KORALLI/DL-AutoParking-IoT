<!-- HEADER -->
<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&center=true&vCenter=true&width=500&lines=🚗+Deep+Learning-Powered+Autonomous+Parking+System;IoT-Enabled+Real-Time+Smart+City+Management;Built+with+YOLO11n+%2B+ESP8266+%2B+Flask" alt="Typing SVG" />
</h1>

<p align="center">
  <a href="https://github.com/SHIVALING-KORALLI/DL-AutoParking-IoT/stargazers"><img src="https://img.shields.io/github/stars/SHIVALING-KORALLI/DL-AutoParking-IoT?style=for-the-badge&color=yellow" /></a>
  <a href="https://github.com/SHIVALING-KORALLI/DL-AutoParking-IoT/network/members"><img src="https://img.shields.io/github/forks/SHIVALING-KORALLI/DL-AutoParking-IoT?style=for-the-badge&color=orange" /></a>
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLOv11n-Deep%20Learning-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/IoT-NodeMCU%20%7C%20MQTT-red?style=for-the-badge" />
</p>

---

## 🧭 Abstract

A **Deep Learning-powered autonomous parking system** integrating **YOLO11n**, **IoT communication**, and **edge hardware** to redefine smart-city parking.  
It replaces conventional sensors with a **camera-based vision pipeline**, **MQTT protocol**, and **NodeMCU + 74HC595 hardware**, achieving:

- 🎯 **94–97 % accuracy**  
- ⚡ **458 ms latency** (CPU inference)  
- 💰 **60–70 % cost reduction** vs. sensor-per-slot models  

> Published & Presented in **ICDTE 2025 (Springer Series)** — Research-backed innovation for scalable smart parking infrastructure.

---

## 🌟 Key Highlights

| Category | Features |
|-----------|-----------|
| 🧠 **Computer Vision** | YOLO11n multi-class detection (Cars / Bikes / Plates), OCR pipeline with adaptive preprocessing |
| 🌐 **IoT Layer** | MQTT-based real-time hardware sync via ESP8266 (NodeMCU) |
| 💡 **Hardware Design** | Shift-register (74HC595) LED arrays + 0.96″ OLED status display |
| 📊 **Analytics** | Hybrid **ARIMA + Linear Regression** forecasting (8.3 % MAPE) |
| 🔒 **Web System** | Flask + Socket.IO dashboards for clients & admins |
| 🔔 **Automation** | SMS alerts (Twilio), auto-release bookings, violation detection |
| ⚙️ **Cloud Access** | Cloudflare Zero Trust tunnel for secure public access |

---

## 🧩 Architecture Overview

<p align="center">
  <img src="https://github.com/SHIVALING-KORALLI/DL-AutoParking-IoT/assets/system-architecture-diagram.png" alt="System Architecture" width="85%">
</p>

1. **Detection Module:** YOLO11n → OCR → Slot mapping  
2. **Web Backend:** Flask + Socket.IO for real-time updates  
3. **IoT Control:** MQTT ↔ NodeMCU ↔ 74HC595 LED arrays  
4. **Forecasting & Analytics:** ARIMA-LR models for demand prediction  
5. **Public Access:** Cloudflare tunnel → Secure domain + QR-based access  

---

## ⚙️ Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Deep Learning** | YOLO11n, OpenCV, EasyOCR |
| **Backend** | Python, Flask, Socket.IO, SQLAlchemy |
| **IoT / Hardware** | NodeMCU (ESP8266), MQTT, 74HC595, OLED (I²C) |
| **Frontend** | HTML / CSS / JavaScript + Chart.js |
| **Forecasting & Analytics** | ARIMA, Linear Regression, Pandas, WeasyPrint |
| **Notifications & Deployment** | Twilio SMS API, Cloudflare Zero Trust |

---

## 🧱 Installation

### Prerequisites
- Python 3.8 +  
- Arduino IDE  
- MQTT Broker (e.g., Mosquitto)  
- Twilio Account (for SMS alerts)

### Setup
```bash
git clone https://github.com/SHIVALING-KORALLI/DL-AutoParking-IoT.git
cd DL-AutoParking-IoT/server
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
