# 🚗 Deep Learning-Powered Autonomous Parking Solution with IoT-Enabled Real-Time Management

## 📝 Project Overview

A cutting-edge Parking Management System that leverages computer vision and IoT technologies to provide real-time parking slot monitoring, booking, and management. Utilizing YOLO11 for vehicle detection and a robust Flask backend, this system offers an intelligent solution for parking space optimization.

## ✨ Key Features

- **Real-time Vehicle Detection**: Advanced YOLO11 deep learning model
- **Automated Slot Tracking**: Dynamic parking slot status updates
- **User Management**: Role-based login (Client & Management)
- **Booking System**: Reserve and release parking slots
- **Wrong Parking Detection**: Intelligent tracking of booked vs. actual parking
- **SMS Notifications**: Real-time alerts via Twilio
- **Hardware Integration**: NodeMCU control for slot indicators
- **Web Dashboard**: User-friendly interface for parking management

## 🚀 Tech Stack

- **Backend**: Flask
- **Deep Learning**: YOLO11
- **Database**: SQLite
- **IoT**: NodeMCU (ESP8266/ESP32)
- **Communication**: MQTT, WebSockets
- **Notifications**: Twilio SMS
- **Frontend**: HTML, JavaScript

## 🔧 System Architecture

1. **Computer Vision Detection**
   - YOLO11 processes camera feed
   - Identifies vehicle presence
   - Updates slot status in real-time

2. **Web Application**
   - User registration/login
   - Parking slot booking
   - Real-time slot availability
   - Management dashboard

3. **Hardware Control**
   - NodeMCU updates OLED display
   - Controls slot indicator lights
   - Receives MQTT updates

## 📦 Installation

### Prerequisites
- Python 3.8+
- Arduino IDE
- MQTT Broker (Mosquitto)
- Twilio Account

### Quick Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/SHIVALING-KORALLI/DL-AutoParking-IoT.git
   cd DL-AutoParking-IoT
   ```

2. **Install Dependencies**
   ```bash
   cd server
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Copy `.env.example` to `.env`
   - Update Twilio, MQTT, and other credentials

4. **Run Application**
   ```bash
   python app.py
   ```

## 🌐 Access Points

- **Web App**: `http://localhost:5000`
- **Client Dashboard**: Parking slot booking
- **Management Dashboard**: User & slot management

## 🔍 Unique Innovations

- Automated booking expiry
- Wrong parking detection
- Real-time hardware integration
- Comprehensive logging system

## 🚧 Future Roadmap

- [ ] Cloud deployment
- [ ] Mobile application
- [ ] Enhanced ML model accuracy
- [ ] Advanced analytics dashboard

## 📊 System Requirements

- **Server**: 4 GB RAM, 2 Core CPU
- **Storage**: 20 GB
- **Camera**: HD resolution (minimum)
- **Network**: Stable WiFi/Ethernet

## 📜 License

MIT License - See `LICENSE` file for details

## 🤝 Contributions

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 👨‍💻 Author

**Shivaling Koralli**
- GitHub: [@SHIVALING-KORALLI](https://github.com/SHIVALING-KORALLI)

---

**⭐ Don't forget to star this repository if you find it helpful!**