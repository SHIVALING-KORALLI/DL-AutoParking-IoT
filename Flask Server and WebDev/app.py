#app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from twilio.rest import Client
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
import os
import json
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import threading
from threading import Lock
import shutil
import logging
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app)

# Load environment variables
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    raise EnvironmentError("Twilio environment variables not set properly.")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# XML configuration
XML_FOLDER = r"C:\Users\pinky\OneDrive\Documents\Desktop\server\parking_history"
os.makedirs(XML_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

xml_lock = Lock()

def get_xml_file_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(XML_FOLDER, f"parking_{date_str}.xml")

def create_xml_if_not_exists():
    xml_file = get_xml_file_path()
    if not os.path.exists(xml_file):
        root = ET.Element("ParkingSlots")
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)

def parse_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None

def is_duplicate_entry(last_slot, new_timestamp, time_threshold_seconds=60):
    """
    Check if this is a duplicate entry based on time threshold
    """
    if last_slot is None:
        return False
        
    last_entry_time = parse_timestamp(last_slot.get("entry_time"))
    new_entry_time = parse_timestamp(new_timestamp)
    
    if last_entry_time and new_entry_time:
        time_diff = abs((new_entry_time - last_entry_time).total_seconds())
        return time_diff < time_threshold_seconds
    
    return False

def update_parking_slot_xml(slot_id, status, vehicle_type, plate_number, timestamp, client_name):
    create_xml_if_not_exists()
    xml_file = get_xml_file_path()
    temp_file = xml_file + ".tmp"

    with xml_lock:
        # Normalize input values
        vehicle_type = vehicle_type or "Unknown"
        plate_number = plate_number or "Unknown"
        timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        client_name = client_name or "Unknown"

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            logger.error(f"Corrupt XML detected. Recreating {xml_file}")
            os.remove(xml_file)
            create_xml_if_not_exists()
            tree = ET.parse(xml_file)
            root = tree.getroot()

        # Find all slots with the same ID
        slots = root.findall(f"./Slot[@id='{slot_id}']")
        last_slot = slots[-1] if slots else None

        # Case 1: Slot becoming occupied (Free -> Occupied)
        if status == "Occupied":
            # Check if there's no previous entry, or if the previous entry has an exit time 
            # (completed cycle), or if previous entry is already Free
            needs_new_entry = (
                last_slot is None or 
                "exit_time" in last_slot.attrib or 
                last_slot.get("status") != "Occupied"
            )
            
            if needs_new_entry:
                # Create new occupation entry with the current timestamp as entry_time
                new_slot = ET.SubElement(root, "Slot", {
                    "id": slot_id,
                    "status": status,
                    "vehicle_type": vehicle_type,
                    "plate_number": plate_number,
                    "entry_time": timestamp,
                    "client_name": client_name
                })
                logger.info(f"Created new occupation entry for slot {slot_id}")
            else:
                # Update only the allowed fields (NOT entry_time) for an existing incomplete cycle
                if plate_number and plate_number != "Unknown":
                    last_slot.set("plate_number", plate_number)
                if vehicle_type and vehicle_type != "Unknown":
                    last_slot.set("vehicle_type", vehicle_type)
                if client_name and client_name != "Unknown":
                    last_slot.set("client_name", client_name)
                logger.info(f"Updated existing occupation entry for slot {slot_id}")

        # Case 2: Slot becoming free (Occupied -> Free)
        elif status == "Free" and last_slot is not None:
            if last_slot.get("status") == "Occupied" and "exit_time" not in last_slot.attrib:
                # Add exit time to the existing entry
                last_slot.set("exit_time", timestamp)
                logger.info(f"Updated exit time for slot {slot_id}")

        # Safely write to a temporary file and replace the original
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)
        shutil.move(temp_file, xml_file)
        logger.info(f"Successfully updated parking data for slot {slot_id}")

class MQTTManager:
    def __init__(self, broker_host="localhost", broker_port=1883):
        self.broker_host = os.getenv('MQTT_BROKER_HOST', broker_host)
        self.broker_port = int(os.getenv('MQTT_BROKER_PORT', broker_port))
        self.client = mqtt.Client()
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect  # Add disconnect handler
        self.booking_confirmations = {}
        self.is_connected = False
        self.reconnect_timer = None
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker successfully")
            self.is_connected = True
            # Clear any reconnection timer
            if self.reconnect_timer:
                self.reconnect_timer.cancel()
                self.reconnect_timer = None
            # Subscribe to topics
            client.subscribe("parking/commands")
            client.subscribe("parking/booking_confirm")
        else:
            print(f"Failed to connect to MQTT broker with result code {rc}")
            self.is_connected = False
            self.schedule_reconnect()

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        if rc != 0:
            print(f"Unexpected disconnection from MQTT broker with code {rc}")
            self.schedule_reconnect()
        else:
            print("Disconnected from MQTT broker normally")

    def on_message(self, client, userdata, msg):
        print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
        if msg.topic == "parking/booking_confirm":
            try:
                data = json.loads(msg.payload.decode())
                slot_id = data.get('slot_id')
                status = data.get('status')
                if slot_id and status:
                    self.booking_confirmations[slot_id] = status
                    # Handle the confirmation in the application
                    self.handle_booking_confirmation(slot_id, status)
            except json.JSONDecodeError:
                print("Failed to decode booking confirmation message")

    def handle_booking_confirmation(self, slot_id, status):
        if slot_id in parking_slots:
            if status == "confirmed":
                # Update the slot status if needed
                socketio.emit('booking_confirmed', {'slot_id': slot_id})
            else:
                # Reset the slot if booking failed
                parking_slots[slot_id].update({
                    'status': 'Free',
                    'vehicle_type': '',
                    'client_name': '',
                    'timestamp': ''
                })
                socketio.emit('booking_failed', {
                    'slot_id': slot_id,
                    'message': 'Booking could not be confirmed by the system'
                })
    
    def schedule_reconnect(self):
        """Schedule a reconnection attempt after 30 seconds"""
        import threading
        if not self.reconnect_timer:
            print("Scheduling MQTT reconnection in 30 seconds...")
            self.reconnect_timer = threading.Timer(30.0, self.connect)
            self.reconnect_timer.daemon = True  # Allow program to exit even if timer is running
            self.reconnect_timer.start()
    
    def connect(self):
        """Connect to the MQTT broker with retry mechanism"""
        try:
            if self.reconnect_timer:
                self.reconnect_timer.cancel()
                self.reconnect_timer = None
            
            print(f"Attempting to connect to MQTT broker at {self.broker_host}:{self.broker_port}...")
            # Set a shorter connection timeout
            self.client.connect_async(self.broker_host, self.broker_port, 10)
            self.client.loop_start()
            return True
        except Exception as e:
            self.is_connected = False
            print(f"Failed to connect to MQTT broker: {e}")
            self.schedule_reconnect()
            return False

    def publish_parking_status(self, slots):
        """Publish parking status and attempt reconnect if disconnected"""
        if not self.is_connected:
            print("Not connected to MQTT broker. Attempting to reconnect...")
            self.connect()
            return
            
        try:
            status_data = {slot_id: 1 if details["status"].lower() == "occupied" else 0 
                          for slot_id, details in slots.items()}
            message = json.dumps(status_data)
            result = self.client.publish("parking/slots/status", message)
            if result.rc == 0:
                print(f"Published parking status: {message}")
            else:
                print(f"Failed to publish message, result code: {result.rc}")
        except Exception as e:
            print(f"Failed to publish parking status: {e}")
            # If we hit an exception during publish, try to reconnect
            self.is_connected = False
            self.schedule_reconnect()
    
    def cleanup(self):
        """Clean up resources properly"""
        if self.reconnect_timer:
            self.reconnect_timer.cancel()
            self.reconnect_timer = None
            
        self.client.loop_stop()
        self.client.disconnect()

# Initialize the MQTT manager
mqtt_manager = MQTTManager()
mqtt_manager.connect()

# First, let's add the new fields to the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    phone_number = db.Column(db.String(15), unique=True, nullable=False)
    number_plate = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    parking_incidents = db.Column(db.Integer, default=0)
    last_incident_reset = db.Column(db.DateTime, default=datetime.now)

# Add configuration for incident reset period (in days)
INCIDENT_RESET_PERIOD = 30  # Can be modified as needed

def check_and_reset_incidents(user):
    """Reset incident count if reset period has passed"""
    if user and user.last_incident_reset:
        days_passed = (datetime.now() - user.last_incident_reset).days
        if days_passed >= 5:  # Reset after 5 days
            user.parking_incidents = 0
            user.last_incident_reset = datetime.now()
            db.session.commit()
            logger.info(f"Reset parking incidents for user {user.name}")

def get_slot_by_plate(plate_number):
    """Find slot where a specific plate number is parked"""
    for slot_id, slot in parking_slots.items():
        if slot.get('plate_number') == plate_number:
            return slot_id, slot
    return None, None

def handle_wrong_parking(booked_slot_id, actual_slot_id, user):
    """Handle case where user parked in wrong slot"""
    try:
        logger.info(f"Handling wrong parking for user {user.name}")
        
        # Update incident count
        check_and_reset_incidents(user)
        user.parking_incidents += 1
        db.session.commit()
        logger.info(f"Updated incident count for {user.name} to {user.parking_incidents}")

        # Prepare warning message
        warning_msg = ""
        if user.parking_incidents >= 5:
            warning_msg = " WARNING: You have reached the maximum number of incorrect parking incidents. You may lose parking privileges."
        else:
            remaining = 5 - user.parking_incidents
            warning_msg = f" WARNING: You have {user.parking_incidents} incorrect parking incidents. After {remaining} more incidents, you may lose parking privileges."

        # Send SMS notification
        message = (
            f"Incorrect Parking Detected!\n"
            f"You booked {booked_slot_id} but parked in {actual_slot_id}.\n"
            f"Your booked slot has been released.\n{warning_msg}"
        )
        send_sms(user.phone_number, message)
        logger.info(f"Sent wrong parking notification to user {user.name}: {message}")

        # Update booked slot status (release it)
        if booked_slot_id in parking_slots:
            parking_slots[booked_slot_id].update({
                'status': 'Free',
                'vehicle_type': '',
                'client_name': '',
                'plate_number': '',
                'timestamp': ''
            })
            
            # Update XML for released slot
            update_parking_slot_xml(
                slot_id=booked_slot_id,
                status='Free',
                vehicle_type='',
                plate_number='',
                timestamp=datetime.now().isoformat(),
                client_name=''
            )
            logger.info(f"Released booked slot {booked_slot_id}")

        # Do NOT transfer booking - only update client name in actual slot for record
        if actual_slot_id in parking_slots:
            # Just ensure the client name is updated in the actual occupied slot
            client_name = user.name if user else ''
            current_status = parking_slots[actual_slot_id].get('status', 'Occupied')
            
            # Keep the detection data but add client name
            parking_slots[actual_slot_id]['client_name'] = client_name
            
            logger.info(f"Updated actual parked slot {actual_slot_id} client name to {client_name}")

        # Emit updates
        socketio.emit('update_slots_and_frame', {
            'frame': current_frame,
            'slots': parking_slots
        })
        socketio.emit('update_slots_client', {'slots': parking_slots})
        socketio.emit('wrong_parking_detected', {
            'booked_slot': booked_slot_id,
            'actual_slot': actual_slot_id,
            'user': user.name,
            'incidents': user.parking_incidents
        })
        mqtt_manager.publish_parking_status(parking_slots)

    except Exception as e:
        logger.error(f"Error in handle_wrong_parking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.session.rollback()



#app.py
# Fix the wrong parking check function to only focus on booked slots
def check_wrong_parking():
    """Check for vehicles parked in wrong slots"""
    try:
        logger.info("Running wrong parking check...")
        
        # First find all booked slots with client names
        booked_slots = {}
        for slot_id, slot in parking_slots.items():
            # FIXED: Only consider slots that have both client_name AND timestamp
            # as officially booked slots
            if (slot['status'].lower() == 'occupied' and 
                slot.get('client_name') and 
                slot.get('client_name') != '' and
                slot.get('timestamp') and 
                slot.get('timestamp') != ''):
                booked_slots[slot_id] = slot
        
        logger.info(f"Found {len(booked_slots)} booked slots")
        
        # Now check each booked slot
        for booked_slot_id, booked_slot in booked_slots.items():
            # Get user information
            user = User.query.filter_by(name=booked_slot['client_name']).first()
            if not user or not user.number_plate:
                logger.info(f"No user found for slot {booked_slot_id} with client name {booked_slot['client_name']}")
                continue
                
            # Find where this user's car is actually parked
            actual_slot_id = None
            for slot_id, slot in parking_slots.items():
                if (slot.get('plate_number') == user.number_plate and 
                    slot.get('plate_number') != '' and 
                    slot.get('status', '').lower() == 'occupied'):
                    actual_slot_id = slot_id
                    break
            
            # Log the check for debugging
            logger.info(f"Checking user {user.name} with plate {user.number_plate}:")
            logger.info(f"  - Booked slot: {booked_slot_id}")
            logger.info(f"  - Actual slot: {actual_slot_id}")
            
            # If car is found and it's in a different slot than booked
            if actual_slot_id and actual_slot_id != booked_slot_id:
                logger.info(f"WRONG PARKING DETECTED: User {user.name} booked {booked_slot_id} but parked in {actual_slot_id}")
                handle_wrong_parking(booked_slot_id, actual_slot_id, user)
                
    except Exception as e:
        logger.error(f"Error in check_wrong_parking: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Updated parking slots data structure
parking_slots = {f"slot{i}": {
    "slot_id": f"slot{i}",
    "status": "Free",
    "vehicle_type": "",
    "plate_number": "",
    "client_name": "",
    "timestamp": ""
} for i in range(1, 9)}

current_frame = None

def send_sms(to_number, message):
    try:
        twilio_client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=to_number)
    except Exception as e:
        print(f"SMS sending failed: {e}")

# Updated auto_release_spots function
def auto_release_spots():
    with app.app_context():
        while True:
            socketio.sleep(5)  # Check every 5 seconds
            
            try:
                # First check for wrong parking
                check_wrong_parking()
                
                # Then handle auto-release
                current_time = datetime.now()
                logger.info(f"Auto-release check at {current_time}")
                
                for slot_id, slot in parking_slots.items():
                    if slot['status'].lower() == 'occupied' and slot.get('client_name'):
                        timestamp_str = slot.get('timestamp', '')
                        booking_duration = int(slot.get('booking_duration', 3600))
                        
                        if timestamp_str:
                            try:
                                # Improved timestamp parsing with better error handling
                                try:
                                    # First try standard ISO format parsing
                                    slot_time = datetime.fromisoformat(timestamp_str)
                                except ValueError:
                                    # Fall back to alternative parsing as backup
                                    logger.warning(f"Failed to parse timestamp {timestamp_str} using fromisoformat, trying alternative")
                                    slot_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
                                
                                time_diff = (current_time - slot_time).total_seconds()
                                
                                # Debug logging
                                logger.info(f"Slot {slot_id}: Current time: {current_time.isoformat()}, Slot time: {slot_time.isoformat()}")
                                logger.info(f"Time difference: {time_diff}s, Booking duration: {booking_duration}s")
                                
                                if time_diff >= booking_duration:
                                    logger.info(f"Auto-releasing slot {slot_id} after {time_diff}s (duration was {booking_duration}s)")
                                    
                                    # Get user for notification
                                    user = User.query.filter_by(name=slot['client_name']).first()
                                    if user:
                                        send_sms(user.phone_number, f"Your booking for {slot_id} has expired and has been released.")
                                    
                                    # Release the slot completely
                                    slot.update({
                                        'status': 'Free',  # Always set to Free when booking expires
                                        'client_name': '',
                                        'timestamp': '',
                                        'booking_duration': 3600,  # Reset to default
                                        'plate_number': '',  # Clear plate number
                                        'vehicle_type': ''  # Clear vehicle type
                                    })
                                    
                                    # Update XML
                                    update_parking_slot_xml(
                                        slot_id=slot_id,
                                        status='Free',
                                        vehicle_type='',
                                        plate_number='',
                                        timestamp=datetime.now().isoformat(),
                                        client_name=''
                                    )
                                    
                                    mqtt_manager.publish_parking_status(parking_slots)
                                    socketio.emit('slot_auto_released', {'slot_id': slot_id})
                                    socketio.emit('update_slots_client', {'slots': parking_slots})
                                    socketio.emit('update_slots_and_frame', {
                                        'frame': current_frame,
                                        'slots': parking_slots
                                    })
                            except ValueError as e:
                                logger.error(f"Error parsing timestamp for slot {slot_id}: {e}")
                            except Exception as e:
                                logger.error(f"Error in auto-release for slot {slot_id}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                
            except Exception as e:
                logger.error(f"Error in auto_release_spots: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

# Additional Socket.IO event handler to handle explicit requests for slot updates
@socketio.on('get_slots')
def handle_get_slots():
    """
    Handle explicit requests for slot updates from clients.
    """
    try:
        logger.info(f"Client {request.sid} requested slots update")
        socketio.emit('update_slots_client', {'slots': parking_slots}, room=request.sid)
        
        # If you have a current frame, send that too
        if current_frame:
            socketio.emit('update_slots_and_frame', {
                'frame': current_frame,
                'slots': parking_slots
            }, room=request.sid)
    except Exception as e:
        logger.error(f"Error handling get_slots request: {e}")
            
@socketio.on('connect')
def handle_connect():
    # Send current parking slots state to newly connected client
    socketio.emit('update_slots_client', {'slots': parking_slots}, room=request.sid)
    
    # If you have a current frame, send that too
    if current_frame:
        socketio.emit('update_slots_and_frame', {
            'frame': current_frame,
            'slots': parking_slots
        }, room=request.sid)
    
    logger.info(f"Client connected: {request.sid}, sent current state with {len(parking_slots)} slots")

@socketio.on('frame_and_slots')
def handle_frame_and_slots(data):
    global current_frame
    current_frame = data.get('frame')
    slots = data.get('slots', {})
    update_type = data.get('update_type', 'full')
    
    # Log the update type
    if update_type == 'partial':
        logger.info(f"Received partial update for {len(slots)} slots")
    else:
        logger.info(f"Received full update with {len(slots)} slots")
    
    for slot_id, details in slots.items():
        if slot_id in parking_slots:
            # Plate number from CV
            plate_number = details.get('plate_number', '')
            
            # Get current status and other details
            current_status = parking_slots[slot_id].get('status', 'Free')
            new_status = details.get('status', 'Free')
            current_client_name = parking_slots[slot_id].get('client_name', '')
            current_plate = parking_slots[slot_id].get('plate_number', '')
            
            # Check if this is a meaningful update
            is_meaningful_update = False
            
            # Status change is meaningful
            if current_status != new_status:
                is_meaningful_update = True
                logger.info(f"Status change for {slot_id}: {current_status} -> {new_status}")
                
                # Critical logic for slot status
                if current_client_name:
                    # If there's an active booking
                    current_time = datetime.now()
                    timestamp_str = parking_slots[slot_id].get('timestamp', '')
                    
                    if timestamp_str:
                        try:
                            slot_time = datetime.fromisoformat(timestamp_str)
                            booking_duration = int(parking_slots[slot_id].get('booking_duration', 3600))
                            time_diff = (current_time - slot_time).total_seconds()
                            
                            # If booking has expired AND no vehicle is present, release the slot
                            if time_diff >= booking_duration and new_status == 'Free':
                                logger.info(f"Slot {slot_id} booking expired and no vehicle present - fully releasing")
                                parking_slots[slot_id]['status'] = 'Free'
                                parking_slots[slot_id]['client_name'] = ''
                                parking_slots[slot_id]['timestamp'] = ''
                                parking_slots[slot_id]['plate_number'] = ''
                                parking_slots[slot_id]['vehicle_type'] = ''
                            elif time_diff < booking_duration:
                                # Booking still active, preserve status
                                parking_slots[slot_id]['status'] = current_status
                            
                        except Exception as e:
                            logger.error(f"Error checking booking status for {slot_id}: {e}")
                else:
                    # No active booking, set status normally
                    parking_slots[slot_id]['status'] = new_status
            
            # Plate number change is meaningful
            if plate_number and plate_number != "Unknown" and current_plate != plate_number:
                is_meaningful_update = True
                logger.info(f"Plate change for {slot_id}: {current_plate} -> {plate_number}")
                
                # Update the plate number
                parking_slots[slot_id]['plate_number'] = plate_number
                
                # Only update vehicle_type if plate matches booked user
                if current_client_name:
                    user = User.query.filter_by(name=current_client_name).first()
                    if user and user.number_plate == plate_number:
                        parking_slots[slot_id]['vehicle_type'] = details.get('vehicle_type', '')
            
            # Only update XML if meaningful or if this is a full update
            if is_meaningful_update or update_type == 'full':
                # Update timestamp for this meaningful change
                parking_slots[slot_id]['timestamp'] = datetime.now().isoformat()
    
                # Consider updating vehicle_type more broadly when appropriate
                if details.get('vehicle_type'):
                    parking_slots[slot_id]['vehicle_type'] = details.get('vehicle_type', '')
    
                # Update XML with current information
                update_parking_slot_xml(
                    slot_id=slot_id,
                    status=parking_slots[slot_id]['status'],
                    vehicle_type=parking_slots[slot_id].get('vehicle_type', ''),
                    plate_number=parking_slots[slot_id].get('plate_number', ''),
                    timestamp=parking_slots[slot_id]['timestamp'],
                    client_name=parking_slots[slot_id].get('client_name', '')
                )

    # Always publish to MQTT and emit to clients
    mqtt_manager.publish_parking_status(parking_slots)
    socketio.emit('update_slots_and_frame', {
        'frame': current_frame,
        'slots': parking_slots
    })
    socketio.emit('update_slots_client', {'slots': parking_slots})

# Login required decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Role required decorator
def role_required(allowed_roles):
    def decorator(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or session.get('user_role') not in allowed_roles:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator

@app.route('/get_current_user')
@login_required
def get_current_user():
    user = User.query.get(session['user_id'])
    return jsonify({"name": user.name, "phone_number": user.phone_number})

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('user_role') == 'management':
            return redirect(url_for('management_dashboard'))
        return redirect(url_for('parking_info'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "message": "No data provided"}), 400

            name = data.get('name')
            phone_number = data.get('phoneNumber')
            number_plate = data.get('numberPlate')
            password = data.get('password')
            role = data.get('role', 'client')

            # Validate required fields
            if not all([name, phone_number, number_plate, password]):
                return jsonify({"success": False, "message": "All fields are required"}), 400

            # Check if user already exists
            existing_user = User.query.filter_by(phone_number=phone_number).first()
            if existing_user:
                return jsonify({"success": False, "message": "Phone number already registered"}), 409

            # Create new user
            hashed_password = generate_password_hash(password)
            new_user = User(
                name=name,
                phone_number=phone_number,
                number_plate=number_plate,
                password=hashed_password,
                role=role
            )

            db.session.add(new_user)
            db.session.commit()

            # Send SMS notification
            try:
                send_sms(phone_number, f"Registration successful for {name}. \nWelcome to Parking Management System!")
            except Exception as sms_error:
                print(f"SMS sending failed: {sms_error}")
                # Continue even if SMS fails

            return jsonify({"success": True, "message": "Registration successful"}), 200

        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {str(e)}")
            return jsonify({"success": False, "message": "An error occurred during registration"}), 500

    # GET request - render the registration page
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        phone_number = data.get('username')
        password = data.get('password')
        role = data.get('role', 'client')
        
        print(f"Login attempt - Phone: {phone_number}, Role: {role}")  # Debug log
        
        if not phone_number or not password:
            return jsonify({"success": False, "message": "Phone number and password are required"})
        
        user = User.query.filter_by(phone_number=phone_number).first()
        
        if user:
            print(f"Found user: {user.name}, Role: {user.role}")  # Debug log
            if check_password_hash(user.password, password):
                if user.role == role:
                    session['user_id'] = user.id
                    session['user_role'] = user.role
                    session['username'] = user.name
                    return jsonify({"success": True, "role": user.role})
                else:
                    return jsonify({"success": False, "message": "Invalid role for this user"})
            else:
                print("Password mismatch")  # Debug log
                return jsonify({"success": False, "message": "Invalid password"})
        else:
            print(f"No user found with phone number: {phone_number}")  # Debug log
            return jsonify({"success": False, "message": "User not found"})
    
    return render_template('login.html')

@app.route('/manage_users', methods=['GET', 'POST', 'DELETE'])
@role_required(['management'])
def manage_users():
    if request.method == 'GET':
        users = User.query.all()
        return render_template('user_management.html', users=users)
    
    try:
        if request.method == 'DELETE':
            user_id = request.form.get('user_id')
            user = User.query.get(user_id)
            if user:
                db.session.delete(user)
                db.session.commit()
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'User not found'})
        
        elif request.method == 'POST':
            user_id = request.form.get('user_id')
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'success': False, 'message': 'User not found'})
            
            user.name = request.form.get('name', user.name)
            user.phone_number = request.form.get('phone_number', user.phone_number)
            user.number_plate = request.form.get('number_plate', user.number_plate)  # Added this line
            user.role = request.form.get('role', user.role)
            
            if request.form.get('password'):
                user.password = generate_password_hash(request.form.get('password'))
            
            db.session.commit()
            return jsonify({'success': True})
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/management')
@role_required(['management'])
def management_dashboard():
    return render_template('management.html')

@app.route('/parking_info')
@role_required(['client'])
def parking_info():
    return render_template('parking_info.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Add this new route to app.py for viewing parking history
@app.route('/view_history')
@role_required(['management'])
def view_history():
    # Get current date's XML file path
    xml_file_path = get_xml_file_path()
    
    # Check if the file exists
    if not os.path.exists(xml_file_path):
        return render_template('error.html', message="No parking history available for today.")
    
    try:
        # Read the XML content
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        
        # Parse the XML
        root = ET.fromstring(xml_content)
        
        # Convert XML data to more readable format for display
        slots_history = []
        for slot in root.findall("./Slot"):
            slot_data = slot.attrib
            slot_data['entry_time'] = slot_data.get('entry_time', 'N/A')
            slot_data['exit_time'] = slot_data.get('exit_time', 'N/A')
            slots_history.append(slot_data)
        
        # Sort by entry time (newest first)
        slots_history.sort(key=lambda x: x['entry_time'] if x['entry_time'] != 'N/A' else '', reverse=True)
        
        return render_template('history_view.html', slots_history=slots_history, date=datetime.now().strftime("%Y-%m-%d"))
    
    except Exception as e:
        logger.error(f"Error reading history XML: {e}")
        return render_template('error.html', message=f"Error reading parking history: {str(e)}")

# Download raw XML file
@app.route('/download_history_xml')
@role_required(['management'])
def download_history_xml():
    try:
        xml_file_path = get_xml_file_path()
        if not os.path.exists(xml_file_path):
            return "No history file available for today.", 404
            
        return send_file(
            xml_file_path,
            mimetype='application/xml',
            as_attachment=True,
            download_name=f"parking_history_{datetime.now().strftime('%Y-%m-%d')}.xml"
        )
    except Exception as e:
        logger.error(f"Error downloading history XML: {e}")
        return f"Error downloading file: {str(e)}", 500

# Update handle_slot_update function
@socketio.on('slot_update')
def handle_slot_update(slot_data):
    slot_id = slot_data.get('slot_id')
    if slot_id in parking_slots:
        # Make sure to include plate_number in the update
        parking_slots[slot_id].update({
            'status': slot_data.get('status', 'Free'),
            'vehicle_type': slot_data.get('vehicle_type', ''),
            'plate_number': slot_data.get('plate_number', ''),  # Include plate number
            'timestamp': slot_data.get('timestamp', '')
        })
        
        # Update XML with plate number
        update_parking_slot_xml(
            slot_id=slot_id,
            status=slot_data.get('status', 'Free'),
            vehicle_type=slot_data.get('vehicle_type', ''),
            plate_number=slot_data.get('plate_number', ''),  # Pass the plate number
            timestamp=slot_data.get('timestamp', ''),
            client_name=parking_slots[slot_id]['client_name']
        )
        
        socketio.emit('slot_status_changed', {
            'slot_id': slot_id,
            'status': parking_slots[slot_id]['status'],
            'slots': parking_slots
        })

# Update handle_book_spot function
@socketio.on('book_spot')
def handle_book_spot(data):
    spot_id = data['spot_id']
    vehicle_type = data.get('vehicle_type', 'Car')
    
    # Make sure booking_duration is an integer in seconds
    try:
        booking_duration = int(data.get('booking_duration', 3600))
        logger.info(f"Parsed booking duration: {booking_duration} seconds")
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing booking duration: {e}. Using default 3600 seconds.")
        booking_duration = 3600  # Default to 1 hour if conversion fails
    
    logger.info(f"Booking request for spot {spot_id} with duration {booking_duration} seconds")
    
    if 'user_id' not in session:
        return {'success': False, 'message': 'Please login first'}
    
    user = User.query.get(session['user_id'])
    if spot_id in parking_slots and parking_slots[spot_id]['status'].lower() == 'free':
        current_time = datetime.now()
        timestamp = current_time.isoformat()
        
        # Enhanced logging
        logger.info(f"Booking slot {spot_id} for user {user.name}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Booking duration: {booking_duration} seconds ({booking_duration/60} minutes)")
        
        # Update to include plate number from user profile
        parking_slots[spot_id].update({
            'status': 'Occupied',
            'client_name': user.name,
            'vehicle_type': vehicle_type,
            'plate_number': user.number_plate,
            'timestamp': timestamp,
            'booking_duration': booking_duration  # Store as integer
        })
        
        # Log confirmation of slot data update
        logger.info(f"Updated slot data: {parking_slots[spot_id]}")
        
        # First emit the UI updates immediately before doing slower operations
        socketio.emit('update_slots_client', {'slots': parking_slots})
        socketio.emit('slot_status_changed', {
            'slot_id': spot_id,
            'status': 'Occupied',
            'slots': parking_slots
        })
        socketio.emit('update_slots_and_frame', {
            'frame': current_frame,
            'slots': parking_slots
        })
        
        # Start background operations in a separate thread to avoid blocking
        def background_operations():
            try:
                # Update XML
                update_parking_slot_xml(
                    slot_id=spot_id,
                    status='Occupied',
                    vehicle_type=vehicle_type,
                    plate_number=user.number_plate,
                    timestamp=timestamp,
                    client_name=user.name
                )
                
                # Publish to MQTT
                mqtt_manager.publish_parking_status(parking_slots)
                
                # Send SMS notification - last step as it's often slowest
                send_sms(user.phone_number, f"Parking {spot_id} has been booked for {booking_duration//60} minutes.")
            except Exception as e:
                logger.error(f"Error in booking background operations: {e}")
                
        threading.Thread(target=background_operations).start()
        
        return {'success': True, 'message': f'{spot_id} booked successfully!'}
    else:
        socketio.emit('slot_booking_failed', {
            'slot_id': spot_id,
            'message': 'Slot is already occupied or does not exist'
        })
        return {'success': False, 'message': 'Slot is already occupied or does not exist'}

# First, let's fix the release_spot function to properly handle the slot ID
@socketio.on('release_spot')
def handle_release_spot(data):
    spot_id = data.get('spot_id')
    
    # Log the incoming release request for debugging
    logger.info(f"Release request received for spot: {spot_id}")
    
    if not spot_id:
        logger.error("No spot_id provided in release request")
        return {'success': False, 'message': 'No slot ID provided'}
    
    # Normalize spot_id format to ensure it always has 'slot' prefix
    if not spot_id.startswith('slot'):
        spot_id = f"slot{spot_id}"
    
    if 'user_id' not in session:
        logger.warning("User not logged in attempting to release spot")
        return {'success': False, 'message': 'Please login first'}
    
    user = User.query.get(session['user_id'])
    logger.info(f"User {user.name} attempting to release spot {spot_id}")
    
    if spot_id in parking_slots:
        # Check if the slot is actually booked by checking for a client name
        if not parking_slots[spot_id].get('client_name'):
            logger.warning(f"Slot {spot_id} release failed: not booked by any client")
            socketio.emit('slot_release_failed', {
                'slot_id': spot_id,
                'message': 'This slot is not booked by any client'
            })
            return {'success': False, 'message': 'This slot is not booked by any client'}
        
        # Check if the current user is the one who booked the slot
        if parking_slots[spot_id].get('client_name', '').lower() != user.name.lower():
            logger.warning(f"Slot {spot_id} release failed: not booked by requesting user")
            socketio.emit('slot_release_failed', {
                'slot_id': spot_id,
                'message': 'You can only release your own booked slot'
            })
            return {'success': False, 'message': 'You can only release your own booked slot'}
        
        # Now safely release the slot - IMPORTANT: Always set to Free regardless of plate detection
        # This ensures UI updates correctly
        parking_slots[spot_id].update({
            'status': 'Free',  # ALWAYS set to Free for UI update
            'client_name': '',
            'timestamp': '',
            'booking_duration': 3600,  # Reset to default
            'vehicle_type': '',  # Clear vehicle type
            'plate_number': ''   # Clear plate number - IMPORTANT for UI
        })
        
        # First emit UI updates immediately
        socketio.emit('slot_status_changed', {
            'slot_id': spot_id,
            'status': 'Free',  # Always Free to ensure UI updates
            'slots': parking_slots
        })
        socketio.emit('update_slots_client', {'slots': parking_slots})
        socketio.emit('update_slots_and_frame', {
            'frame': current_frame,
            'slots': parking_slots
        })
        
        # Start background operations in a separate thread
        def background_operations():
            try:
                # Update XML
                update_parking_slot_xml(
                    slot_id=spot_id,
                    status='Free',  # Always free in XML too
                    vehicle_type='',
                    plate_number='',
                    timestamp=datetime.now().isoformat(),
                    client_name=''
                )
                
                # Publish the updated status to MQTT
                mqtt_manager.publish_parking_status(parking_slots)
                
                # Send SMS notification
                send_sms(user.phone_number, f"Parking {spot_id} has been released.")
            except Exception as e:
                logger.error(f"Error in release background operations: {e}")
        
        threading.Thread(target=background_operations).start()
        
        logger.info(f"Slot {spot_id} released successfully by {user.name}")
        return {'success': True, 'message': f'{spot_id} released successfully!'}
    
    logger.error(f"Invalid slot ID in release request: {spot_id}")
    socketio.emit('slot_release_failed', {
        'slot_id': spot_id,
        'message': 'Invalid slot ID'
    })
    return {'success': False, 'message': 'Invalid slot ID'}


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    threading.Thread(target=auto_release_spots, daemon=True).start()
    try:
        socketio.run(app, debug=True)
    finally:
        mqtt_manager.cleanup()