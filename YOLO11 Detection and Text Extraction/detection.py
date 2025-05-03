from ultralytics import YOLO
import cv2
import numpy as np
import socketio
import base64
import time
import easyocr
import string
import json
import re
from collections import defaultdict
import threading

class LicensePlateDetector:
    def __init__(self):
        print("Initializing OCR... (Using CPU - this would be faster with GPU)")
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.debug = False  # Reduced debugging for better performance
        # Add plate memory for the improved system
        self.plate_memory = {}
        self.memory_timeout = 10  # Seconds before forgetting a plate
        # Initialize OCR mode
        self.ocr_mode = "accurate"  # Default to accurate mode

    def set_ocr_mode(self, mode):
        """Set the OCR processing mode: 'accurate' or 'fast'"""
        if mode in ["accurate", "fast"]:
            self.ocr_mode = mode
            print(f"LicensePlateDetector OCR mode set to: {mode}")
        else:
            print(f"Invalid OCR mode: {mode}. Using 'accurate'")
            self.ocr_mode = "accurate"

    def enhance_plate_image(self, img):
        try:
            if img is None or img.size == 0:
                return None

            # 1. Resize the plate image larger
            height, width = img.shape[:2]
            scale_factor = 2
            img = cv2.resize(img, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)

            # 2. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

            # 4. Reduce noise with a faster denoising method for low-end hardware
            denoised = cv2.medianBlur(thresh, 3)  # Faster alternative to fastNlMeansDenoising

            # 5. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # 6. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            # 7. Edge enhancement
            kernel_sharpening = np.array([[-1,-1,-1], 
                                        [-1, 9,-1],
                                        [-1,-1,-1]])
            sharpened = cv2.filter2D(morph, -1, kernel_sharpening)

            if self.debug:
                debug_images = {
                    'original': img,
                    'grayscale': gray,
                    'threshold': thresh,
                    'denoised': denoised,
                    'enhanced': enhanced,
                    'morphology': morph,
                    'final': sharpened
                }
                self.save_debug_images(debug_images)

            return sharpened

        except Exception as e:
            print(f"Error enhancing plate image: {e}")
            return None

    def save_debug_images(self, images):
        for name, img in images.items():
            cv2.imwrite(f'debug_plate_{name}.jpg', img)

    def read_plate(self, plate_img, plate_id=None):
        """
        Read license plate text from the image. Uses memory if plate_id is provided.
        Returns just the text (for backward compatibility).
        """
        result = self.read_plate_with_confidence(plate_img, plate_id)
        return result['text']

    def read_plate_with_confidence(self, plate_img, plate_id=None):
        """
        Read license plate text from the image with confidence score.
        Uses memory if plate_id is provided.
        Returns a dict with 'text' and 'confidence'.
        """
        if plate_img is None or plate_img.size == 0:
            return {'text': "Unknown", 'confidence': 0.0}

        # Get current OCR mode
        ocr_mode = getattr(self, 'ocr_mode', "accurate")  # Default to accurate if not set

        height, width = plate_img.shape[:2]
        aspect_ratio = width / height
        if self.debug:
            print(f"Plate image size: {width}x{height}, Aspect ratio: {aspect_ratio:.2f}")

        if width < 60 or height < 20:
            if self.debug:
                print("Plate image too small, skipping OCR")
            return {'text': "Unknown", 'confidence': 0.0}

        if aspect_ratio < 1.5 or aspect_ratio > 5.0:
            if self.debug:
                print("Invalid plate aspect ratio, skipping OCR")
            return {'text': "Unknown", 'confidence': 0.0}
            
        # Check memory if plate_id provided
        if plate_id and plate_id in self.plate_memory:
            mem_entry = self.plate_memory[plate_id]
            if time.time() - mem_entry['timestamp'] < self.memory_timeout:
                if self.debug:
                    print(f"Using memorized plate: {mem_entry['text']} (confidence: {mem_entry['confidence']:.2f})")
                return {'text': mem_entry['text'], 'confidence': mem_entry['confidence']}

        # Apply different preprocessing based on OCR mode
        if ocr_mode == "fast":
            # Fast mode - minimal preprocessing
            enhanced_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            # Skip the more intensive enhancement steps
        else:  # "accurate" mode
            # Full preprocessing pipeline
            enhanced_img = self.enhance_plate_image(plate_img)
            
        if enhanced_img is None:
            return {'text': "Unknown", 'confidence': 0.0}

        try:
            # Multiple OCR attempts with different preprocessing
            # For fast mode, we'll use fewer attempts
            if ocr_mode == "fast":
                ocr_attempts = [
                    (enhanced_img, 0.4),  # Just try grayscale with lower threshold
                ]
            else:
                ocr_attempts = [
                    (enhanced_img, 0.4),  # Lower confidence threshold
                    (cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 0.4),
                    (cv2.threshold(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 
                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 0.4)
                ]

            best_result = {'text': "Unknown", 'confidence': 0.0}

            for img, threshold in ocr_attempts:
                # Adjust reader parameters based on OCR mode
                if ocr_mode == "fast":
                    # Faster OCR settings for speed
                    results = self.reader.readtext(
                        img,
                        allowlist=string.ascii_uppercase + string.digits + ' -',
                        batch_size=1,
                        detail=1,
                        min_size=20,
                        text_threshold=threshold,
                        paragraph=False,
                        width_ths=0.9,  # Less strict
                        height_ths=0.9   # Less strict
                    )
                else:
                    # More accurate OCR settings
                    results = self.reader.readtext(
                        img,
                        allowlist=string.ascii_uppercase + string.digits + ' -',
                        batch_size=1,
                        detail=1,
                        min_size=20,
                        text_threshold=threshold,
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7
                    )

                # Combine all text segments
                plate_text = ""
                total_confidence = 0
                num_segments = len(results)
                
                if num_segments > 0:
                    for (bbox, text, confidence) in results:
                        cleaned_text = text.strip().upper()
                        if cleaned_text:
                            plate_text += " " + cleaned_text
                            total_confidence += confidence
                    
                    plate_text = plate_text.strip()
                    avg_confidence = total_confidence / num_segments if num_segments > 0 else 0
                    
                    if self.validate_plate(plate_text):
                        if self.debug:
                            print(f"OCR attempt found: {plate_text} (confidence: {avg_confidence:.2f})")
                        
                        # If this is better than our previous best, update
                        if avg_confidence > best_result['confidence']:
                            best_result = {
                                'text': plate_text,
                                'confidence': avg_confidence
                            }

            # Store in memory if plate_id provided and we got a result
            if plate_id and best_result['text'] != "Unknown":
                self.plate_memory[plate_id] = {
                    'text': best_result['text'],
                    'confidence': best_result['confidence'],
                    'timestamp': time.time()
                }
                
            if best_result['text'] == "Unknown" and self.debug:
                print("No valid plate text found in any OCR attempt")
                
            return best_result

        except Exception as e:
            print(f"Error reading plate: {e}")
            return {'text': "Unknown", 'confidence': 0.0}

    def validate_plate(self, text):
        if not text:
            if self.debug:
                print("Empty plate text")
            return False
            
        # Remove spaces for length check
        text_no_spaces = text.replace(" ", "")
        if len(text_no_spaces) < 8 or len(text_no_spaces) > 11:  # Adjusted for Indian plates
            if self.debug:
                print(f"Invalid length ({len(text_no_spaces)}): {text}")
            return False
            
        # Check for state code format (e.g., "KA 08")
        parts = text.split()
        if len(parts) < 2:
            if self.debug:
                print("Missing state code format")
            return False
            
        # Verify first part is state code (2 letters)
        if not (len(parts[0]) == 2 and parts[0].isalpha()):
            if self.debug:
                print("Invalid state code")
            return False
            
        # Verify second part is district number
        if not parts[1].isdigit():
            if self.debug:
                print("Invalid district number")
            return False
            
        return True
        
    def update_memory(self):
        """Clean up expired entries from plate memory"""
        current_time = time.time()
        expired_keys = [k for k, v in self.plate_memory.items() 
                      if current_time - v['timestamp'] > self.memory_timeout]
        
        for key in expired_keys:
            del self.plate_memory[key]


class ParkingDetectionSystem:
    def __init__(self, model_path, coordinates_path):
        self.model = YOLO(model_path)
        self.plate_detector = LicensePlateDetector()
        self.parking_coordinates = self.load_coordinates(coordinates_path)
        self.confidence_threshold = 0.4  # Lower threshold for better detection
         # Add these to your existing __init__ method
        self.processing_active = True  # Flag for the processing thread
        self.frame_lock = threading.Lock()  # For thread safety
        self.latest_frame = None  # Store the latest processed frame
        self.frame_count = 0  # Keep track of frames in the class
        self.frame_skip = 10  # Process every 10th frame for low-end hardware
        self.process_all_frames = False  # Toggle for processing all frames
        self.display_fps = 0  # Store FPS for display
        self.last_full_update = time.time()  # For full updates
        self.full_update_interval = 30.0  # Send full update every 30 seconds
        
        # OCR mode toggle
        self.ocr_mode = "accurate"  # Options: "accurate" or "fast"
        # Server connection setup with reconnection logic
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0,  # Infinite reconnection attempts
                                  reconnection_delay=1, reconnection_delay_max=5)
        self.server_url = 'http://127.0.0.1:5000'
        self.server_connected = False
        self.server_frame_size = (640, 640)
        
        # Setup connection event handlers
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('connect_error', self.on_connect_error)
        
        # Processing metrics
        self.last_process_time = time.time()
        self.fps = 0
        
        # Track vehicles in slots
        self.slot_memory = defaultdict(lambda: {
            'status': 'Free',
            'vehicle_type': None,
            'plate_number': 'Unknown',
            'confidence': 0,
            'stability': 0,
            'last_seen': 0,
            'last_sent': 0,  # Timestamp of last update sent to server
            'plate_confidence': 0  # Track confidence in plate reading
        })
        
        # For tracking changes that require server updates
        self.last_sent_states = {}
        self.min_update_interval = 1.0  # Minimum seconds between updates for the same slot
        self.significant_confidence_change = 0.15  # Confidence change threshold for plate updates
        
        # Class indices
        self.CLASS_BIKE = 0
        self.CLASS_CAR = 1
        self.CLASS_NUMBER_PLATE = 2
        
        # Data queue for server communication to prevent blocking
        self.data_queue = []
        self.queue_lock = threading.Lock()
        
        # Start server communication thread
        self.server_thread_running = True
        self.server_thread = threading.Thread(target=self.server_communication_thread)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Connect to server on initialization
        self.connect_to_server()

    def on_connect(self):
        print("Connected to server")
        self.server_connected = True
        
    def on_disconnect(self):
        print("Disconnected from server")
        self.server_connected = False
        
    def on_connect_error(self, error):
        print(f"Connection error: {error}")
        self.server_connected = False

    def connect_to_server(self):
        try:
            if not self.server_connected:
                print("Connecting to server...")
                self.sio.connect(self.server_url)
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.server_connected = False

    def load_coordinates(self, path):
        try:
            with open(path, 'r') as file:
                coordinates = json.load(file)
            return {f'slot{i+1}': np.array(coord, np.int32) 
                    for i, coord in enumerate(coordinates)}
        except Exception as e:
            print(f"Error loading coordinates: {e}")
            return {}

    def set_ocr_mode(self, mode):
        """Set the OCR processing mode: 'accurate' or 'fast'"""
        if mode in ["accurate", "fast"]:
            self.ocr_mode = mode
            print(f"LicensePlateDetector OCR mode set to: {mode}")
        else:
            print(f"Invalid OCR mode: {mode}. Using 'accurate'")
            self.ocr_mode = "accurate"
    
    def process_frame(self, frame):
        start_time = time.time()
        
        try:
            # Create a copy for drawing
            output_frame = frame.copy()
            
            # Track which slots have changes that need reporting
            changed_slots = set()
            
            # Expire old slot memory entries
            current_time = time.time()
            for slot, data in self.slot_memory.items():
                if data['status'] == 'Occupied' and current_time - data['last_seen'] > 2.0:
                    data['stability'] -= 1
                    if data['stability'] <= 0:
                        # Status changing from Occupied to Free - mark for update
                        if slot in self.last_sent_states and self.last_sent_states[slot]['status'] == 'Occupied':
                            changed_slots.add(slot)
                        
                        data['status'] = 'Free'
                        data['vehicle_type'] = None
                        data['plate_number'] = 'Unknown'
                        data['confidence'] = 0
                        data['stability'] = 0
                        data['plate_confidence'] = 0

            # Get model predictions
            results = self.model(frame)[0]
            
            # Extract detections
            if results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()
                
                # Get all detections
                all_detections = []
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    if conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box[:4])
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        detection = {
                            'box': (x1, y1, x2, y2),
                            'center': center,
                            'conf': conf,
                            'class_id': int(cls_id),
                            'type': 'Bike' if cls_id == self.CLASS_BIKE else 
                                   'Car' if cls_id == self.CLASS_CAR else 'Plate'
                        }
                        all_detections.append(detection)
                
                # Group detections by slot
                slot_detections = defaultdict(list)
                for detection in all_detections:
                    for slot, coords in self.parking_coordinates.items():
                        if cv2.pointPolygonTest(coords, detection['center'], False) >= 0:
                            slot_detections[slot].append(detection)
                            break
                
                # Process each slot
                for slot, detections in slot_detections.items():
                    # Separate plates and vehicles
                    plates = [d for d in detections if d['class_id'] == self.CLASS_NUMBER_PLATE]
                    vehicles = [d for d in detections if d['class_id'] != self.CLASS_NUMBER_PLATE]
                    
                    # Process if we have vehicles
                    if vehicles:
                        # Sort by confidence
                        vehicles.sort(key=lambda x: x['conf'], reverse=True)
                        vehicle = vehicles[0]  # Use highest confidence vehicle
                        
                        # Check if status changed from Free to Occupied
                        was_free = self.slot_memory[slot]['status'] == 'Free'
                        
                        # Check if vehicle type changed
                        vehicle_type_changed = (self.slot_memory[slot]['vehicle_type'] != vehicle['type'] and 
                                               self.slot_memory[slot]['vehicle_type'] is not None)
                        
                        # Update slot memory with vehicle
                        self.slot_memory[slot]['status'] = 'Occupied'
                        self.slot_memory[slot]['vehicle_type'] = vehicle['type']
                        self.slot_memory[slot]['confidence'] = vehicle['conf']
                        self.slot_memory[slot]['last_seen'] = current_time
                        self.slot_memory[slot]['stability'] = min(10, self.slot_memory[slot]['stability'] + 1)
                        
                        # Mark for update if significant changes happened
                        if was_free or vehicle_type_changed:
                            changed_slots.add(slot)
                        
                        # Draw vehicle box
                        x1, y1, x2, y2 = vehicle['box']
                        color = (255, 0, 0) if vehicle['type'] == 'Bike' else (0, 255, 255)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(output_frame, f"{vehicle['type']} {vehicle['conf']:.2f}", 
                                  (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                        
                        # Process plate if we have any
                        if plates:
                            # Sort by confidence
                            plates.sort(key=lambda x: x['conf'], reverse=True)
                            plate = plates[0]  # Use highest confidence plate
                            
                            # Extract plate image
                            px1, py1, px2, py2 = plate['box']
                            plate_img = frame[py1:py2, px1:px2]
                            
                            # Read plate text with memory
                            plate_id = f"{slot}_{vehicle['type']}"
                            old_plate = self.slot_memory[slot]['plate_number']
                            
                            # Get plate reading and confidence
                            plate_result = self.plate_detector.read_plate_with_confidence(plate_img, plate_id)
                            plate_text = plate_result['text']
                            plate_confidence = plate_result['confidence']
                            
                            if plate_text != "Unknown" and plate_confidence > self.slot_memory[slot]['plate_confidence']:
                                # Check if plate changed significantly or is better quality
                                plate_changed = (old_plate != plate_text and old_plate != "Unknown")
                                confidence_improved = (plate_confidence - self.slot_memory[slot]['plate_confidence'] > 
                                                     self.significant_confidence_change)
                                
                                if plate_changed or confidence_improved:
                                    changed_slots.add(slot)
                                
                                # Update with new plate info
                                self.slot_memory[slot]['plate_number'] = plate_text
                                self.slot_memory[slot]['plate_confidence'] = plate_confidence
                            
                            # Draw plate box
                            cv2.rectangle(output_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                            cv2.putText(output_frame, f"{self.slot_memory[slot]['plate_number']} ({plate_confidence:.2f})", 
                                      (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Draw parking slots
            for slot, coords in self.parking_coordinates.items():
                slot_data = self.slot_memory[slot]
                
                # Determine color based on status and vehicle type
                if slot_data['status'] == 'Occupied':
                    # Colors based on vehicle type and stability
                    if slot_data['stability'] > 5:  # Stable detection
                        if slot_data['vehicle_type'] == 'Bike':
                            color = (255, 0, 0)  # Blue
                        else:  # Car
                            color = (0, 255, 255)  # Yellow
                    else:  # Less stable detection
                        color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green for free
                
                # Draw the slot
                cv2.polylines(output_frame, [coords], True, color, 2)
                
                # Add slot number and info
                M = cv2.moments(coords)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(output_frame, slot, (cX-20, cY), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add vehicle type if occupied
                    if slot_data['status'] == 'Occupied':
                        text = f"{slot_data['vehicle_type']}"
                        cv2.putText(output_frame, text, (cX-20, cY+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update plate detector memory
            self.plate_detector.update_memory()
            
            # Calculate processing time and FPS
            process_time = time.time() - start_time
            self.fps = 1.0 / process_time
            
            # Queue updates for changed slots (non-blocking)
            if changed_slots:
                self.queue_slot_updates(output_frame, changed_slots)
            
            return output_frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def get_slot_states(self, slots=None):
        """Convert slot memory to server format, optionally filtering for specific slots"""
        slot_states = {}
        for slot, data in self.slot_memory.items():
            if slots is None or slot in slots:
                slot_states[slot] = {
                    'status': data['status'],
                    'vehicle_type': data['vehicle_type'],
                    'plate_number': data['plate_number'],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Update last sent state
                self.last_sent_states[slot] = slot_states[slot].copy()
                self.slot_memory[slot]['last_sent'] = time.time()
                
        return slot_states

    def queue_slot_updates(self, frame, changed_slots):
        """Queue slot updates for sending via separate thread"""
        try:
            # Filter for slots that haven't been updated too recently
            current_time = time.time()
            slots_to_update = set()
            
            for slot in changed_slots:
                # If it's been long enough since last update, or this is a first update
                if (slot not in self.last_sent_states or 
                    current_time - self.slot_memory[slot]['last_sent'] >= self.min_update_interval):
                    slots_to_update.add(slot)
            
            if not slots_to_update:
                return  # No slots need updating
            
            # Get filtered slot states
            slot_states = self.get_slot_states(slots_to_update)
            
            # Resize for server
            server_frame = cv2.resize(frame, self.server_frame_size)
            _, buffer = cv2.imencode('.jpg', server_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduce quality for performance
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Queue data for sending
            with self.queue_lock:
                self.data_queue.append({
                    'frame': encoded_frame,
                    'slots': slot_states,
                    'update_type': 'partial'
                })
            
        except Exception as e:
            print(f"Error queuing data: {e}")

    def queue_full_update(self, frame):
        """Queue a full update of all slots"""
        try:
            # Get all current slot states
            slot_states = self.get_slot_states()
            
            # Resize for server
            server_frame = cv2.resize(frame, self.server_frame_size)
            _, buffer = cv2.imencode('.jpg', server_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduce quality for performance
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Queue data for sending
            with self.queue_lock:
                self.data_queue.append({
                    'frame': encoded_frame,
                    'slots': slot_states,
                    'update_type': 'full'
                })
            
        except Exception as e:
            print(f"Error queuing full update: {e}")

    # Add this new method
    def cleanup_memory(self, max_age=300):  # 5 minutes
        """Remove old entries from plate detector memory"""
        try:
            current_time = time.time()
            # Clean up old slot memory entries that haven't been seen recently
            slots_to_reset = []
            for slot, data in self.slot_memory.items():
                if data['status'] == 'Free' and current_time - data['last_seen'] > max_age:
                    slots_to_reset.append(slot)
        
            for slot in slots_to_reset:
                # Reset completely instead of just removing
                self.slot_memory[slot] = {
                    'status': 'Free',
                    'vehicle_type': None,
                    'plate_number': 'Unknown',
                    'confidence': 0,
                    'stability': 0,
                    'last_seen': 0,
                    'last_sent': 0,
                    'plate_confidence': 0
                }
        
            # Clean up the plate detector's memory
            # Fixed: Use plate_memory instead of memory
            if hasattr(self.plate_detector, 'plate_memory'):
                keys_to_remove = []
                for plate_id, data in self.plate_detector.plate_memory.items():
                    if current_time - data['timestamp'] > max_age:
                        keys_to_remove.append(plate_id)
                    
                for key in keys_to_remove:
                    del self.plate_detector.plate_memory[key]
            
                if keys_to_remove:
                    print(f"Cleaned up {len(keys_to_remove)} old plate entries")
                
            return len(slots_to_reset)
        except Exception as e:
            print(f"Error cleaning up memory: {e}")
            return 0


    def server_communication_thread(self):
        """Thread for handling server communications without blocking the main processing"""
        while self.server_thread_running:
            # Check server connection
            if not self.server_connected:
                try:
                    self.connect_to_server()
                except Exception as e:
                    pass  # Connection attempt handled by socketio client
            
            # Process any queued data
            if self.server_connected:
                try:
                    data = None
                    with self.queue_lock:
                        if self.data_queue:
                            data = self.data_queue.pop(0)
                    
                    if data:
                        # Print occupied slots for debugging (only for partial updates)
                        if data['update_type'] == 'partial':
                            occupied_slots = []
                            for slot, state in data['slots'].items():
                                if state['status'] == 'Occupied':
                                    occupied_slots.append(f"{slot}: {state['vehicle_type']}, Plate: {state['plate_number']}")
                            
                            if occupied_slots:
                                print(f"UPDATE: {', '.join(occupied_slots)}")
                        
                        # Send to server
                        self.sio.emit('frame_and_slots', data)
                        
                        if data['update_type'] == 'full':
                            print("Sent full update for all slots")
                        else:
                            print(f"Sent updates for {len(data['slots'])} slots")
                
                except Exception as e:
                    print(f"Error in server communication: {e}")
            
            # Sleep to prevent CPU overuse
            time.sleep(0.05)

def processing_thread_function(system, video_source):
    """Dedicated thread for video processing independent of UI"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error opening video source in processing thread")
        return
        
    while system.processing_active:
        try:
            ret, frame = cap.read()
            if not ret:
                print("End of video in processing thread")
                # Send final state before breaking
                system.queue_full_update(system.latest_frame if system.latest_frame is not None 
                                      else np.zeros((480, 640, 3), dtype=np.uint8))
                time.sleep(1)  # Give time for the update to be sent
                break
                
            # Process based on configured rate
            if system.process_all_frames or system.frame_count % system.frame_skip == 0:
                processed_frame = system.process_frame(frame)
                
                # Check if it's time for a periodic full update
                current_time = time.time()
                if current_time - system.last_full_update >= system.full_update_interval:
                    system.queue_full_update(processed_frame)
                    system.last_full_update = current_time
                    
                    # Also run memory cleanup here
                    system.cleanup_memory()
                
                # Store the latest frame and metrics
                with system.frame_lock:
                    system.latest_frame = processed_frame
                    system.display_fps = system.fps  # Copy processing FPS for display
            else:
                # Just store the raw frame without processing
                with system.frame_lock:
                    system.latest_frame = frame
            
            # Update counters
            system.frame_count += 1
            
        except Exception as e:
            print(f"Error in processing thread: {e}")
            # Try to recover by reopening the video source
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(video_source)
            
        # Small sleep to prevent CPU overload
        time.sleep(0.001)
    
    cap.release()
    print("Processing thread ended")

def main():
    try:
        system = ParkingDetectionSystem(
            model_path=r'C:\Users\YOLO11 Detection and Text Extraction\yolo11n_model.pt',
            coordinates_path=r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\coordinates_Model_Video.txt'
            #coordinates_path=r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\live.txt'
        )

        # Video source - can be camera index or path
        video_source = r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\Demo_Video.mp4'
        #video_source = 0  # For webcam
        
        # Start the processing thread
        proc_thread = threading.Thread(target=processing_thread_function, 
                                    args=(system, video_source))
        proc_thread.daemon = True
        proc_thread.start()
        
        # Create window
        cv2.namedWindow('Parking Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parking Detection', 1280, 720)  # Adjust size as needed
        
        # For FPS calculation of display
        fps_start_time = time.time()
        fps_frame_count = 0
        display_fps = 0

        # Main loop - now only handles display and UI
        while True:
            # Get the latest processed frame
            frame_to_display = None
            with system.frame_lock:
                if system.latest_frame is not None:
                    frame_to_display = system.latest_frame.copy()
                    proc_fps = system.display_fps
            
            if frame_to_display is None:
                # No frame available yet
                time.sleep(0.01)
                continue
                
            # Update display FPS counter
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                display_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Add FPS counter
            cv2.putText(frame_to_display, f"Display: {display_fps:.1f} | Proc: {proc_fps:.1f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add OCR mode indicator
            cv2.putText(frame_to_display, f"OCR: {system.ocr_mode}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Parking Detection', frame_to_display)
            
            # Process keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                system.processing_active = False
                break
            elif key == ord('t'):  # Toggle frame processing mode
                system.process_all_frames = not system.process_all_frames
                if system.process_all_frames:
                    print("HIGH RATE MODE: Processing all frames")
                else:
                    print(f"LOW RATE MODE: Processing every {system.frame_skip}th frame")
            elif key == ord('c'):  # Decrease confidence threshold
                system.confidence_threshold = max(0.1, system.confidence_threshold - 0.05)
                print(f"Confidence threshold: {system.confidence_threshold:.2f}")
            elif key == ord('v'):  # Increase confidence threshold
                system.confidence_threshold = min(0.9, system.confidence_threshold + 0.05)
                print(f"Confidence threshold: {system.confidence_threshold:.2f}")
            elif key == ord('u'):  # Force full update
                with system.frame_lock:
                    if system.latest_frame is not None:
                        system.queue_full_update(system.latest_frame)
                system.last_full_update = time.time()
                print("Forced full update sent")
            elif key == ord('o'):  # Toggle OCR processing mode
                system.ocr_mode = "fast" if system.ocr_mode == "accurate" else "accurate"
                print(f"OCR Mode: {system.ocr_mode}")
                
                # Update the plate detector's OCR mode
                if hasattr(system.plate_detector, 'set_ocr_mode'):
                    system.plate_detector.set_ocr_mode(system.ocr_mode)
                else:
                    # If no setter method, directly set the attribute
                    system.plate_detector.ocr_mode = system.ocr_mode
            
            # Small sleep to prevent CPU overuse in the UI thread
            time.sleep(0.01)

        # Clean shutdown
        print("Waiting for processing thread to complete...")
        proc_thread.join(timeout=3.0)  # Wait up to 3 seconds for thread to end
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure proper cleanup
        cv2.destroyAllWindows()
        
        if 'system' in locals() and system is not None:
            system.processing_active = False  # Signal thread to stop
            print("Shutting down server communication...")
            system.server_thread_running = False
            time.sleep(0.5)  # Give threads time to exit
            
            try:
                if system.sio.connected:
                    system.sio.disconnect()
            except:
                pass

if __name__ == "__main__":
    main()
                    