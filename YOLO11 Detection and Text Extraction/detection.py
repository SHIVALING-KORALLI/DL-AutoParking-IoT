#combined_detection.py

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

class LicensePlateDetector:
    def __init__(self):
        print("Initializing OCR... (Using CPU - this would be faster with GPU)")
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.debug = True
        # Add plate memory for the improved system
        self.plate_memory = {}
        self.memory_timeout = 10  # Seconds before forgetting a plate

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

            # 4. Reduce noise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

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

        height, width = plate_img.shape[:2]
        aspect_ratio = width / height
        print(f"Plate image size: {width}x{height}, Aspect ratio: {aspect_ratio:.2f}")

        if width < 60 or height < 20:
            print("Plate image too small, skipping OCR")
            return {'text': "Unknown", 'confidence': 0.0}

        if aspect_ratio < 1.5 or aspect_ratio > 5.0:
            print("Invalid plate aspect ratio, skipping OCR")
            return {'text': "Unknown", 'confidence': 0.0}
            
        # Check memory if plate_id provided
        if plate_id and plate_id in self.plate_memory:
            mem_entry = self.plate_memory[plate_id]
            if time.time() - mem_entry['timestamp'] < self.memory_timeout:
                print(f"Using memorized plate: {mem_entry['text']} (confidence: {mem_entry['confidence']:.2f})")
                return {'text': mem_entry['text'], 'confidence': mem_entry['confidence']}

        enhanced_img = self.enhance_plate_image(plate_img)
        if enhanced_img is None:
            return {'text': "Unknown", 'confidence': 0.0}

        try:
            # Multiple OCR attempts with different preprocessing
            ocr_attempts = [
                (enhanced_img, 0.4),  # Lower confidence threshold
                (cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 0.4),
                (cv2.threshold(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 
                             0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 0.4)
            ]

            best_result = {'text': "Unknown", 'confidence': 0.0}

            for img, threshold in ocr_attempts:
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
                
            if best_result['text'] == "Unknown":
                print("No valid plate text found in any OCR attempt")
                
            return best_result

        except Exception as e:
            print(f"Error reading plate: {e}")
            return {'text': "Unknown", 'confidence': 0.0}

    def validate_plate(self, text):
        if not text:
            print("Empty plate text")
            return False
            
        # Remove spaces for length check
        text_no_spaces = text.replace(" ", "")
        if len(text_no_spaces) < 8 or len(text_no_spaces) > 11:  # Adjusted for Indian plates
            print(f"Invalid length ({len(text_no_spaces)}): {text}")
            return False
            
        # Check for state code format (e.g., "KA 08")
        parts = text.split()
        if len(parts) < 2:
            print("Missing state code format")
            return False
            
        # Verify first part is state code (2 letters)
        if not (len(parts[0]) == 2 and parts[0].isalpha()):
            print("Invalid state code")
            return False
            
        # Verify second part is district number
        if not parts[1].isdigit():
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
        self.sio = socketio.Client()
        self.server_frame_size = (640, 640)
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
        
        # Connect to server
        self.connect_to_server()

    def connect_to_server(self):
        try:
            self.sio.connect('http://127.0.0.1:5000')
            print("Connected to server")
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def load_coordinates(self, path):
        try:
            with open(path, 'r') as file:
                coordinates = json.load(file)
            return {f'slot{i+1}': np.array(coord, np.int32) 
                    for i, coord in enumerate(coordinates)}
        except Exception as e:
            print(f"Error loading coordinates: {e}")
            return {}

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
            
            # Send updates for changed slots
            if changed_slots:
                self.send_changed_slots(output_frame, changed_slots)
            
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

    def send_changed_slots(self, frame, changed_slots):
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
            _, buffer = cv2.imencode('.jpg', server_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Print occupied slots for debugging
            for slot, state in slot_states.items():
                if state['status'] == 'Occupied':
                    print(f"UPDATE: {slot}: {state['vehicle_type']}, Plate: {state['plate_number']}")
            
            # Send to server - same format as original code but only for changed slots
            self.sio.emit('frame_and_slots', {
                'frame': encoded_frame,
                'slots': slot_states,
                'update_type': 'partial'  # Indicate this is a partial update
            })
            
            print(f"Sent updates for {len(slots_to_update)} slots: {', '.join(slots_to_update)}")
            
        except Exception as e:
            print(f"Error sending data: {e}")

    def send_full_update(self, frame):
        """Send a full update of all slots (for periodic updates)"""
        try:
            # Get all current slot states
            slot_states = self.get_slot_states()
            
            # Resize for server
            server_frame = cv2.resize(frame, self.server_frame_size)
            _, buffer = cv2.imencode('.jpg', server_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Send to server with full update type
            self.sio.emit('frame_and_slots', {
                'frame': encoded_frame,
                'slots': slot_states,
                'update_type': 'full'  # Indicate this is a full update
            })
            
            print("Sent full update for all slots")
            
        except Exception as e:
            print(f"Error sending full update: {e}")

def main():
    try:
        system = ParkingDetectionSystem(
            model_path=r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\runs\detect\train\weights\best.pt',
            coordinates_path=r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\coordinates_Model_Video.txt'
        )

        # Video source - can be camera index or path
        video_source = r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\Demo_Video.mp4'
        # video_source = 0  # For webcam
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error opening video source: {video_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video source opened: {width}x{height} at {fps} FPS")
        
        # Create window
        cv2.namedWindow('Parking Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parking Detection', width, height)
        #cv2.resizeWindow('Parking Detection', 1280, 720)
        # Processing settings
        frame_skip = 10  # Process every 10th frame
        frame_count = 0
        process_all_frames = False  # Toggle for processing all frames
        
        # For FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        display_fps = 0
        
        # For periodic full updates
        last_full_update = time.time()
        full_update_interval = 30.0  # Send full update every 30 seconds

        # Main loop
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                # Send final state before breaking
                system.send_full_update(frame)
                break
            
            # Update counters
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate display FPS (not processing FPS)
            if time.time() - fps_start_time >= 1.0:
                display_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Process frame based on frame skip or all frames mode
            if process_all_frames or frame_count % frame_skip == 0:
                processed_frame = system.process_frame(frame)
                
                # Check if it's time for a periodic full update
                current_time = time.time()
                if current_time - last_full_update >= full_update_interval:
                    system.send_full_update(processed_frame)
                    last_full_update = current_time
                
                # Show processing FPS too
                proc_fps = system.fps
            else:
                # Just display without processing
                processed_frame = frame
                proc_fps = 0
            
            # Add FPS counter
            cv2.putText(processed_frame, f"FPS: {display_fps:.1f} | Proc: {proc_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Parking Detection', processed_frame)
            
            # Process keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):  # Toggle frame processing mode
                process_all_frames = not process_all_frames
                print(f"Processing {'all frames' if process_all_frames else f'every {frame_skip}th frame'}")
            elif key == ord('c'):  # Decrease confidence threshold
                system.confidence_threshold = max(0.1, system.confidence_threshold - 0.05)
                print(f"Confidence threshold: {system.confidence_threshold:.2f}")
            elif key == ord('v'):  # Increase confidence threshold
                system.confidence_threshold = min(0.9, system.confidence_threshold + 0.05)
                print(f"Confidence threshold: {system.confidence_threshold:.2f}")
            elif key == ord('u'):  # Force full update
                system.send_full_update(processed_frame)
                last_full_update = time.time()
                print("Forced full update sent")

        # Final cleanup
        print("Video processing completed.")
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if 'system' in locals() and system is not None:
            print("Disconnecting from server...")
            system.sio.disconnect()

if __name__ == "__main__":
    main()