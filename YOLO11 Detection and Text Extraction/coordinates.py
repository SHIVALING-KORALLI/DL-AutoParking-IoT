import cv2
import numpy as np
import json

cap = cv2.VideoCapture(r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\Demo_Video.mp4')

# Get original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create window with original dimensions
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', original_width, original_height)

posList = []
current_rectangle = None  # Index of the currently selected rectangle

defaultHeight = 50
defaultWidth = 80
a = 1
b = 1
c = 1

def Mouse(events, x, y, flags, params):
    global current_rectangle, a, b, c
    if events == cv2.EVENT_LBUTTONDOWN:
        # Append the position and the current height, width, a, b, and c
        posList.append((x, y, defaultHeight, defaultWidth, a, b, c))
        current_rectangle = len(posList) - 1  # The last rectangle is the current rectangle
    elif events == cv2.EVENT_RBUTTONDOWN:
        # Check if the right-clicked point is inside any of the polygons
        for i, pos in enumerate(posList):
            x1, y1, height, width, a, b, c = pos
            x2 = x1 + width + a
            x3 = x1 + b
            x4 = x1 + c
            y2 = y1 + height
            pts = np.array([(x1, y1), (x2, y1), (x3, y2), (x4, y2)], np.int32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                # If the point is inside the polygon, remove the polygon and break the loop
                del posList[i]
                current_rectangle = None  # Reset current_rectangle
                break

def draw_instructions(frame):
    # Add instructions text to the frame
    instructions = [
        "Controls:",
        "Left Click: Add new slot",
        "Right Click: Remove slot",
        "W/S: Adjust height",
        "A/D: Adjust width",
        "O/P: Adjust left edge",
        "K/L: Adjust right edge",
        "H/J: Adjust bottom edge",
        "Q: Save and quit"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, y_offset + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw all polygons
        for pos in posList:
            x, y, height, width, a, b, c = pos
            x1 = x + width + a
            x2 = x1 + b
            x3 = x + c
            y2 = y + height
            
            pts = np.array([(x, y), (x1, y), (x2, y2), (x3, y2)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Highlight the currently selected rectangle
            if current_rectangle is not None and pos == posList[current_rectangle]:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            display_frame = cv2.polylines(display_frame, [pts], True, color, 4)
            
            # Add slot number
            slot_num = posList.index(pos) + 1
            cv2.putText(display_frame, str(slot_num), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw instructions
        draw_instructions(display_frame)

        cv2.imshow('Frame', display_frame)
        cv2.setMouseCallback('Frame', Mouse)
        KEY = cv2.waitKey(25) & 0xff

        if KEY == ord('q'):
            break

        elif current_rectangle is not None:  # Only change a, b, c if a rectangle is selected
            x, y, height, width, _, _, _ = posList[current_rectangle]
            if KEY == ord('o'):  # increase a
                a += 1
            elif KEY == ord('p'):  # decrease a
                a -= 1
            elif KEY == ord('l'):  # increase b
                b += 1
            elif KEY == ord('k'):  # decrease b
                b -= 1
            elif KEY == ord('j'):  # increase c
                c += 1
            elif KEY == ord('h'):  # decrease c
                c -= 1
            elif KEY == ord('d'):  # increase width
                width += 1
            elif KEY == ord('a'):  # decrease width
                width -= 1
            elif KEY == ord('s'):  # increase height
                height += 1
            elif KEY == ord('w'):  # decrease height
                height -= 1
            posList[current_rectangle] = (x, y, height, width, a, b, c)

    # Save coordinates
    coordinates = []
    for pos in posList:
        x, y, height, width, a, b, c = pos
        x1 = x + width + a
        x2 = x1 + b
        x3 = x + c
        y2 = y + height
        coordinates.append([(x, y), (x1, y), (x2, y2), (x3, y2)])

    with open('rectangle_coordinates.json', 'w') as f:
        json.dump(coordinates, f)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
