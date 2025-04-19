import cv2
import numpy as np
import json

# === Setup Video ===
cap = cv2.VideoCapture(r'C:\Users\pinky\OneDrive\Documents\Desktop\Detection\Demo_Video.mp4')
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', original_width, original_height)

# === Globals ===
posList = []
dragging = False
start_point = (-1, -1)
current_rectangle = None

# Default modifiers
default_a, default_b, default_c = 10, 20, 15
step = 5  # adjustment step for keyboard

# === Mouse Callback ===
def Mouse(events, x, y, flags, params):
    global dragging, start_point, posList, current_rectangle

    if events == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)

    elif events == cv2.EVENT_MOUSEMOVE and dragging:
        # Live preview (ignored if not dragging)
        pass

    elif events == cv2.EVENT_LBUTTONUP:
        dragging = False
        end_x, end_y = x, y
        x1, y1 = start_point
        width = end_x - x1
        height = end_y - y1
        # Add new shape
        posList.append((x1, y1, height, width, default_a, default_b, default_c))
        current_rectangle = len(posList) - 1

    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x, y, h, w, a, b, c = pos
            pts = np.array([
                (x, y),
                (x + w + a, y),
                (x + w + a + b, y + h),
                (x + c, y + h)
            ], np.int32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                del posList[i]
                current_rectangle = None
                break

# === Draw Help Text ===
def draw_instructions(frame):
    instructions = [
        "Left Click + Drag: Create shape",
        "Right Click: Delete shape",
        "W/S: Height -/+",
        "A/D: Width -/+",
        "O/P: a -/+",
        "K/L: b -/+",
        "H/J: c -/+",
        "Q: Save & Quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

# === Main Loop ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_display = frame.copy()

        # Draw all shapes
        for i, pos in enumerate(posList):
            x, y, h, w, a, b, c = pos
            pts = np.array([
                (x, y),
                (x + w + a, y),
                (x + w + a + b, y + h),
                (x + c, y + h)
            ], np.int32).reshape((-1, 1, 2))
            color = (0, 255, 255) if i == current_rectangle else (0, 255, 0)
            cv2.polylines(frame_display, [pts], True, color, 3)
            cv2.putText(frame_display, str(i + 1), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        draw_instructions(frame_display)
        cv2.imshow("Frame", frame_display)
        cv2.setMouseCallback("Frame", Mouse)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

        # Keyboard adjustment
        if current_rectangle is not None and 0 <= current_rectangle < len(posList):
            x, y, h, w, a, b, c = posList[current_rectangle]
            if key == ord('w'):
                h -= step
            elif key == ord('s'):
                h += step
            elif key == ord('a'):
                w -= step
            elif key == ord('d'):
                w += step
            elif key == ord('o'):
                a += step
            elif key == ord('p'):
                a -= step
            elif key == ord('l'):
                b += step
            elif key == ord('k'):
                b -= step
            elif key == ord('j'):
                c += step
            elif key == ord('h'):
                c -= step
            posList[current_rectangle] = (x, y, h, w, a, b, c)

    # Save on quit
    coordinates = []
    for pos in posList:
        x, y, h, w, a, b, c = pos
        coords = [
            (x, y),
            (x + w + a, y),
            (x + w + a + b, y + h),
            (x + c, y + h)
        ]
        coordinates.append(coords)

    with open("rectangle_coordinates.json", "w") as f:
        json.dump(coordinates, f)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()


""" 
Guide to Draw coordinates

Action            | Description
Left Click + Drag | Draw a new coordinate/shape by dragging from one corner to another.
Right Click       | Delete the shape under the mouse cursor.

Key               | Function
W                 | 🔼 Decrease height (move bottom edge up)
S                 | 🔽 Increase height (move bottom edge down)
A                 | ◀️ Decrease width (move right edge left)
D                 | ▶️ Increase width (move right edge right)
O                 | ➕ Increase modifier a (shifts top-right X more to the right)
P                 | ➖ Decrease modifier a (brings top-right X closer to top-left)
K                 | ➖ Decrease modifier b (brings bottom-right X closer to top-right)
L                 | ➕ Increase modifier b (pushes bottom-right X farther to the right)
H                 | ➖ Decrease modifier c (brings bottom-left X closer to top-left)
J                 | ➕ Increase modifier c (pushes bottom-left X farther to the right)
Q                 | 💾 Save all coordinates to rectangle_coordinates.json and quit the program

 """