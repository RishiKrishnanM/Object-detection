import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

object_positions = {}
object_status = {}  

def calculate_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_hand_near_object(hand_landmarks, obj_center, threshold=50):
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        if np.linalg.norm(np.array([x, y]) - np.array(obj_center)) < threshold:
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detections = model.predict(frame, conf=0.25, verbose=False)[0]
    boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
    classes = detections.boxes.cls.cpu().numpy() if detections.boxes else []

    current_ids = set()
    
    for i, box in enumerate(boxes):
        class_id = int(classes[i])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box)
        obj_center = calculate_center((x1, y1, x2, y2))

        current_ids.add(i)

        prev_pos = object_positions.get(i)
        object_positions[i] = obj_center

        label = f"{class_name} Not Picked"
        picked = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_hand_near_object(hand_landmarks, obj_center):
                    if prev_pos:
                        dy = prev_pos[1] - obj_center[1]
                        if dy > 15:  
                            object_status[i] = "Picked Up"
                            picked = True
                    else:
                        label = f"{class_name} In Contact"
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if object_status.get(i) == "Picked Up":
            label = f"{class_name} Picked Up"
        elif not picked:
            object_status[i] = "Not Picked"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for obj_id, status in object_status.items():
        if obj_id not in current_ids and status == "Picked Up":
            cv2.putText(frame, f"Object {obj_id}: Picked Up (Off-Screen)", (10, 30 + obj_id * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(boxes) == 0 and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Unrecognized Object Interaction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Object Pickup Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
