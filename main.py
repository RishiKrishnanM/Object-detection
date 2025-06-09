import cv2
from object_detector import ObjectDetector
from hand_detector import HandDetector
from helper import calculate_center, is_hand_near_object

object_detector = ObjectDetector()
hand_detector = HandDetector()

cap = cv2.VideoCapture(0)

object_positions = {}
object_status = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hand_detector.detect_hands(frame_rgb)

    boxes, classes = object_detector.detect_objects(frame)
    current_ids = set()

    for i, box in enumerate(boxes):
        class_id = int(classes[i])
        class_name = object_detector.get_class_name(class_id)
        x1, y1, x2, y2 = map(int, box)
        obj_center = calculate_center((x1, y1, x2, y2))

        current_ids.add(i)
        prev_pos = object_positions.get(i)
        object_positions[i] = obj_center

        label = f"{class_name} Not Picked"
        picked = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_hand_near_object(hand_landmarks, obj_center, frame.shape):
                    if prev_pos:
                        dy = prev_pos[1] - obj_center[1]
                        if dy > 15:
                            object_status[i] = "Picked Up"
                            picked = True
                    else:
                        label = f"{class_name} In Contact"
                    hand_detector.draw_hands(frame, hand_landmarks)

        if object_status.get(i) == "Picked Up":
            label = f"{class_name} Picked Up"
        elif not picked:
            object_status[i] = "Not Picked"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for obj_id, status in object_status.items():
        if obj_id not in current_ids and status == "Picked Up":
            cv2.putText(frame, f"Object {obj_id}: Picked Up (Off-Screen)", (10, 30 + obj_id * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(boxes) == 0 and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_detector.draw_hands(frame, hand_landmarks)
            cv2.putText(frame, "Unrecognized Object Interaction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Object Pickup Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()