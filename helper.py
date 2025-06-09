import numpy as np

def calculate_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_hand_near_object(hand_landmarks, obj_center, frame_shape, threshold=50):
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame_shape[1])
        y = int(lm.y * frame_shape[0])
        if np.linalg.norm(np.array([x, y]) - np.array(obj_center)) < threshold:
            return True
    return False