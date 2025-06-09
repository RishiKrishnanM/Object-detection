import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

    def detect_hands(self, frame_rgb):
        return self.hands.process(frame_rgb)

    def draw_hands(self, frame, hand_landmarks):
        self.drawer.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)