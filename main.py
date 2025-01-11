import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller
from time import time

class VirtualKeyboard:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        self.keyboard_layout = [
            list('QWERTYUIOP'),
            list('ASDFGHJKL;'),
            list('ZXCVBNM,./'),
            [' ']
        ]

        self.key_width = 50
        self.key_height = 50
        self.space_bar_width = self.key_width * 5
        self.keyboard_x = 100
        self.keyboard_y = 100

        self.last_press_time = 0
        self.press_cooldown = 0.3
        self.press_threshold = 0.05

        self.keyboard = Controller()

    def create_keyboard_overlay(self, frame):
        for row_idx, row in enumerate(self.keyboard_layout):
            for col_idx, key in enumerate(row):
                x = self.keyboard_x + col_idx * self.key_width
                y = self.keyboard_y + row_idx * self.key_height
                width = self.space_bar_width if key == ' ' else self.key_width
                cv2.rectangle(frame, (x, y), (x + width, y + self.key_height), (255, 255, 255), 1)
                cv2.putText(frame, key.strip(), (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def detect_key_press(self, index_tip, thumb_tip):
        distance = np.sqrt(
            (index_tip.x - thumb_tip.x) ** 2 +
            (index_tip.y - thumb_tip.y) ** 2
        )
        current_time = time()
        if distance < self.press_threshold and current_time - self.last_press_time > self.press_cooldown:
            x_pos = index_tip.x * 640
            y_pos = index_tip.y * 480
            row = int((y_pos - self.keyboard_y) // self.key_height)

            if row == 3:  # Space bar row
                x_start = self.keyboard_x
                x_end = self.keyboard_x + self.space_bar_width
                if x_start <= x_pos <= x_end:
                    self.keyboard.press(' ')
                    self.keyboard.release(' ')
                    self.last_press_time = current_time
                    return "Space"
            else:
                col = int((x_pos - self.keyboard_x) // self.key_width)
                if 0 <= row < len(self.keyboard_layout) and 0 <= col < len(self.keyboard_layout[row]):
                    key = self.keyboard_layout[row][col]
                    self.keyboard.press(key.lower())
                    self.keyboard.release(key.lower())
                    self.last_press_time = current_time
                    return key
        return None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            frame = self.create_keyboard_overlay(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    pressed_key = self.detect_key_press(index_tip, thumb_tip)
                    if pressed_key:
                        print(f"Pressed: {pressed_key}")
            cv2.putText(frame, "Virtual Keyboard Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Virtual Keyboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    keyboard = VirtualKeyboard()
    keyboard.run()
