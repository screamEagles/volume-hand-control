import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img


    def findPosition(self, img, hand_number=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, landmark in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                channel_x, channel_y = int(landmark.x * width), int(landmark.y * height)
                # print(id, channel_x, channel_y)
                landmark_list.append([id, channel_x, channel_y])
                # if id == 4:
                if draw:
                    cv2.circle(img, (channel_x, channel_y), 15, (255, 0, 255), cv2.FILLED)
        
        return landmark_list

    
def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img, draw=False)
        # if len(landmark_list) != 0:
        #     print(landmark_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        img = cv2.flip(img, 1)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
