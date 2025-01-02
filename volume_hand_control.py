import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


frame_width, frame_height = 640, 480
colour = (102, 145, 61)

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

current_time, previous_time = 0, 0

detector = htm.handDetector(detection_confidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
minimum_volume = volume_range[0]
maximum_volume = volume_range[1]
volume_interpreter = 0
volume_bar = 400
volume_percentage = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)

    if len(landmark_list) != 0:
        # print(landmark_list)

        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        centre_x, centre_y = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, colour, cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, colour, cv2.FILLED)
        cv2.circle(img, (centre_x, centre_y), 8, colour, cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), colour, 3)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        volume_interpreter = np.interp(length, [50, 300], [minimum_volume, maximum_volume])
        volume_bar = np.interp(length, [50, 300], [400, 150])
        volume_percentage = np.interp(length, [50, 300], [0, 100])
        # print(volume_interpreter)
        volume.SetMasterVolumeLevel(volume_interpreter, None)

        if length < 50:
            cv2.circle(img, (centre_x, centre_y), 8, (0, 0, 255), cv2.FILLED)
        elif length > 300:
            cv2.circle(img, (centre_x, centre_y), 8, (255, 0, 0), cv2.FILLED)


    img = cv2.flip(img, 1)
    cv2.rectangle(img, (50, 150), (70, 400), (0, 0, 0), 2)
    cv2.rectangle(img, (50, int(volume_bar)), (70, 400), colour, cv2.FILLED)
    cv2.putText(img, f"{int(volume_percentage)}%", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)


    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time    
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
