import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow("MediaPipe Feed",frame)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()