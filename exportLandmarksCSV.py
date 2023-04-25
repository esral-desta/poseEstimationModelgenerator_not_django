import csv
import os
import numpy as np
from matplotlib import pyplot as plt

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ["class"]
for val in range(1,33+1):
    landmarks += ['x{}'.format(val),'y{}'.format(val),"z{}".format(val),'v{}'.format(val)]


## lables of the csv
landmarks[1:]

## saving the lables
with open("newmodel.csv",mode="w",newline="") as f:
    csv_wirter = csv.writer(f,delimiter=",",quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_wirter.writerow(landmarks)

### saving single row of a video 
def export_landmark(results,action):
    try:
        keypoints = np.array([[res.x,res.y,res.z,res.visibility]for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0,action)
        if action == "down":
            print(action)
        with open("newmodel.csv",mode="a",newline="") as f:
            csv_wirter = csv.writer(f,delimiter=",",quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_wirter.writerow(keypoints)
    except Exception as e:
        pass
# export_landmark(results,"up")

import cv2

cap = cv2.VideoCapture("Can Cam do 100 Push Ups Unbroken _ That's Good Money.mp4")
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        try:
            ret,frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                             mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,236),thickness=2,circle_radius=2)
                )


            cv2.imshow("Recorded video analysis",image)
            # if cv2.waitKey(10) & 0xff == ord("q"):
            #     break

            k = cv2.waitKey(1)
            print("k",k)
            if k == 84:
                print("down")
                export_landmark(results,"down") 
            if k == 82:
                export_landmark(results,"up")
                # break
       
        except Exception as e:
            break
cap.release()
cv2.destroyAllWindows()
