import json
import csv
import os
import cv2
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
with open("coordsNew.csv",mode="w",newline="") as f:
    csv_wirter = csv.writer(f,delimiter=",",quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_wirter.writerow(landmarks)

### saving single row of a video 
def export_landmark(results,action):
    try:
        keypoints = np.array([[res.x,res.y,res.z,res.visibility]for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0,action)
        if action == "down":
            print(action)
        with open("coords.csv",mode="a",newline="") as f:
            csv_wirter = csv.writer(f,delimiter=",",quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_wirter.writerow(keypoints)
    except Exception as e:
        pass
# export_landmark(results,"up")

#import cv2

cap = cv2.VideoCapture("djangovideouploaddisplay/uploads/newexersice/Update __ እመንኒ ( Emenni) - Amanuel Yemane ( ኣማኑኤል የማነ ) New Tigrigna Music 2023.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)
# minutes = 0
# seconds = 28
# frame_id = int(fps*(minutes*60 + seconds))
# print('frame id =',frame_id)
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
# ret, frame = cap.read()

# t_msec = 1000*(minutes*60 + seconds)
# cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
# ret, frame = cap.read()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    data = ""
    with open("The Deadlift_ CrossFit Foundational Movement.mp4") as f:
        data = json.load(f)
        for label in data.keys():
            for seconds,minutes in data[label]:
                frame_id = int(fps*(minutes*60 + seconds))
                print('frame id =',frame_id)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

                try:
                    ret, frame = cap.read()
                

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    results = pose.process(image)

                    export_landmark(results,label) 
       
                except Exception as e:
                    break
cap.release()
cv2.destroyAllWindows()





        