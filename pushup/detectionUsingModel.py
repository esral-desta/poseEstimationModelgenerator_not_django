import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open("mymodel.pkl","rb") as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(0)
counter =0
current_stage = "up"

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
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

        try:
            landmarks = ["class"]
            for val in range(1,33+1):
                landmarks += ['x{}'.format(val),'y{}'.format(val),"z{}".format(val),'v{}'.format(val)]


            if landmarks:
                row = np.array([[res.x,res.y,res.z,res.visibility]for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row],columns=landmarks[1:])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()]>=.7:
                #     current_stage = "down"
                # check this
                if current_stage == "down" and body_language_class == "up" and body_language_prob[body_language_prob.argmax()]:
                    counter +=1
                    current_stage = "up"
                    print("counter",counter)
                elif current_stage == "up" and body_language_class == "down" and body_language_prob[body_language_prob.argmax()]:
                    # counter +=1

                    current_stage = "down"
                    print(current_stage)

                cv2.rectangle(image,(0,0),(250,60),(245,117,16),-1)

                cv2.putText(image,"CLASS",
                            (95,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1,cv2.LINE_AA
                            )
                # cv2.putText(image,body_language_class.split(" ")[0],
                cv2.putText(image,current_stage,
                            (90,40),
                                cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2,cv2.LINE_AA
                            )
                cv2.putText(image,"PROB",
                            (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1,cv2.LINE_AA
                            )
                cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2)),
                            (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2,cv2.LINE_AA
                            )

                cv2.putText(image,"COUNT",
                            (180,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1,cv2.LINE_AA
                            )
                cv2.putText(image,str(counter),
                            (175,40),
                                cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2,cv2.LINE_AA
                            )
        except Exception  as e:
            print(e)

        cv2.imshow("Raw Webcam Feed",image)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



# aa=[
# x          : 0.47072380781173706
# y          : 3.7720251083374023
# z          : -0.18670247495174408
# visibility : 6.406049215001985e-05, 
# x          : 0.7597333788871765
# y          : 3.9096837043762207
# z          : -0.7201924324035645
# visibility : 0.0001091237209038809, 
# x          : 0.5230337977409363
# y          : 3.873260498046875
# z          : -1.1689863204956055
# visibility : 0.0002051375195151195
# ]
