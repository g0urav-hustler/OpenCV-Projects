import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

vid_path = "dancer_videos/one_pers_dance_1.mp4"
cap = cv2.VideoCapture(vid_path)
# fps= int(cap.get(cv2.CAP_PROP_FPS))


while cap.isOpened():
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    img = cv2.resize(img, (550,600))

    if success:

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape

                cx , cy = int(lm.x*w) , int(lm.y*h)
               

        # time.sleep(1/fps)
        cv2.imshow("vid", img)

        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()