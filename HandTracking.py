import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands # compulsory code

hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

# for timing
ptime, ctime = 0,0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # it require rgb image

    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:  # if hand mark is detected
        for handLms in result.multi_hand_landmarks: # for more hands
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape # hieght, width , channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Checking ids
                cv2.putText(img, str(int(id)), (cx,cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)


            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    
    # puting frame rates
    ctime = time.time()
    fps = 1/(ctime -ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3 ) 

    
    cv2.imshow("image", img)
    cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()
