import cv2
import HandTrackingModule as htm
import autopy
import numpy as np


wScr, hScr = autopy.screen.size() # getting screen size 
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector()
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)

    cor_list , img = detector.findHands(img, draw= False)

    if len(cor_list) != 0:
        lmList = cor_list[0]["lmList"]
        x1,y1 = lmList[8][0], lmList[8][1]
        x2,y2 = lmList[12][0], lmList[12][1]

        fingers = detector.fingersUp({"type": "Right", "lmList": lmList})

        print(x1,y1)

        # selection mode
        # if fingers[1] and fingers[2]:
        #     # xp,yp=0,0
        
        # moving mode
        if fingers[1] and (fingers[2] == False):
            # xp,yp=x1,y1

            # converting coordinates
            x3 = np.interp(x1, (0,wCam), (0,wScr))
            y3 = np.interp(y1, (0,hCam), (0,hScr))
            
            # move the mouse
            autopy.mouse.move(x3,y3)

    
    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    
cap.release()
cv2.destroyAllWindows()