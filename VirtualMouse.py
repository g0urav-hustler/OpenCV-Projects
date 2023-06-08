import cv2
import HandTrackingModule as htm
import autopy
import numpy as np


wScr, hScr = autopy.screen.size() # getting screen size 
fReduce = 50 # for detection
smoothening = 5 # for smooth cursor 
plocx, plocy = 0,0
clocx, clocy = 0,0

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



        # detection area
        cv2.rectangle(img ,(fReduce, fReduce), (wCam -fReduce, hCam-fReduce), (255,0,255), 8)


        # selection mode
        if fingers[1] and fingers[2]:
            length ,info, img = detector.findDistance([x1,y1],[x2,y2], img)
            cx, cy = info[4],info[5]

            # clocx = plocx + (cx - plocx)/ smoothening
            # clocy = plocy + (cy - plocy)/ smoothening



            if length < 30:
                cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)

                # click the mouse
                autopy.mouse.click()
            
            # moving in two finger
            # autopy.mouse.move(clocx, clocy)
            # plocx , plocy = clocx, clocy

        
        # moving mode
        if fingers[1] and (fingers[2] == False):
            # xp,yp=x1,y1

            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            # converting coordinates
            x3 = np.interp(x1, (fReduce,wCam-fReduce), (0,wScr))
            y3 = np.interp(y1, (fReduce,hCam-fReduce), (0,hScr))

            # cursor smoothing
            clocx = plocx + (x3 - plocx)/ smoothening
            clocy = plocy + (y3 - plocy)/ smoothening
            
            # move the mouse
            autopy.mouse.move(clocx,clocy)
            plocx , plocy = clocx, clocy

    
    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    
cap.release()
cv2.destroyAllWindows()