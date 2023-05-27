import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

## For volume control 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# web cam height and width 
wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

minVol = volRange[0]
maxVol = volRange[1]

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findhands(img)
    lmList = detector.findPositions(img, draw= False)
    
    # getting thumb =4 and first finger tip 8
    if len(lmList) !=0:

        x1, y1 = lmList[4][1] , lmList[4][2] # finger coordinate
        x2, y2 = lmList[8][1] , lmList[8][2] # thumb point coordinate
        cx,cy = (x1+x2)//2, (y1+y2)//2  # center of finger and thumb

        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)

        cv2.line(img , (x1,y1), (x2,y2) , (255,0,255), 2)

        length = math.hypot(x2-x1, y2-y1)

        if length< 30:
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)

    if success:
        cv2.imshow("img",img)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()