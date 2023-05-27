import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math


####### linux system
from pynput.keyboard import Key,Controller
keyboard = Controller()


##### Windows system ######
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# # volume = interface.QueryInterface(IAudioEndpointVolume)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# # volume.GetMute()
# # volume.GetMasterVolumeLevel()
# volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)

# minVol = volRange[0]
# maxVol = volRange[1]
# vol =0

####### /windows


# web cam height and width 
wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handDetector()


last_length = None

volBar = 400
volper =0

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

        cv2.line(img , (x1,y1),(x2,y2), (255,0,255), 2)

        length = math.hypot(x2-x1, y2-y1)

        # hand length = 30 to 300
        # volume = -65 to 0

        # vol = np.interp(length, [30,300], [minVol, maxVol])
        volBar = np.interp(length, [30,280], [400, 150])
        volper = np.interp(length, [30,300], [0,100])

        # print(vol)
        # volume.SetMasterVolumeLevel(vol, None)

        if last_length:
            if length>last_length:
                keyboard.press(Key.media_volume_up)
                keyboard.release(Key.media_volume_up)
                print("VOL UP")
            elif length<last_length:
                keyboard.press(Key.media_volume_down)
                keyboard.release(Key.media_volume_down)
                print("VOL DOWN")
        
        # last_angle=angle
        last_length=length

        if length< 30:
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (255,0,0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85,400), (255,0,0), cv2.FILLED )

    cv2.putText(img, str(int(volper)), (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

            

    if success:
        cv2.imshow("img",img)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()