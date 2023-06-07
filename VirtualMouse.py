import cv2
import HandTrackingModule as htm
import autopy

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

while cap.isOpened():
    success, img = cap.read()

    

    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    
cap.release()
cv2.destroyAllWindows()