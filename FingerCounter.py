import cv2
import HandTrackingModule as htm



detector = htm.handDetector()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findhands(img)
    lmList = detector.findPositions(img, draw= False)

    width_index =1
    height_index = 2

    if len(lmList) !=0:

        if lmList[8][height_index] < lmList[6][2]:
            print("fingers UP ")
        else:
            print("fingers down")




    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    

cap.release()
cv2.destroyAllWindows()