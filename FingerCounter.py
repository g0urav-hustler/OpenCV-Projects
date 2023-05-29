import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()

    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    

cap.release()
cv2.destroyAllWindows()