import cv2
import HandTrackingModule as htm



detector = htm.handDetector()

finger_dict = {"first_finger": {"tip_point":8 , "lower_point": 6},
               "second_finger": {"tip_point":12 , "lower_point": 10},
               "third_finger": {"tip_point":16 , "lower_point": 14},
               "fourth_finger": {"tip_point":20 , "lower_point": 18}}

# web cam height and width 
wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
finger_count =0
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findhands(img)
    lmList = detector.findPositions(img, draw= False)

    width_index =1
    height_index = 2

    if len(lmList) !=0:
        finger_count =0
        
        for key,value in finger_dict.items():
            if lmList[value["tip_point"]][height_index] < lmList[value["lower_point"]][height_index]:
                finger_count = finger_count +1
        

        # print("Total fingers up", finger_count)
    
    cv2.rectangle(img, (50,30), (200,200), (255,255,0), cv2.FILLED )
    cv2.putText(img,str(int(finger_count)), (100,140), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 2)



    if success:
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    

cap.release()
cv2.destroyAllWindows()