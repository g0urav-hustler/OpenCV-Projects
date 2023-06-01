import cv2
import HandTrackingModule as htm
import numpy as np
import os

folder_path = "canva_photos"

image_list = os.listdir(folder_path)
image_list.sort()
# print(image_list)

overlayList = []

for img_name in image_list:
    image = cv2.imread(os.path.join(folder_path, img_name))
    overlayList.append(image)

# first image as header
header = overlayList[0]

wcam, hcam = 1280,720
cap = cv2.VideoCapture(0)
# cap.set(3, wcam)
# cap.set(4, hcam)

drawColor = (255, 0, 255)
detector = htm.handDetector(detectionCon= 0.85) # high confidence

xp , yp = 0,0

imgCanvas = np.zeros((720,1280,3 ), np.uint8)
brushThickness = 15
eraserThickness = 30

while cap.isOpened():

    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img , (wcam, hcam))

    img[0:125, 0:1280] = header


    """ 
    1. import image
    2. find hand landmarks
    3. which finger is up
    4. selection mode if two finger up
    5. drawing mode if index finger is up

    """

    # find hand landmarks
    img = detector.findhands(img)

    lmList = detector.findPositions(img, draw= False)

  

    if len(lmList) != 0:
        # tip of indexes
        x1,y1 = lmList[8][1], lmList[8][2]
        x2,y2 = lmList[12][1], lmList[12][2]

         # fingers up
        fingers = detector.fingersUp()

        if fingers[0] and fingers[1]:
            xp,yp=0,0
            #print("Selection Mode")
            #checking for click
            if y1 < 125:
                if 250 < x1 < 450:#if i m clicking at purple brush
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:#if i m clicking at blue brush
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:#if i m clicking at green brush
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:#if i m clicking at eraser
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)#selection mode is represented as rectangle


        
        if fingers[0] and (fingers[1] == False):

            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)

            if xp ==0 and yp ==0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)#gonna draw lines from previous coodinates to new positions 
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1
            #print("Drawing mode")
    
    # 1 converting img to gray
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    # 2 converting into binary image and thn inverting
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)#on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask
    
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)#converting again to gray bcoz we have to add in a RGB image i.e img
    
    #add original img with imgInv ,by doing this we get our drawing only in black color
    img = cv2.bitwise_and(img,imgInv)
    
    #add img and imgcanvas,by doing this we get colors on img
    img = cv2.bitwise_or(img,imgCanvas)
    

    if success:
        cv2.imshow("img", img)
        # cv2.imshow("canvas", imgCanvas)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
