import cv2
import mediapipe as mp
import time



# mphands = mp.solutions.hands # compulsory code

# hands = mphands.Hands()
# mpDraw = mp.solutions.drawing_utils


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
            """
            :param mode: In static mode, detection is done on each image: slower
            :param maxHands: Maximum number of hands to detect
            :param detectionCon: Minimum Detection Confidence Threshold
            :param minTrackCon: Minimum Tracking Confidence Threshold
            """
            self.mode = mode
            self.maxHands = maxHands
            self.detectionCon = detectionCon
            self.minTrackCon = minTrackCon

            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                            min_detection_confidence=self.detectionCon,
                                            min_tracking_confidence=self.minTrackCon)
            self.mpDraw = mp.solutions.drawing_utils
            self.tipIds = [8, 12, 16, 20]
        
    def findhands(self,img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # it require rgb image
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:  # if hand mark is detected
            for handLms in self.results.multi_hand_landmarks: # for more hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        

        return img
    


    def findPositions(self, img, handNo = 0, draw = True):
         
         self.lmList = [] # landmark list

         if self.results.multi_hand_landmarks:
              myHand = self.results.multi_hand_landmarks[handNo]
              
              for id, lm in enumerate(myHand.landmark):
                    h,w,c = img.shape # hieght, width , channels
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    self.lmList.append([id,cx,cy])
                    # Checking ids
                    if draw:
                        cv2.putText(img, str(int(id)), (cx,cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

         return self.lmList
    
    def fingersUp(self, no_fingers = 4):#checking which finger is open 
        fingers = []#storing final result
        # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign
        

        # Fingers
        if len(self.lmList) !=0:
            for id in range(no_fingers):#checking tip point is below tippoint-2 (only in Y direction)
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

def main():

    ptime, ctime = 0,0
    cap = cv2.VideoCapture(0)

    detector = handDetector() # making the object 

    while cap.isOpened():
        success, img = cap.read()

        img = detector.findhands(img)

        lmlist = detector.findPositions(img)
        
        if len(lmlist) !=0:
            print(lmlist[8])


        ctime = time.time()
        fps = 1/(ctime -ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3 ) 

        
        if success:
         cv2.imshow("img",img)
         
         if cv2.waitKey(25) & 0xff == ord('q'):
            break
    


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()


