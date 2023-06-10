import cv2
import mediapipe as mp
import math


class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5, upBody = False):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.upBody = upBody

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon,
                                     upBody = self.upBody)
    
    def findPose(self, img, draw=True):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    

def main():
    vid_path = str(input("Enter video path: "))

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(vid_path)
    # fps= int(cap.get(cv2.CAP_PROP_FPS))


    while cap.isOpened():
        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(imgRGB)

        img = cv2.resize(img, (550,600))

        if success:

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape

                    cx , cy = int(lm.x*w) , int(lm.y*h)
                

            # time.sleep(1/fps)
            cv2.imshow("vid", img)

            if cv2.waitKey(20) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()