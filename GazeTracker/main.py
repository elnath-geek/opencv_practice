import cv2
import dlib
import time
from gaze_tracker import GazeTracker

if __name__ == '__main__':

    CAMERA_ID = 1
    CLK = 0.03 #

    gaze = GazeTracker()
    webcam = cv2.VideoCapture(CAMERA_ID)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        time.sleep(CLK)
        
        _, bgr = webcam.read()
        bgr = cv2.resize(bgr, dsize=(2560, 1440))

        gaze.refresh(bgr)
        bgr = gaze.imshow_frame()

        cv2.imshow("test", bgr)

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


