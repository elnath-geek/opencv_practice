import cv2
import dlib
from imutils import face_utils

class GazeTracker(object):
    def __init__(self):
        self.bgr = None
        self.face = None #[x, y, w, h]
        self.eyes = None
        self.eye_left = None
        self.eye_right = None

        self.face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye_tree_eyeglasses.xml')
        self.face_parts_predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')
    
    @property
    def all_detected(self):
        try:
            int(len(self.face))
            int(len(self.eyes))
            tuple(self.eye_left)
            tuple(self.eye_right)
            return True
        except Exception:
            return False


    def get_center(self, gray):
        moments = cv2.moments(gray, False)
        try:
            return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
        except:
            return None

    def is_close(self, y0, y1):
        if abs(y0 - y1) < 10:
            return True
        return False

    def eye_point(self, gray, landmarks, left=True):
        if left:
            eyes = [
                    landmarks[36],
                    min(landmarks[37], landmarks[38], key=lambda x: x.y),
                    max(landmarks[40], landmarks[41], key=lambda x: x.y),
                    landmarks[39],
                    ]
        else:
            eyes = [
                    landmarks[42],
                    min(landmarks[43], landmarks[44], key=lambda x: x.y),
                    max(landmarks[46], landmarks[47], key=lambda x: x.y),
                    landmarks[45],
                    ]
        org_x = eyes[0].x
        org_y = eyes[1].y
        if self.is_close(org_y, eyes[2].y):
            return None

        eye = gray[org_y:eyes[2].y, org_x:eyes[-1].x]
        _, eye = cv2.threshold(eye, 150, 255, cv2.THRESH_BINARY_INV)

        center = self.get_center(eye)
        if center:
            return center[0] + org_x, center[1] + org_y
        return center


    def _analyze(self):
        gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=3, minSize=(100,100)
        )
        
        try:
            self.face = faces[0]

            # 両目の枠を取る
            eyes_gray = gray[
                self.face[1]:self.face[1] + int(self.face[3]/2),
                self.face[0]:self.face[0] + self.face[2],
            ]
            self.eyes = self.eye_cascade.detectMultiScale(
                eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(10, 10)
            )

            # landmark から左右の目の座標を求める。
            dlib_face = dlib.rectangle(faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3])
            landmarks = self.face_parts_predictor(gray, dlib_face).parts()
            # landmarks = face_utils.shape_to_np(landmarks)
            self.eye_left = self.eye_point(gray, landmarks)
            self.eye_right = self.eye_point(gray, landmarks, False)

        except IndexError:
            self.face = None
            self.eyes = None
            self.eye_left = None
            self.eye_right = None


    def refresh(self, bgr):
        self.bgr = bgr
        self._analyze()


    def imshow_frame(self):
        bgr = self.bgr.copy()

        if self.all_detected:
            cv2.rectangle( # face_rectangle
                bgr, 
                (self.face[0], self.face[1]),
                (self.face[0] + self.face[2], self.face[1] + self.face[3]),
                (255, 0, 0), 2
            )
            for ex, ey, ew, eh in self.eyes: #eye_rectangle
                cv2.rectangle(
                    bgr, 
                    (self.face[0]+ex, self.face[1]+ey),
                    (self.face[0]+ex+ew, self.face[1]+ey+eh),
                    (255,255,0), 1
                )
            # eye_pupil circle
            cv2.circle(bgr, tuple(self.eye_left), 3, (255, 255, 0), -1)
            cv2.circle(bgr, tuple(self.eye_right), 3, (255, 255, 0), -1)

            if len(self.eyes) == 2:
                eyes_point = [
                    [self.face[0] + self.eyes[0][0] + self.eyes[0][2]/2, self.face[1] + self.eyes[0][1] + self.eyes[0][3]/2], 
                    [self.face[0] + self.eyes[1][0] + self.eyes[1][2]/2, self.face[1] + self.eyes[1][1] + self.eyes[1][3]/2], 
                ]
                
                x_diff = (eyes_point[1][0] - self.eye_left[0]) + (eyes_point[0][0] - self.eye_right[0])
                y_diff = (eyes_point[1][1] - self.eye_left[1]) + (eyes_point[0][1] - self.eye_right[1])

                cv2.putText(bgr, "x_diff:{} ".format(x_diff), 
                (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr, "y_diff:{} ".format(y_diff), 
                (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                gaze_x = x_diff*(-100) + 2000
                gaze_y = y_diff*(-300)
                cv2.circle(bgr, (int(gaze_x), int(gaze_y)), 10, (255,255,255), -1)

        return bgr
