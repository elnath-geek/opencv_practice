import cv2
import dlib
import time
import datetime
from imutils import face_utils
from scipy.spatial import distance

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def minutes_report(face_count, blink_count, clk_count):
    print("{}: blink_count:{}, face_c:{},clk_c:{}".format(datetime.datetime.now(), blink_count, face_count, clk_count))
    return 0, 0, 0

# ビデオの起動と、カスケード分類器の読み込み。
cap = cv2.VideoCapture(1)
cuda_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default_cuda.xml')
face_parts_cascade = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# もろもろの変数定義
face_count = 0
blink_count = 0
clk_count = 0
not_blinking_time = 0
report_flag = 0
clk = 0.05 # clk second
report_cycle = 60 # second for 1 report 

cuda_bgr = cv2.cuda_GpuMat()
cuda_gray = cv2.cuda_GpuMat()
cuda_cascade_faces = cv2.cuda_GpuMat()

while True:
    time.sleep(clk)
    clk_count += 1

    if int( time.time() )%report_cycle == 0 and not report_flag: # 
        face_count, blink_count, clk_count= minutes_report(face_count, blink_count, clk_count)
        report_flag = 1
    elif report_flag and int( time.time() )%5 != 0:
        report_flag = 0

    ret, bgr = cap.read()
    cuda_bgr.upload(bgr)
    cuda_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cuda_cascade_faces = cuda_cascade.detectMultiScale(
        cuda_gray, scaleFactor=1.3, minNeighbors=3, minSize=(100,100)
    )

    if len(cuda_cascade_faces)==1:
        face_count += 1

        x, y, w, h = cuda_cascade_faces[0, :]
        cv2.cuda.rectangle(cuda_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gray = cuda_gray.download()

        face = dlib.rectangle(x, y, x + w, y + h)
        face_parts = face_parts_cascade(gray, face)
        face_parts = face_utils.shape_to_np(face_parts)
        
        left_eye_ear = calc_ear(face_parts[42:48])
        cv2.putText(bgr, "left eye EAR:{} ".format(round(left_eye_ear, 3)), 
            (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        right_eye_ear = calc_ear(face_parts[36:42])
        cv2.putText(bgr, "right eye EAR:{} ".format(round(right_eye_ear, 3)), 
            (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        if (left_eye_ear + right_eye_ear) < 0.40 and not_blinking_time >= 2:
            blink_count += 1
            not_blinking_time = 0
        elif (left_eye_ear + right_eye_ear) > 0.50:
            not_blinking_time += 1
        
    cv2.putText(bgr, "blink:{}".format(blink_count), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('frame',bgr)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
