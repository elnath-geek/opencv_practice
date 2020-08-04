import cv2
import dlib
import time
import datetime
from imutils import face_utils
from scipy.spatial import distance
import numpy as np

# ビデオの起動と、カスケード分類器の読み込み。
cap = cv2.VideoCapture(1)
cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_parts_cascade = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# もろもろの変数定義
face_count = 0
blink_count = 0
clk_count = 0
not_blinking_time = 0
report_flag = 0
clk = 0.1 # clk second
report_cycle = 5 # second for 1 report 


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ans_pos = [680, 0]
zero_point_idx = 27
zero_point_idx_x = 27
zero_point_idx_y = 32

while True:
    time.sleep(clk)
    clk_count += 1
    # ans_pos[0] += 10
    # ans_pos[0] %+ 2560
    # ans_pos[1] += 10
    # ans_pos[1] %+ 1440

    ret, bgr = cap.read()
    bgr = cv2.resize(bgr, dsize=(2560, 1440))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    cv2.circle(gray, tuple(ans_pos), 5, (0,0,0), -1)

    cascade_faces = cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=3, minSize=(100,100)
    )

    if len(cascade_faces)==1:
        face_count += 1

        x, y, w, h = cascade_faces[0, :]
        # cv2.rectangle(bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = dlib.rectangle(x, y, x + w, y + h)
        face_parts = face_parts_cascade(gray, face)
        face_parts = face_utils.shape_to_np(face_parts)
        
        for (x, y) in face_parts: #顔全体の68箇所のランドマークをプロット
            cv2.circle(bgr, (x, y), 1, (255, 255, 255), -1)

        est_x = 3785 + (face_parts[0][0] - face_parts[zero_point_idx_x][0])*10.45
        est_y = 4256 + (face_parts[8][1] - face_parts[zero_point_idx_y][1])*(-16.75)

        cv2.putText(bgr, "est_x:{} est_y:{} ".format(round(est_x, 3), round(est_y, 3)), 
            (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(bgr, (int(est_x), int(est_y)), 30, (0,0,0), -1)
        
        # points = []
        # for i in range(len(face_parts)):
        #     points.append([
        #         face_parts[i][0] - face_parts[zero_point_idx][0],
        #         face_parts[i][1] - face_parts[zero_point_idx][1]
        #     ])
        # print(*sum(points, []), *ans_pos)


        
    cv2.imshow('frame', bgr)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
