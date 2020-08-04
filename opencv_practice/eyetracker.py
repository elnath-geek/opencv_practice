import cv2
import dlib
import numpy as np
import time
import datetime
from imutils import face_utils
from scipy.spatial import distance

def calc_ear(eye): # EARの計算
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def blink_report(face_count, blink_count, clk_count):
    print("{}: blink_count:{}, face_c:{}, clk_c:{}".format(datetime.datetime.now(), blink_count, face_count, clk_count))

def detect_blink(face_parts, bgr_frame, blink_count, not_blinking_time):
    # 左目のランドマークとEARの計算
    left_eye_ear = calc_ear(face_parts[42:48])
    cv2.putText(bgr_frame, "left eye EAR:{} ".format(round(left_eye_ear, 3)), 
        (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 右目のランドマークとEARの計算
    right_eye_ear = calc_ear(face_parts[36:42])
    cv2.putText(bgr_frame, "right eye EAR:{} ".format(round(right_eye_ear, 3)), 
        (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # まばたきの検知
    if (left_eye_ear + right_eye_ear) < 0.40 and not_blinking_time >= 2:
        blink_count += 1
        not_blinking_time = 0
    elif (left_eye_ear + right_eye_ear) > 0.50:
        not_blinking_time += 1

    return blink_count, not_blinking_time

def detect_headpose(face_parts, bgr_frame, frame_size):
    # 顔のパーツを何点か取り出し、顔の方向を計算する。
    # そのために、フレーム上のパーツ位置と現実のパーツ位置の調整を行う。
    frame_parts_points = np.array([
        tuple(face_parts[30]),#鼻頭
        tuple(face_parts[21]),
        tuple(face_parts[22]),
        tuple(face_parts[39]),
        tuple(face_parts[42]),
        tuple(face_parts[31]),
        tuple(face_parts[35]),
        # tuple(face_parts[48]),
        # tuple(face_parts[54]),
        tuple(face_parts[57]),
        tuple(face_parts[8]),
    ], dtype='double')

    model_parts_points = np.array([
        (0.0,0.0,0.0), # 30
        (-30.0,-125.0,-30.0), # 21
        (30.0,-125.0,-30.0), # 22
        (-60.0,-70.0,-60.0), # 39
        (60.0,-70.0,-60.0), # 42
        (-40.0,40.0,-50.0), # 31
        (40.0,40.0,-50.0), # 35
        # (-70.0,130.0,-100.0), # 48
        # (70.0,130.0,-100.0), # 54
        (0.0,158.0,-10.0), # 57
        (0.0,250.0,-50.0) # 8
        ])

    # 
    focal_length = frame_size[0]
    frame_center = (frame_size[0]/2, frame_size[1]/2)
    camera_matrix = np.array([
        [focal_length, 0, frame_center[0]],
        [0, focal_length, frame_center[1]],
        [0, 0, 1]
    ], dtype = "double")

    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_parts_points, frame_parts_points, camera_matrix,
        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    #回転行列とヤコビアン
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))

    #yaw,pitch,rollの取り出し
    (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
    yaw = eulerAngles[1]
    pitch = eulerAngles[0]
    roll = eulerAngles[2]

    # print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll)) # 頭部姿勢データの取り出し

    cv2.putText(bgr_frame, 'yaw : ' + str(yaw), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(bgr_frame, 'pitch : ' + str(pitch), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(bgr_frame, 'roll : ' + str(roll), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    #計算に使用した点のプロット/顔方向のベクトルの表示
    for p in frame_parts_points:
        cv2.drawMarker(bgr_frame, (int(p[0]), int(p[1])), (0.0, 1.409845, 255), markerType=cv2.MARKER_CROSS, thickness=1)

    p1 = (int(frame_parts_points[0][0]), int(frame_parts_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.arrowedLine(bgr_frame, p1, p2, (255, 0, 0), 2)

    return yaw, pitch, roll


def main():
    # 分類器の学習データ読み込み
    face_detector = dlib.get_frontal_face_detector()
    face_parts_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

    # 定数の宣言
    CAMERA_ID = 1
    CLK = 0.03 #
    REPORT_CYCLE = 5

    # グローバルっぽい変数宣言
    capture = cv2.VideoCapture(CAMERA_ID)
    face_count = 0
    blink_count = 0
    clk_count = 0
    not_blinking_time = 0
    reporting_flag = 0

    frame_size = [
        capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ]

    # 画像処理のメインループ sleepで上限決めつつ処理する。
    while(True):
        time.sleep(CLK)
        clk_count += 1

        # REPORT_CYCLE に応じてデータを吐き出し、リセットをする。
        if int( time.time() )%REPORT_CYCLE == 0 and not reporting_flag:
            blink_report(face_count, blink_count, clk_count)
            face_count = blink_count = clk_count = 0
            reporting_flag = 1
        elif reporting_flag and int( time.time() )%REPORT_CYCLE != 0:
            reporting_flag = 0

        # 動画の1frameを取得し、顔の検出を行う。
        _, bgr_frame = capture.read()
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector(gray_frame, 0)

        # 顔が一つ検出できた場合、顔のパーツの検出に進む
        if len(detected_faces) == 1:
            face_count += 1
            
            # 68ランドマークへの変形
            face_parts = face_parts_predictor(gray_frame, detected_faces[0])
            face_parts = face_utils.shape_to_np(face_parts)
            
            # まばたき判断の関数を動かす
            (blink_count, not_blinking_time) = detect_blink(face_parts, bgr_frame, blink_count, not_blinking_time)

            # 顔の姿勢計算の関数を動かす
            (yaw, pitch, roll) = detect_headpose(face_parts, bgr_frame, frame_size)

        cv2.imshow('frame', bgr_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
