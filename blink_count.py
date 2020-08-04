import cv2
import time
import datetime

class Face:
    def __init__(self, w, h): # 顔の予測範囲を設定する。
        self.expected_xs = 0
        self.expected_ys = 0
        self.expected_xe = w
        self.expected_ye = h
        self.maxw = int(w)
        self.maxh = int(h)
        
    def set_pos(self, x, y, w, h): # 顔の検知範囲を記憶＆予測範囲の更新
        self.detected_x = x + self.expected_xs
        self.detected_y = y + self.expected_ys
        self.detected_w = w
        self.detected_h = h
        # self.update_expected()
    
    def update_expected(self): # 予測範囲を決めると、顔認識が結構ぶれるのでとめてる
        self.expected_xs = max(self.detected_x - self.detected_w, 0)
        self.expected_ys = max(self.detected_y - self.detected_h, 0)
        self.expected_xe = min(self.detected_x + self.detected_w, self.maxw)
        self.expected_ye = min(self.detected_y + self.detected_h, self.maxh)

def minutes_report(face_count, blink_count, clk_count):
    print("{}: blink_count:{}, face_c:{} clk_c:{}".format(datetime.datetime.now(), blink_count, face_count, clk_count))
    return 0, 0, 0

# ビデオの起動と、カスケード分類器の読み込み。
cap = cv2.VideoCapture(1)
cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')

# Faceクラスのfaceを生成。
face = Face(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# まばたき数/分を出すためにグローバル変数を定義、
face_count = 0
blink_count = 0
clk_count = 0
not_blinking_time = 0
report_flag = 0
clk = 0.05 # clk second
report_cycle = 5

while True:
    time.sleep(clk)
    clk_count += 1

    if int( time.time() )%report_cycle == 0 and not report_flag: # 
        face_count, blink_count, clk_count= minutes_report(face_count, blink_count, clk_count)
        report_flag = 1
    elif report_flag and int( time.time() )%5 != 0:
        report_flag = 0

    ret, bgr = cap.read()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_gray = gray[
        face.expected_ys : int(face.expected_ye),
        face.expected_xs : int(face.expected_xe)
    ]
    cascade_faces = cascade.detectMultiScale(
        face_gray, scaleFactor=1.3, minNeighbors=3, minSize=(120,120)
    )

    if len(cascade_faces) == 0: # 顔が検知できなかったら、予測範囲のリセットをする。
        face.set_pos(0, 0, face.maxw, face.maxh)
    elif len(cascade_faces)==1:
        face_count += 1
        face.set_pos(*cascade_faces[0])
        cv2.rectangle(
            bgr, 
            (face.detected_x, face.detected_y),
            (face.detected_x+face.detected_w, face.detected_y+face.detected_h),
            (255, 0, 0), 2
        )

        eyes_gray = gray[
            face.detected_y: face.detected_y + int(face.detected_h/2),
            face.detected_x: face.detected_x + face.detected_w
        ]
        eyes = eye_cascade.detectMultiScale(
            eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(10, 10)
        )

        # 目の検知数が０かつ、0.1秒間まばたきをしていないならカウント
        if len(eyes)==0 and not_blinking_time >= 2:
            blink_count += 1
            not_blinking_time = 0
        elif len(eyes)==2:
            not_blinking_time += 1
        
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(
                bgr, 
                (face.detected_x+ex, face.detected_y+ey), 
                (face.detected_x+ex+ew, face.detected_y+ey+eh), 
                (255,255,0), 1
            )

    cv2.putText(bgr, "blink:{}".format(blink_count), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('frame',bgr)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
