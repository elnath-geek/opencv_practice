import cv2
import numpy as np

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    print(frame)
    if not ret:
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame, (5,5))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key==27:
        break

    
cap.release()
cv2.destroyAllWindows()
