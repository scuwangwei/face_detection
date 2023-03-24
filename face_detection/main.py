import cv2
# import numpy as np

# kernel = np.ones((3, 3), np.uint8)

img = cv2.imread('face1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階

faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceDetect = faceCascade.detectMultiScale(gray_img, 1.1, 3)#parameter:img 縮小倍數 偵測次數(越大越嚴謹)

for (x, y, w, z) in faceDetect:
    cv2.rectangle(img, (x, y), (x + w, y + z), (0, 255, 0) ,2)

cv2.imshow('img', img)
cv2.waitKey(0)

