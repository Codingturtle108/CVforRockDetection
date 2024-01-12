import cv2 as cv
import numpy as np
import imgpr
Capture = cv.VideoCapture(0)
while True:
    read,frame = Capture.read()
    if not read:
        print("Image Feed Not Read")
        continue
    edges = cv.Canny(frame,50,100)
    cv.imshow('Video',edges)
    print(f'Image Read')
    if cv.waitKey(1) & 0xff == ord('q'):
        break
Capture.release()
cv.destroyAllWindows()