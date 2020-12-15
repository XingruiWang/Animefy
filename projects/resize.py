import cv2 as cv
import os


dir = 'real'

file = os.listdir(dir)

for f in file:
    img = cv.imread(os.path.join(dir, f))
    img = cv.resize(img, (512, 512), interpolation = cv.INTER_CUBIC)
    cv.imwrite(os.path.join(dir, f), img)



