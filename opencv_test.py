import numpy as np
import cv2
import clarity_process as cproc


# Try first calibration image
img = cv2.imread('calib_h.tiff',cv2.IMREAD_GRAYSCALE)
cv2.imshow('input image',img)

e0 = cv2.getTickCount()

cp = cproc.clarity_processor(img)

e1 = cv2.getTickCount()

res = cp.process(img,0.95)

e2 = cv2.getTickCount()

t = (e1 - e0)/cv2.getTickFrequency()
print('Calibration time ',t)
t = (e2 - e1)/cv2.getTickFrequency()
print('Processing time ',t)

result=cv2.multiply(res,10)

cv2.imshow('Registered and subtracted image',result)

# Try second test calibration image
img2 = cv2.imread('calib_rot_h.tiff',cv2.IMREAD_GRAYSCALE)
cp2 = cproc.clarity_processor(img2)
res2 = cp2.process(img2,0.95)
cv2.imshow('2nd image...',cv2.multiply(res2,10))

res3 = cp.process(img2,0.95)
cv2.imshow('Wrong calibration!',cv2.multiply(res3,10))

cv2.waitKey(0)
