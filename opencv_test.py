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
e0 = cv2.getTickCount()
cp2 = cproc.clarity_processor(img2)
e1 = cv2.getTickCount()
res2 = cp2.process_gpu3(img2,0.95)
e2 = cv2.getTickCount()
cv2.imshow('2nd image...',cv2.multiply(res2,10))
t = (e1 - e0)/cv2.getTickFrequency()
print('Calibration time ',t)
t = (e2 - e1)/cv2.getTickFrequency()
print('Processing time ',t)

# Benchmark different processing routines

# Original code (runs on GPU)
# Pass numpy image, take subimages then convert to UMat
e0 = cv2.getTickCount()

for i in range(5000) :
    res2=cp2.process(img2,0.95)

e1 = cv2.getTickCount()

t = (e1 - e0)/cv2.getTickFrequency()
print('5000 original gpu loop ',t/5,' ms')

# Original code (runs on CPU)
e0 = cv2.getTickCount()

for i in range(5000) :
    res2=cp2.process_cpu(img2,0.95)

e1 = cv2.getTickCount()
t = (e1 - e0)/cv2.getTickFrequency()
print('5000 cpu loop ',t/5,' ms')

# GPU but passing 2 UMat images
width=int(np.size(img2,1)/2)
img_l=cv2.UMat(img2[:,0:width])
img_r=cv2.UMat(img2[:,width:])
e0 = cv2.getTickCount()

for i in range(5000) :
    res2=cp2.process_gpu1(img_l,img_r,0.95)

e1 = cv2.getTickCount()

t = (e1 - e0)/cv2.getTickFrequency()
print('5000 gpu1 loop ',t/5,' ms')

# GPU, passing 1 numpy image, convert to UMat, then take subimages
e0 = cv2.getTickCount()

for i in range(5000) :
    res2=cp2.process_gpu2(img2,0.95)

e1 = cv2.getTickCount()

t = (e1 - e0)/cv2.getTickFrequency()
print('5000 gpu2 loop ',t/5,' ms')

# GPU, passing 1 numpy image, convert to UMat, then take subimages, use scaleAdd routine
e0 = cv2.getTickCount()

for i in range(5000) :
    res2=cp2.process_gpu3(img2,0.95)

e1 = cv2.getTickCount()

t = (e1 - e0)/cv2.getTickFrequency()
print('5000 gpu2 loop ',t/5,' ms')

res3 = cp.process(img2,0.95)
cv2.imshow('Wrong calibration!',cv2.multiply(res3,10))

cv2.waitKey(0)
