import numpy as np
import cv2

img = cv2.imread("input.jpg")
img2 = np.array(img.shape)

#Transpose
#img = img.transpose((1,0,2))

#Horizontal flip
#img2 = img[:,::-1]

#Vertical flip
#img2 = img[::-1,:]

#Sampling
img2 = img[::4,::4]

print(img2.shape)
cv2.imwrite("output.jpg", img2)