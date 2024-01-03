#Python program to perform binary thresholding
import numpy as np, cv2

#Input
img = cv2.imread("input.jpg")

#Grayscale
img2 = np.zeros(img.shape)
img2 = np.sum(img,2)/3

#Binary Thresholding
T = 127  #Threshold value
img2[img2>T] = 255 #255 is white 
img2[img2<=T] = 0  #0 is black 

#Output
cv2.imwrite("output.jpg", img2)