import cv2
import numpy as np

img = cv2.imread('test3.jpg') # Read the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale

_, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV) # Threshold the image
kernal = np.ones((2, 2)) # Create a convolution kernel

dilation = cv2.dilate(thresh, kernal, iterations=2) # Dilate the image

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours

print("Number of objects detected :",len(contours)) # Print the number of contours


cv2.imshow('img', img) # Show the image
cv2.imshow('thresh', thresh) # Show the image
cv2.imshow('dilation', dilation) # Show the image
cv2.waitKey(0) # Wait for keystroke