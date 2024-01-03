#move, transparent
import numpy as np
import cv2

def move(img2, img,a,b,c,d):
    img2[a:b+img.shape[0],c:d+img.shape[1]] = (img2[a:b+img.shape[0],c:d+img.shape[1]]+img[:,:])*0.5
    return img2

img = cv2.imread("pusheen0.jpg")

img2 = np.zeros((305,305,3))
img2 = move(img2, img,0,0,0,0)
img2[:,:,0] = img2[:,:,0]*0.2
img2 = move(img2,img,40,40,40,40)
img2[:,:,1] = img2[:,:,1]*0.2
img2 = move(img2,img,80,80,80,80)
img2[:,:,2] = img2[:,:,2]*0.2

cv2.imwrite("output.jpg", img2)