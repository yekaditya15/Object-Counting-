import cv2
import numpy as np
            

class cropImage:
    def __init__(self, image):
        self.image = image
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.cropping = False
        self.imgOrg = image.copy()

        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.mouse_crop)

        cv2.imshow("image", image)
        while True:
            self.i = image.copy()
            if cv2.waitKey(1) == ord('z'):
                break

    def getCropped(self):
        return self.roi
    
    def mouse_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping == True:
                self.x_end, self.y_end = x, y
                cv2.rectangle(self.i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
        elif event == cv2.EVENT_LBUTTONUP:
            self.x_end, self.y_end = x, y
            self.cropping = False # cropping is finished
            self.refPoint = [(self.x_start, self.y_start), (self.x_end, self.y_end)]
            if len(self.refPoint) == 2: #when two points were found
                self.roi = self.imgOrg[self.refPoint[0][1]:self.refPoint[1][1], self.refPoint[0][0]:self.refPoint[1][0]]
            cv2.imshow("image", self.image)


a = cropImage(cv2.imread('images/bricks_0.jpg'))
cv2.imshow('img', a.getCropped())
cv2.waitKey()