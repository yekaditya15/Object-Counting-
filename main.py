import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def show(title, img):
    cv2.imshow(title, img)
    cv2.imwrite(title + ".png", img)
    if cv2.waitKey(0) == ord('z'):
        cv2.destroyAllWindows()

class cropImage:
    def __init__(self, image):
        self.image = image
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.cropping = False
        self.imgOrg = image.copy()

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.mouse_crop)

        cv2.imshow("image", image)
        while True:
            self.i = image.copy()
            if cv2.waitKey(1) == ord('z'):
                cv2.destroyAllWindows()
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
                cv2.rectangle(self.i, (self.x_start, self.y_start),
                              (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.x_end, self.y_end = x, y
            self.cropping = False  # cropping is finished
            self.refPoint = [(self.x_start, self.y_start),
                             (self.x_end, self.y_end)]
            if len(self.refPoint) == 2:  # when two points were found
                self.roi = self.imgOrg[self.refPoint[0][1]:self.refPoint[1]
                                       [1], self.refPoint[0][0]:self.refPoint[1][0]]
            cv2.imshow("image", self.image)


def highContrast(img):
    imgHC = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHC = imgHC.astype(np.uint16)
    imgHC[:, :, 1] = imgHC[:, :, 1] * 4.5
    imgHC[:, :, 2] = imgHC[:, :, 2] * 1.5
    imgHC[imgHC > 255] = 255
    imgHC = imgHC.astype(np.uint8)
    imgHC = cv2.cvtColor(imgHC, cv2.COLOR_HSV2BGR)
    # show("highcontrast",imgHC)
    return imgHC


def kMeans(img, k=3):
    tmpClus = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmpClus = tmpClus.reshape((-1, 3))
    tmpClus = np.float32(tmpClus)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(tmpClus, k, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    # segmented_data = centers[labels.flatten()]
    # segmented_image = segmented_data.reshape((img.shape))
    # show("segmented", segmented_image)
    return labels, centers


def selectSegment(img, labels, centers, k=3):
    segments = labels.flatten()
    largestLabel = np.argmax(np.bincount(segments))
    centers = np.zeros((k, 3), dtype=np.uint8)
    centers[largestLabel] = [255, 255, 255]

    bw = np.zeros(segments.shape, dtype=np.uint8)
    bw = bw[segments[segments == 1]]
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    #show('segmented', segmented_image)
    return segmented_image


def contourMap(img, orig):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, hierarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    #show("contours", cv2.drawContours(img, contours, -1, (0, 0, 255), 2))
    cArea = [cv2.contourArea(contours[i]) for i in range(len(contours))]
    print(sorted(cArea))

    mean = np.mean(cArea, axis=0)
    sd = np.std(cArea, axis=0)
    final_list = [i for i, x in enumerate(cArea) if (x < mean - 2 * sd)][::-1]
    for i in final_list:
        contours = np.delete(contours, i)
    img = orig
    show("contoursRemov", cv2.drawContours(img, contours, -1, (70, 70, 0), 2))

    for p, i in enumerate(contours):
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(img, str(p+1), (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 100), 2)
        print(f"x: {cx} y: {cy}")
    
    cv2.putText(img, "Count : " + str(len(contours)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 0, 80), 2)
    show("contoursMarked", img)


def cannyEdge(img):
    #guassian blur
    blur = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    imgCanny = cv2.Canny(blur, 100, 200)
    #show("canny", imgCanny)
    dilation = cv2.dilate(imgCanny, kernel=np.ones((2, 2)), iterations=2)
    #show("cannydilated", dilation)
    dilation = cv2.bitwise_not(dilation)
    #show("cannyinverted", dilation)
    return dilation


def mergeEdge(imgBg, imgEdge):
    imgBg = cv2.GaussianBlur(imgBg, (3, 3), cv2.BORDER_DEFAULT)
    imgBg = cv2.erode(imgBg, kernel=np.ones((3, 3)), iterations=1)
    #show("guasErosion", imgBg)
    imgBg[imgEdge == 0] = [0, 0, 0]
    #show("merged", imgBg)
    return imgBg


def main(img, k = 3):
    # image with unnecessary details removed
    imgOrg = cv2.imread(img)
    #img0 = cropImage(imgOrg).getCropped()
    img0 = imgOrg.copy()
    #show('Cropped_Image', img0)

    # canny edge detection
    imgCan = cannyEdge(img0)

    # Enhance the contrast of the image
    imgHC = highContrast(img0)

    # kmean clustering
    labels, centers = kMeans(imgHC, k)

    # remove the segment with the least number of pixels
    imgSeg = selectSegment(imgHC, labels, centers, k)

    # merge the edge with the background
    imgMrg = mergeEdge(imgSeg, imgCan)

    # find contours
    contourMap(imgMrg, imgOrg)

if __name__ == '__main__':
    main('Cropped_Image.png')