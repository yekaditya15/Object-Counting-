import numpy as np
import cv2, glob

i = 0
images_folder = glob.glob("images/*.png")
while (i < len(images_folder) and i >= 0):
    img_path = images_folder[i]
    print("Img path", img_path)
    image = cv2.imread(img_path)

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #GrayScaling
    gray = cv2.GaussianBlur(gray, (3, 3), 0) #Blurring
    edges = cv2.Canny(gray, 50, 200) #Edge detection

    # Finding and sorting contours based on contour area
    cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:6]

    vertices = []
    for j, c in enumerate(cnts):
        if j == 0:
            # This is the largest contour
            # For overlapping case the largest one will be the only one contour
            peri = cv2.arcLength(cnts[j], True)
            approx = cv2.approxPolyDP(cnts[j], 0.02 * peri, True)
            vertices.append(approx)
        elif j < len(cnts) - 1:
            # Searches for any other inner contour
            # Also filters out close contours generated due to thick line
            if not np.isclose(cv2.contourArea(cnts[j]), cv2.contourArea(cnts[j+1]), atol=20000):
                peri = cv2.arcLength(cnts[j+1], True)
                approx = cv2.approxPolyDP(cnts[j+1], 0.02 * peri, True)
                vertices.append(approx)

    if len(vertices) == 1:
        # This case is where there is only one contour (the overlapping case)
        # There are eight extreme points for two overlapping rectangles
        # The distinct rectangles are colored in 'green' and 'red'
        extLeft1 = tuple(vertices[0][vertices[0][:, :, 0].argmin()][0])
        extRight1 = tuple(vertices[0][vertices[0][:, :, 0].argmax()][0])
        extTop1 = tuple(vertices[0][vertices[0][:, :, 1].argmin()][0])
        extBot1 = tuple(vertices[0][vertices[0][:, :, 1].argmax()][0])
        mask = np.isin(vertices[0][:, :, 1], (extRight1, extLeft1, extTop1, extBot1))
        indices = np.where(mask)
        vertices = np.delete(vertices[0], indices, 0)
        extLeft2 = tuple(vertices[vertices[:, :, 0].argmin()][0])
        extRight2 = tuple(vertices[vertices[:, :, 0].argmax()][0])
        extTop2 = tuple(vertices[vertices[:, :, 1].argmin()][0])
        extBot2 = tuple(vertices[vertices[:, :, 1].argmax()][0])

        x, y, w, h = cv2.boundingRect(np.array([extLeft1, extLeft2, extRight1, extRight2]))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([extTop1, extTop2, extBot1, extBot2]))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        # This case is where there are inner rectangle (the embedded case)
        # The distinct rectangles are colored in 'green' and 'red'
        x, y, w, h = cv2.boundingRect(vertices[0])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(vertices[1])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Displaying the images with identified colored rectangles
    cv2.imshow("Input", orig)
    cv2.imshow("Contour", image)


    if cv2.waitKey(0) == ord('x'):
        i += 1
    elif cv2.waitKey(0) == ord('z'):
        i -= 1
    elif cv2.waitKey(0) == ord('q'):
        break