import cv2 as cv
import numpy as np
import math

def testHoughLine():
    src = cv.imread(cv.samples.findFile("../selected/ZB_0087_02_sl.png"), cv.IMREAD_GRAYSCALE)


    # Edge detection
    dst = cv.Canny(src, 0, 0, None, 3)
    #lines = cv.HoughLines(dst, 1, np.pi / 180, 150)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # Draw the lines
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
    #         pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
    #         cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 5, None, 10, 20)
    print("Line Count: " + str(len(linesP)))

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    cv.imwrite('test.png', cdstP)

    #cv.imshow("Source", src)
    #cv.imshow("Destination", dst)

    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    #while(1):
    #    if cv.waitKey() == 27:
    #        cv.imwrite('test.png', cdstP)
    #        return 0

#img = cv.imread("../selected/ZB_0087_06_al.png")

#cv.imshow("Display window", img)
#k = cv.waitKey(0)

def testHarrisCorners():
    img = cv.imread(cv.samples.findFile("../selected/ZB_0087_07_os.png"))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray, 5,3,0.002)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[100,0,255]
    
    cv.imshow('dst', img)
    while(1):
        if cv.waitKey() == 27:
            cv.imwrite('test.png', img)
            return 0


def testHoughCircle():
    # Loads an image
    src = cv.imread(cv.samples.findFile("../selected/ZB_0087_02_sl.png"), cv.IMREAD_COLOR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #gray = cv.Canny(gray, 200, 20, None, 3) #ground plan
    gray = cv.medianBlur(gray, 5)

    #cv.imshow("Display window", gray)
    #k = cv.waitKey(0)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=30, param2=20, minRadius=200, maxRadius=3000)

    #gray = cv.medianBlur(gray, 5) #columns
    #circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=20, param2=10, minRadius=5, maxRadius=50)

    if circles is not None:

        circles = np.uint16(np.around(circles))
        print("Circle Count: " + str(len(circles[0])))

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 0, 0), 3)
            # circle outlines
            radius = i[2]
            cv.circle(src, center, radius, (0, 0, 255), 3)


    cv.imshow("Destination", src)

    while(1):
        if cv.waitKey() == 27:
            cv.imwrite('test.png', src)
            return 0

def testContours():
    im = cv.imread("../selected/ZB_0087_02_sl.png")
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(im, contours, -1, (0,255,0), 1)
    cv.imshow('dst', im)
    while(1):
        if cv.waitKey() == 27:
            cv.imwrite('test.png', im)
            return 0

if __name__ == "__main__":
    #print("OpenCV version:", cv.__version__)
    #testHoughLine()
    #testHoughCircle()
    #testHarrisCorners()
    testContours()