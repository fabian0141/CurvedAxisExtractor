import cv2 as cv
import numpy as np
import math

def testHoughLine():
    src = cv.imread(cv.samples.findFile("../Dataset/Selected/ZB_0087_02_sl.png"), cv.IMREAD_GRAYSCALE)
    src2 = cv.imread(cv.samples.findFile("../Dataset/Selected/ZB_0087_02_sl.png"))


    # Edge detection
    dst = cv.Canny(src, 0, 0, None, 3)
    lines = cv.HoughLines(dst, 0.3, np.pi / 360, 50)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    #Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
            cv.line(src2, pt1, pt2, (0,0,255), 2)


    #linesP = cv.HoughLinesP(dst, 1, np.pi / 720, 70, None, 10, 20)
    #print("Line Count: " + str(len(linesP)))

    # if linesP is not None:
    #     col = 250
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(src2, (l[0], l[1]), (l[2], l[3]), (col,col, 0), 1)
    #         cv.circle(src2, (l[0], l[1]), 1, (155,0,155), 2)
    #         cv.circle(src2, (l[2], l[3]), 1, (155,0,155), 2)
    #         col = (col+50)%250

    cv.imwrite('test.png', src2)

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
    print("Test")
    img = cv.imread(cv.samples.findFile("../Dataset/Selected/Test.png"))
    #img = cv.imread(cv.samples.findFile("../Dataset/Selected/ZB_0476_07_os.png"))
    #img = cv.imread(cv.samples.findFile("../Dataset/07_os/ZB_0008_07_os.png"))


    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray, 2,3,0.005)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.0265*dst.max()]=[100,0,255]
    cv.imwrite('test.png', img)


def testHoughCircle():
    # Loads an image
    src = cv.imread(cv.samples.findFile("../Dataset/Selected/ZB_0476_02_sl.png"), cv.IMREAD_COLOR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #gray = cv.Canny(gray, 200, 20, None, 3) #ground plan
    gray = cv.medianBlur(gray, 5)

    #cv.imshow("Display window", gray)
    #k = cv.waitKey(0)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 300, param1=200, param2=30, minRadius=100, maxRadius=3000)

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



    cv.imwrite('test.png', src)


def findCirclesFromContour(contour, img):

    if vecDist(contour[0][0], contour[-1][0]) < 20:
        return None, None
    
    vecDist(contour[0][0], contour[-1][0])

    idxs = [0, len(contour) // 2, -1]

    a = [0] * 3
    b = [0] * 3
    for i in range(3):
        idx = idxs[i]
        a[i] = [2 * contour[idx][0][0], 2 * contour[idx][0][1], 1]
        b[i] = -(np.power(contour[idx][0][0], 2) + np.power(contour[idx][0][1], 2))

    if np.linalg.det(a) == 0:
        return None, None

    x = np.linalg.solve(a, b)
    middlePoint = [-x[0], -x[1]]
    radius = np.sqrt(np.power(middlePoint[0], 2) + np.power(middlePoint[1], 2) - x[2])

    if radius < 3 * vecDist(contour[0][0], contour[-1][0]):
        cv.circle(img, [int(middlePoint[0]), int(middlePoint[1])], int(radius), (0, 0, 0), 1)
        cv.circle(img, [int(middlePoint[0]), int(middlePoint[1])], 1, (0, 0, 255), 3)
        return middlePoint, radius 

    return None, None

def findBigCircleFromContour(contour, img):

    
    vecDist(contour[0][0], contour[-1][0])

    idxs = [0, len(contour) // 3, 2 * len(contour) // 3]

    a = [0] * 3
    b = [0] * 3
    for i in range(3):
        idx = idxs[i]
        a[i] = [2 * contour[idx][0][0], 2 * contour[idx][0][1], 1]
        b[i] = -(np.power(contour[idx][0][0], 2) + np.power(contour[idx][0][1], 2))

    if np.linalg.det(a) == 0:
        return None, None

    x = np.linalg.solve(a, b)
    middlePoint = [-x[0], -x[1]]
    radius = np.sqrt(np.power(middlePoint[0], 2) + np.power(middlePoint[1], 2) - x[2])

    if radius < 3 * vecDist(contour[0][0], contour[idxs[1]][0]):
        cv.circle(img, [int(middlePoint[0]), int(middlePoint[1])], int(radius), (0, 0, 0), 1)
        cv.circle(img, [int(middlePoint[0]), int(middlePoint[1])], 1, (0, 0, 255), 3)
        return middlePoint, radius 

    return None, None

def findCornersFromContour(contour, img):

    lastCorner = 0
    lastAng = 0

    corners = []
    splitContours = []
    firstCorner = 0

    for i in range(len(contour)):
        
        idxs = [i, (i + 5) % len(contour), (i + 10) % len(contour)]

        vec1 = contour[idxs[0]][0] - contour[idxs[1]][0]
        vec2 = contour[idxs[1]][0] - contour[idxs[2]][0]
        ang = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        ang = np.rad2deg(ang)
        if ang > 30:
            if vecDist(contour[lastCorner][0], contour[idxs[1]][0]) < 10:
                if lastAng < ang:
                    lastCorner = idxs[1]
                    lastAng = ang
            else:
                if lastAng > 0:
                    if len(corners) == 0:
                        firstCorner = lastCorner

                    corners.append(contour[lastCorner][0])
                    cv.circle(img, contour[lastCorner][0], 3, (255, 0, 0), 2)
                    if lastCorner < idxs[1]:
                        splitContours.append(contour[lastCorner:idxs[1]])
                    else:
                        splitContours.append(np.concatenate([contour[lastCorner:], contour[:idxs[1]]]))

                lastCorner = idxs[1]
                lastAng = ang

    if lastAng > 0:
        corners.append(contour[lastCorner][0])
        cv.circle(img, contour[lastCorner][0], 3, (255, 0, 0), 2)
        if lastCorner < idxs[1]:
            splitContours.append(contour[lastCorner:idxs[1]])
        else:
            splitContours.append(np.concatenate([contour[lastCorner:], contour[:firstCorner]]))

    if len(corners) == 0:
        return None, contour

    return corners, splitContours

def findLinesFromContour(cons, img):


    #sum = [0, 0]

    #for i in range(len(cons)):
    #    sum += cons[i][0]

    #angle = np.arcsin(sum[0] / np.sqrt(np.dot(sum, sum)))



    angle = None
    distance = 0
    startPoint = cons[0][0]

    for i in range(1, len(cons)):
        vec1 = cons[i][0]

        vecD = startPoint - vec1

        newAngle = np.arcsin(vecD[0] / np.sqrt(np.dot(vecD, vecD)))

        if angle != None:
            if abs(newAngle - angle) > 0.1:
                cv.circle(img, vec1, 2, (255, 0, 0), 1)
                angle = newAngle

        else:
            angle = newAngle


if __name__ == "__main__":
    testHarrisCorners()