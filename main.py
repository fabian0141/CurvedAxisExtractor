import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool

#imgFile = "../Dataset/Selected/ZB_0087_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0094_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0114_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0177_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0403_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0476_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0661_02_sl.png"
#imgFile = "../Dataset/Selected/ZB_0673_02_sl.png"




def testContours(imgFile, columnImg, out = "test.png"):
    img = cv.imread(imgFile)
    if img is None:
        return
    
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.drawContours(img, contours, -1, (0,255,0), 1)
    coumns = getColumnCenter(columnImg, img)

    for con in contours:
        miniCons = splitContours(con, img)
        points = findLines(miniCons, img)
        circles = findCircles(points, img)
        #findShapes(miniCons, img)

        #findLinesFromContour(con, img)
        #corners, innerCons = findCornersFromContour(con, img)
        #if corners == None:
        #    findBigCircleFromContour(innerCons, img)



        #for innerCon in innerCons:
        #    findLinesFromContour(innerCon, img)
            #findCirclesFromContour(innerCon, img)
        #findLinesFromContour(cons, im)
        


    #cv.imshow('dst', im)
    #while(1):
    #    if cv.waitKey() == 27:
    #        cv.imwrite('test.png', im)
    #        return 0

    cv.imwrite(out, img)

    if out == "test.png":
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', "test.png"))
        elif platform.system() == 'Windows':    # Windows
            os.startfile("test.png")
        else:                                   # linux variants
            subprocess.call(('xdg-open', "test.png"))

def vecLeng(vec):
    return np.sqrt(np.dot(vec, vec))

def vecDist(vec1, vec2):
    return vecLeng(vec1 - vec2)

def distancePointToLine(l1, l2, p):
    a = l2[1] - l1[1]
    b = l2[0] - l1[0]
    return abs(a * p[0] - b * p[1] + l2[0]*l1[1] - l2[1]*l1[0]) / np.sqrt(a*a + b*b)

def splitContours(con, img):
    miniCons = [con[0][0]]
    for i in range(0, len(con) - 10, 10):
        
        max = 0
        idx = None
        for j in range(10):
            dist = distancePointToLine(con[i][0], con[i+10][0], con[i+j][0])
            if dist > 1 and dist > max:
                #cv.circle(img, con[i+j][0], 0, (0, 0, 0), 1)
                idx = i+j
                max = dist

        if idx == None:
            miniCons.append(con[i+10][0])
            #cv.circle(img, con[i][0], 1, (255, 0, 0), 1)
            #cv.line(img, con[i][0], con[i+10][0], (150,150,150), 1)

        else:
            miniCons.append(con[idx][0])
            miniCons.append(con[i+10][0])
            #cv.circle(img, con[i][0], 1, (255, 0, 0), 1)
            #cv.line(img, con[i][0], con[idx][0], (150,150,150), 1)

            #cv.circle(img, con[idx][0], 1, (255, 0, 0), 1)
            #cv.line(img, con[idx][0], con[i+10][0], (150,150,150), 1)

    # TODO: handle end by finding corners inside
    miniCons.append(con[-1][0])

    return miniCons

def findLines(points, img):
    startPoint = points[0]
    lines = [startPoint]

    for i in range(1, len(points) - 1):
        betweenPoint = points[i]
        endPoint = points[i+1]
        dist = distancePointToLine( startPoint, endPoint, betweenPoint)
        if dist > 1:
            cv.line(img, startPoint, betweenPoint, (0,0,0), 1)
            cv.circle(img, betweenPoint, 3, (255, 100, 100), 2)
            startPoint = betweenPoint
            lines.append(betweenPoint)

    return lines

def findCircles(points, img):
    startPoint = points[0]
    middlePoint = None
    circles = []

    for i in range(1, len(points) - 3):
        if middlePoint == None:
            middlePoint, radius = getCircle(startPoint, points[i], points[i+1])
            if middlePoint == None:
                startPoint = points[i]
            elif abs(vecDist(middlePoint, points[i+2]) - radius) < 2 and abs(vecDist(middlePoint, points[i+3]) - radius) < 2:
                middlePoint, radius = getCircle(startPoint, points[i+1], points[i+3])
                #cv.circle(img, (int(middlePoint[0]), int(middlePoint[1])), int(radius), (255, 100, 100), 2)
                circles.append((middlePoint, radius))
                #print(middlePoint, radius)
            else:
                middlePoint = None
                startPoint = points[i]
        else:
            if abs(vecDist(middlePoint, points[i+3]) - radius) >= 2:
                middlePoint = None
                startPoint = points[i]
            else:
                middlePoint, radius = getCircle(startPoint, points[i+1], points[i+3])
                circles[-1] = (middlePoint, radius)

    for c, r in circles:
        cv.circle(img, (int(c[0]), int(c[1])), int(r), (255, 100, 100), 2)


    return circles


def getCircle(p1, p2, p3):
    a = [[2*p1[0], 2*p1[1], 1], \
         [2*p2[0], 2*p2[1], 1], \
         [2*p3[0], 2*p3[1], 1]]
    
    b = [-p1[0]*p1[0] - p1[1]*p1[1], \
         -p2[0]*p2[0] - p2[1]*p2[1], \
         -p3[0]*p3[0] - p3[1]*p3[1]]

    if np.linalg.det(a) == 0:
        return None, None

    x = np.linalg.solve(a, b)
    middlePoint = [-x[0], -x[1]]
    radius = np.sqrt(np.power(middlePoint[0], 2) + np.power(middlePoint[1], 2) - x[2])

    return middlePoint, radius 

def getColumnCenter(imgFile, img):
    columnImg = cv.imread(imgFile, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(columnImg, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours[1:]
    columns = []

    for con in contours:
        outline = con[:,0]
        center, radius = cv.minEnclosingCircle(outline)
        cv.circle(img, (int(center[0]), int(center[1])), int(radius - 3), (0, 0, 150), 3)
        columns.append(center)

    return columns

def test(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    #fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    testContours(fn_slab, "testOutput/" + str(num) + ".png")


if __name__ == "__main__":
    #print("OpenCV version:", cv.__version__)
    #testHoughLine()
    #testHoughCircle()
    #testHarrisCorners()

    nums = np.arange(1,1111)

    #with Pool(16) as p:
    #    p.map(test, nums)
    
    testContours("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    # testContours("../Dataset/Selected/ZB_0094_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0114_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0177_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0403_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0476_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0661_02_sl.png")
    # testContours("../Dataset/Selected/ZB_0673_02_sl.png")

