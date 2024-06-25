import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool
from extractor.vec import Vec2
from extractor.helper import distancePointToLine, angle
from extractor.area import CircleArea

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

    contours = Vec2.convertContour(contours)

    columns = getColumnCenter(columnImg, img)

    for points in contours:
        findCornersFromContour(points, img)
        #miniCons = splitContours(points, img)
        #parts = splitIntoParts(img, points, miniCons)
        #for part in parts:
        #    lines = findLines(part, img)
        #    circles = findCircles(part, lines, img, columns)

    for col in columns:
        cv.circle(img, col.toIntArr(), 3, (0, 0, 150), 3)

    cv.imwrite(out, img)

    if out == "test.png":
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', "test.png"))
        elif platform.system() == 'Windows':    # Windows
            os.startfile("test.png")
        else:                                   # linux variants
            subprocess.call(('xdg-open', "test.png"))


def findCornersFromContour(contour, img):

    lastAng = 0

    corners = []
    splitContours = []
    step = 5
    fullStep = step * 2
    firstCorner = -fullStep
    lastCorner = -fullStep


    for i in range(-fullStep, len(contour) - fullStep):
        
        idxs = [i, (i + step) % len(contour), (i + fullStep) % len(contour)]
        ang = angle(contour[i], contour[i+step], contour[i+fullStep])

        if ang > 30:
            if contour[lastCorner].dist(contour[idxs[1]]) < fullStep:
                if lastAng < ang:
                    lastCorner = idxs[1]
                    lastAng = ang
            else:
                if lastAng > 0:
                    if len(corners) == 0:
                        firstCorner = lastCorner

                    corners.append(contour[lastCorner])
                    cv.circle(img, contour[lastCorner].toIntArr(), 3, (255, 0, 0), 2)
                    if lastCorner < idxs[1]:
                        splitContours.append(contour[lastCorner:idxs[1]])
                    else:
                        splitContours.append(np.concatenate([contour[lastCorner:], contour[:idxs[1]]]))

                lastCorner = idxs[1]
                lastAng = ang

    if lastAng > 0:
        corners.append(contour[lastCorner])
        cv.circle(img, contour[lastCorner].toIntArr(), 3, (255, 0, 0), 2)
        if lastCorner < idxs[1]:
            splitContours.append(contour[lastCorner:idxs[1]])
        else:
            splitContours.append(np.concatenate([contour[lastCorner:], contour[:firstCorner]]))

    if len(corners) == 0:
        return None, contour

    print(len(corners))
    return corners, splitContours

def splitContours(con, img):
    miniCons = [0]
    for i in range(0, len(con) - 10, 10):
        
        max = 0
        idx = None
        for j in range(10):
            dist = distancePointToLine(con[i], con[i+10], con[i+j])
            if dist > 1 and dist > max:
                cv.circle(img, con[i+j].toIntArr(), 0, (0, 0, 0), 1)
                idx = i+j
                max = dist

        if idx == None:
            miniCons.append(i+10)
            cv.circle(img, con[i].toIntArr(), 1, (255, 0, 0), 1)
            cv.line(img, con[i].toIntArr(), con[i+10].toIntArr(), (150,150,150), 1)

        else:
            miniCons.append(idx)
            miniCons.append(i+10)
            cv.circle(img, con[i].toIntArr(), 1, (255, 0, 0), 1)
            cv.line(img, con[i].toIntArr(), con[idx].toIntArr(), (150,150,150), 1)

            cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
            cv.line(img, con[idx].toIntArr(), con[i+10].toIntArr(), (150,150,150), 1)

    # TODO: handle end by finding corners inside
    miniCons.append(0)

    return miniCons

def splitIntoParts(img, points, con):
    parts = []
    startIdx = 0
    for i in range(0, len(con) - 2):
        if angle(points[con[i]], points[con[i+1]], points[con[i+2]]) > 45:
            parts.append(points[startIdx:i+1])
            #cv.line(img, points[con[startIdx]].toArr(), points[con[i+1]].toArr(), (100,0,0), 1)
            #cv.circle(img, points[con[i+1]].toArr(), 3, (0, 200, 200), 2)

            startIdx = i+1

    print(len(parts))


def findLines(points, img):
    startPoint = points[0]
    lines = [startPoint]

    for i in range(1, len(points) - 1):
        betweenPoint = points[i]
        endPoint = points[i+1]
        dist = distancePointToLine( startPoint, endPoint, betweenPoint)
        if dist > 1:
            cv.line(img, startPoint.toArr(), betweenPoint.toArr(), (0,0,0), 1)
            cv.circle(img, betweenPoint.toArr(), 3, (255, 100, 100), 2)
            startPoint = betweenPoint
            lines.append(betweenPoint)

    cv.line(img, startPoint.toArr(), endPoint.toArr(), (0,0,0), 1)
    cv.circle(img, endPoint.toArr(), 3, (255, 100, 100), 2)
    lines.append(endPoint)

    return lines

def findCircles(points, img, columns):
    startPoint = points[0]
    middlePoint = None
    circles = []
    tresh = 5
    startIdx = 0
    firstCircleIdx = None
    lastCircleIdx = None

    #startEndDist = 0

    for i in range(1, len(points) - 3):
        if middlePoint is None:
            middlePoint, radius = getCircle(startPoint, points[i+1], points[i+3])
            if middlePoint is None:
                startPoint = points[i]
                startIdx = i
                #startEndDist = 0
            elif areBetweenPointsInside(middlePoint, radius, points[i:i+3]):
                circles.append((startPoint, points[i+1], points[i+3], middlePoint, radius))
                if firstCircleIdx == None:
                    firstCircleIdx = startIdx
                #startEndDist = middlePoint.dist(points[i+3])
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = points[i]
                startIdx = i
                #startEndDist = 0
        else:
            pointDist1 = startPoint.dist(points[(startIdx+i+3) // 2])
            pointDist2 = startPoint.dist(points[i+3])

            if pointDist1 > pointDist2:
                middlePoint, radius = getCircle(startPoint, points[(startIdx+i+3) // 6], points[(startIdx+i+3) // 3])
            else:
                middlePoint, radius = getCircle(startPoint, points[(startIdx+i+3) // 2], points[i+3])

            # TODO: better check for if radius is smaller
            if middlePoint is None:
                circles = circles[:-1]
                continue

            if areBetweenPointsInside(middlePoint, radius, points[startIdx+1:i+3]):
                circles[-1] = (startPoint, points[(startIdx+i+3) // 2], points[i+3], middlePoint, radius)
                #startEndDist = pointDist
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = points[i]
                startIdx = i
                #startEndDist = 0

    if len(circles) > 0:
        #combine two circles around end
        if len(circles) > 1:
            if circles[0][0] == circles[-1][2] and circles[0][4] - circles[-1][4]:
                middlePoint, radius = getCircle(circles[-1][0], circles[0][0], circles[0][2])

                circles[0] = (circles[-1][0], circles[0][0], circles[0][2], middlePoint, radius)
                circles = circles[:-1]
                return getCirclesAreas(img, columns, circles)

        #check for bigger circle at start
        if circles[0][0] == points[0]:
            endPoint = circles[0][2]

            for i in range(len(points)-1, lastCircleIdx, -1):
                pointDist1 = startPoint.dist(points[(startIdx+i) // 2])
                pointDist2 = startPoint.dist(points[i])

                if pointDist1 > pointDist2:
                    middlePoint, radius = getCircle(startPoint, points[(startIdx+i) // 6], points[(startIdx+i) // 3])
                else:
                    middlePoint, radius = getCircle(startPoint, points[(startIdx+i) // 2], points[i])

                # TODO: better check for if radius is smaller
                if middlePoint is None:
                    circles = circles[:-1]
                    continue

                if areBetweenPointsInside(middlePoint, radius, points[startIdx+1:i]):
                    circles[-1] = (startPoint, points[(startIdx+i) // 2], points[i], middlePoint, radius)

                else:
                    break

        #check for bigger circle at end
        elif circles[-1][2] == points[-1]:
            startPoint = circles[-1][0]

            for i in range(1, firstCircleIdx):
                pointDist1 = startPoint.dist(points[(startIdx+i) // 2])
                pointDist2 = startPoint.dist(points[i])

                if pointDist1 > pointDist2:
                    middlePoint, radius = getCircle(startPoint, points[(startIdx+i) // 6], points[(startIdx+i) // 3])
                else:
                    middlePoint, radius = getCircle(startPoint, points[(startIdx+i) // 2], points[i])

                # TODO: better check for if radius is smaller
                if middlePoint is None:
                    circles = circles[:-1]
                    continue

                if areBetweenPointsInside(middlePoint, radius, points[startIdx+1:i]):
                    circles[-1] = (startPoint, points[(startIdx+i) // 2], points[i], middlePoint, radius)

                else:
                    break

    # for i in range(1, len(points) - 3):
    #     if middlePoint == None:
    #         middlePoint, radius = getCircle(startPoint, points[i], points[i+1])
    #         if middlePoint == None:
    #             startPoint = points[i]
    #             startIdx = i
    #         elif abs(middlePoint.dist(points[i+2]) - radius) < tresh and abs(middlePoint.dist(points[i+3]) - radius) < tresh:
    #             middlePoint, newRadius = getCircle(startPoint, points[i+1], points[i+3])
    #             if middlePoint == None:
    #                 circles = circles[:-1]
    #                 continue
    #             #cv.circle(img, (int(middlePoint[0]), int(middlePoint[1])), int(radius), (255, 100, 100), 2)
    #             circles.append((startPoint, points[i+1], points[i+3], middlePoint, newRadius))
    #             #print(middlePoint, radius)
    #         else:
    #             middlePoint = None
    #             startPoint = points[i]
    #             startIdx = i
    #     else:
    #         dist = middlePoint.dist(points[i+3])
    #         if abs(dist - newRadius) >= tresh:
    #             middlePoint = None
    #             startPoint = points[i]
    #             startIdx = i
    #         else:
    #             if startPoint.dist(points[i+3]) < 100:
    #                 middlePoint, newRadius = getCircle(startPoint, points[(startIdx+i+3) // 3], points[(2 * (startIdx+i+3)) // 3])
    #             else:
    #                 middlePoint, newRadius = getCircle(startPoint, points[(startIdx+i+3) // 2], points[i+3])

    #             if middlePoint == None:
    #                 circles = circles[:-1]
    #                 continue
    #             circles[-1] = (startPoint, points[(startIdx+i+3) // 2], points[i+3], middlePoint, newRadius)

def getCirclesAreas(img, columns, circles):
    for p1, p2, p3, c, r in circles:
        #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
        area = CircleArea(p1, p2, p3, c, r)
        area.drawOutline(img, 3)
        area.testColumns(columns)
        area.drawColumns(img, 3)
        area.findCurves(img, 3)

    return circles

def areBetweenPointsInside(middle, radius, between):
    for i in range(len(between)):
        dist = middle.dist(between[i])
        if abs(dist - radius) > 5:
            return False
    
    return True



def getCircle(p1, p2, p3):
    a = [[2*p1.x, 2*p1.y, 1], \
         [2*p2.x, 2*p2.y, 1], \
         [2*p3.x, 2*p3.y, 1]]
    
    b = [-p1.x**2 - p1.y**2, \
         -p2.x**2 - p2.y**2, \
         -p3.x**2 - p3.y**2]

    if np.linalg.det(a) == 0:
        return None, None

    x = np.linalg.solve(a, b)
    middlePoint = Vec2([-x[0], -x[1]])
    radius = np.sqrt(np.power(middlePoint.x, 2) + np.power(middlePoint.y, 2) - x[2])
    if radius < 100:
        return None, None

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
        center = Vec2(center)
        #cv.circle(img, center.toIntArr(), int(radius - 3), (0, 0, 150), 3)
        columns.append(center)

    return columns

def test(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    #fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    testContours(fn_slab, "testOutput/" + str(num) + ".png")

def selectedTest(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    testContours(fn_slab, fn_column, "selectedOutput/" + str(num) + ".png")

if __name__ == "__main__":
    #print("OpenCV version:", cv.__version__)
    #testHoughLine()
    #testHoughCircle()
    #testHarrisCorners()

    #nums = np.arange(1,1111)

    #with Pool(16) as p:
    #    p.map(test, nums)
    
    #nums = [87, 94, 114, 177, 403, 476, 661, 673]

    #with Pool(8) as p:
    #    p.map(selectedTest, nums)

    testContours("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    #testContours("../Dataset/Selected/ZB_0094_02_sl.png", "../Dataset/Selected/ZB_0094_03_co.png")
    # testContours("../Dataset/Selected/ZB_0114_02_sl.png", "../Dataset/Selected/ZB_0114_03_co.png")
    # testContours("../Dataset/Selected/ZB_0177_02_sl.png")
    #testContours("../Dataset/Selected/ZB_0403_02_sl.png", "../Dataset/Selected/ZB_0403_03_co.png")
    # testContours("../Dataset/Selected/ZB_0476_02_sl.png", "../Dataset/Selected/ZB_0476_03_co.png")
    # testContours("../Dataset/Selected/ZB_0661_02_sl.png", "../Dataset/Selected/ZB_0661_03_co.png")
    # testContours("../Dataset/Selected/ZB_0673_02_sl.png")

