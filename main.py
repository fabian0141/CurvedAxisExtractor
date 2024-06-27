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
    circles = []

    for points in contours:
        #findCornersFromContour(points, img)
        miniCons = splitContours(points, img)
        corners = findLines(points, miniCons, img)
        parts = splitIntoParts(img, points, corners)
        for part in parts:
           circles.extend(findCircles(part, img, columns))

    #checkNeighbourCircles(circles)

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
    miniCons = []
    i = 0
    leng = len(con)
    firstIdx = None
    count = 0

    while True:
        max = 0

        #i10 = (i + 10) % leng 
        if firstIdx != None:
            outer = np.min([firstIdx - i, 20])
        else: 
            outer = 20

        i20 = (i + outer) % leng 
        idx = (i+10) % leng



        for j in range(outer):
            ij = (i+j) % leng
            dist = distancePointToLine(con[i], con[i20], con[ij])
            if dist > 1 and dist > max:
                #cv.circle(img, con[ij].toIntArr(), 0, (0, 0, 0), 1)
                idx = ij
                max = dist

        if outer < 20 and max == 0:
            break

        if idx >= i:
            count += idx - i
        else:
            count += idx + leng - i

        if firstIdx != None and count >= firstIdx:
            break



        miniCons.append(idx)
        #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
        #cv.line(img, con[i].toIntArr(), con[idx].toIntArr(), (150,150,150), 1)



        i = idx


        if firstIdx == None:
            firstIdx = leng + i
           #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
           #cv.line(img, con[idx].toIntArr(), con[i+10].toIntArr(), (150,150,150), 1)

    # TODO: handle end by finding corners inside

    return miniCons

def splitIntoParts(img, points, corners):
    parts = []
    startIdx = None
    startCornerIdx = None
    firstCorner = None
    firstCornerIdx = None

    leng = len(corners)
    for i in range(0, leng):
        i1 = (i+1) % leng
        i2 = (i+2) % leng
        if angle(corners[i][0], corners[i1][0], corners[i2][0]) > 30:
            if firstCorner is None:
                firstCorner = corners[i1]
                firstCornerIdx = i1
            else:
                if startCornerIdx < i2:
                    corns = corners[startCornerIdx:i2]
                else:
                    corns = corners[startCornerIdx:] + corners[:i2]

                if startIdx < corners[i1][1]:
                    parts.append((points[startIdx:corners[i1][1]+1], corns))
                else:
                    parts.append((points[startIdx:] + points[:corners[i1][1]+1], corns))

                cv.line(img, points[startIdx].toIntArr(), points[corners[i1][1]].toIntArr(), (150,0,0), 1)
                cv.circle(img, corners[i1][0].toIntArr(), 5, (0, 100, 100), 3)

            startIdx = corners[i1][1]
            startCornerIdx = i1

    if startCornerIdx is None:
        return [(points, corners)]

    if startCornerIdx < firstCornerIdx:
        corns = corners[startCornerIdx:firstCornerIdx+1]
    else:
        corns = corners[startCornerIdx:] + corners[:firstCornerIdx+1]

    if startIdx < firstCorner[1]:
        parts.append((points[startIdx:firstCorner[1]+1], corns))
    else:
        parts.append((points[startIdx:] + points[:firstCorner[1]+1], corns))



    cv.line(img, points[startIdx].toIntArr(), firstCorner[0].toIntArr(), (150,0,0), 1)
    cv.circle(img, firstCorner[0].toIntArr(), 5, (0, 100, 100), 3)

    pointLeng = len(points)

    #update indexes of corners
    for part in parts:
        startIdx = part[1][0][1]
        
        for i in range(len(part[1])):
            part[1][i] = (part[1][i][0], np.mod(part[1][i][1] - startIdx, pointLeng))



    return parts

def intersectionPoint(p1, p2, q1, q2):

    v1 = p1 - p2
    v2 = q1 - q2
    v3 = p1 - q1

    denom = v1.x * v2.y - v1.y * v2.x

    t = (v3.x * v2.y - v3.y * v2.x) / denom
    #u = -(v1.x * v3.y - v1.x * v3.x) / denom

    return p1 - v1*t

def findLines(points, con, img):
    startPoint = points[con[0]]
    lines = [(startPoint, con[0])]

    for i in range(1, len(con) - 1):
        betweenPoint = points[con[i]]
        endPoint = points[con[i+1]]
        dist = distancePointToLine( startPoint, endPoint, betweenPoint)
        if dist > 1:
            #cv.line(img, startPoint.toArr(), betweenPoint.toArr(), (0,0,0), 1)
            #cv.circle(img, betweenPoint.toArr(), 3, (255, 100, 100), 2)
            startPoint = betweenPoint
            lines.append((betweenPoint, con[i]))

    #cv.line(img, startPoint.toIntArr(), endPoint.toIntArr(), (0,0,0), 1)
    #cv.circle(img, endPoint.toIntArr(), 3, (255, 100, 100), 2)
    #lines.append(endPoint)
    lines.append((endPoint, con[i+1]))

    leng = len(lines)
    # remove rounded corners
    for idx in range(leng):
        i1 = idx % leng
        i2 = (idx + 1) % leng
        j1 = (idx + 2) % leng
        j2 = (idx + 3) % leng
        if lines[i2][0].dist(lines[j1][0]) < 3:
            intersection = intersectionPoint(lines[i1][0], lines[i2][0], lines[j1][0], lines[j2][0])
            #cv.circle(img, intersection.toIntArr(), 3, (155, 50, 0), 2)
            lines[i2] = (intersection, (lines[i2][1] + lines[j1][1]) // 2)
            lines.pop(j1)
            leng -= 1

    for i in range(-1, len(lines)-1):
        #cv.line(img, lines[i].toIntArr(), lines[i+1].toIntArr(), (0,0,0), 1)
        cv.circle(img, lines[i+1][0].toIntArr(), 3, (255, 100, 100), 2)


    return lines

def findCircles(part, img, columns):
    startPoint = part[1][0]
    middlePoint = None
    circles = []
    tresh = 5
    startIdx = 0
    firstCircleIdx = None
    lastCircleIdx = None

    #startEndDist = 0

    for i in range(1, len(part[1]) - 3):
        if middlePoint is None:
            middlePoint, radius = getCircle(startPoint, part[1][i+1], part[1][i+3])
            if middlePoint is None:
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0
            elif areBetweenPointsInside(middlePoint, radius, part[1][i:i+3]) and isContourInside(middlePoint, radius, part, startPoint[1], part[1][i+3][1]):
                circles.append((startPoint[0], part[1][i+1][0], part[1][i+3][0], middlePoint, radius))
                if firstCircleIdx == None:
                    firstCircleIdx = startIdx
                #startEndDist = middlePoint.dist(part[1][i+3])
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0
        else:
            pointDist1 = startPoint[0].dist(part[1][(startIdx+i+3) // 2][0])
            pointDist2 = startPoint[0].dist(part[1][i+3][0])

            if pointDist1 > pointDist2 * 1.5:
                middlePoint, radius = getCircle(startPoint, part[1][(startIdx+i+3) // 6], part[1][(startIdx+i+3) // 3])
            else:
                middlePoint, radius = getCircle(startPoint, part[1][(startIdx+i+3) // 2], part[1][i+3])

            # TODO: better check for if radius is smaller
            if middlePoint is None:
                circles = circles[:-1]
                continue

            if areBetweenPointsInside(middlePoint, radius, part[1][startIdx+1:i+3]) and isContourInside(middlePoint, radius, part, startPoint[1], part[1][i+3][1]):
                circles[-1] = (startPoint[0], part[1][(startIdx+i+3) // 2][0], part[1][i+3][0], middlePoint, radius)
                #startEndDist = pointDist
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0

    if len(circles) > 0:
        print(len(circles))
        #better way to prove
        if part[1][0][0].dist(part[1][-1][0]) < 30 and circles[0][0] == part[1][0][0] and circles[0][2] == part[1][-1][0]:
            return getCirclesAreas(img, columns, circles, True)

        return getCirclesAreas(img, columns, circles)

    return []

def getCirclesAreas(img, columns, circles, fullCircle = False):
    areas = []
    for p1, p2, p3, c, r in circles:
        #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
        area = CircleArea(p1, p2, p3, c, r, fullCircle)
        area.drawOutline(img, 3)
        area.testColumns(columns)
        area.drawColumns(img, 3)
        area.findCurves(img, 3)
        areas.append(area)

    return areas

def areBetweenPointsInside(middle, radius, between):
    between = [b[0] for b in between]

    for i in range(len(between)):
        dist = middle.dist(between[i])
        if abs(dist - radius) > 5:
            return False
    
    return True

def isContourInside(middlePoint, radius, part, startIdx, endIdx):
    contour = part[0][startIdx:endIdx]
    for p in contour:
        d = abs(middlePoint.dist(p) - radius) 
        if d > 3:
            return False
        
    return True

def checkNeighbourCircles(circles):
    neighbourCircles = []

    for i in range(len(circles)):
        for j in range(len(circles)):
            if i == j:
                continue

            if circles[i].endCorner == circles[i].startCorner:
                neighbourCircles.append


def getCircle(p1, p2, p3):
    p1 = p1[0]
    p2 = p2[0]
    p3 = p3[0]

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

    print("Finshed: " + str(num))

if __name__ == "__main__":
    #print("OpenCV version:", cv.__version__)
    #testHoughLine()
    #testHoughCircle()
    #testHarrisCorners()

    #nums = np.arange(1,1111)

    #with Pool(16) as p:
    #    p.map(test, nums)
    
    nums = [87, 94, 114, 177, 403, 476, 661, 673]

    with Pool(8) as p:
        p.map(selectedTest, nums)
    
    # 94 117 
    #testContours("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    #testContours("../Dataset/Selected/ZB_0094_02_sl.png", "../Dataset/Selected/ZB_0094_03_co.png")
    # testContours("../Dataset/Selected/ZB_0114_02_sl.png", "../Dataset/Selected/ZB_0114_03_co.png")
    # testContours("../Dataset/Selected/ZB_0177_02_sl.png")
    #testContours("../Dataset/Selected/ZB_0403_02_sl.png", "../Dataset/Selected/ZB_0403_03_co.png")
    #testContours("../Dataset/Selected/ZB_0476_02_sl.png", "../Dataset/Selected/ZB_0476_03_co.png")
    #testContours("../Dataset/Selected/ZB_0661_02_sl.png", "../Dataset/Selected/ZB_0661_03_co.png")
    # testContours("../Dataset/Selected/ZB_0673_02_sl.png")

