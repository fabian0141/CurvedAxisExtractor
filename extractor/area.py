import cv2 as cv
import numpy as np
from extractor.vec import Vec2
from extractor.helper import angle

class CircleArea:
    def __init__(self, circle, fullCircle):
        self.circle = circle
        self.columns = []
        self.curvedWalls = []
        self.curves = []
    
    def testColumns(self, columns):
        length = len(columns)
        i = 0
        while i < length:
            if self.circle.isInside(columns[i]):
                self.columns.append(columns[i])
                
            i += 1


    def drawColumns(self, img, thickness=1):
        for col in self.columns:
            cv.circle(img, col.toIntArr(), 5, (200, 0, 200), thickness)

    def findCurves(self, img, thickness):
        distCol = []
        for col in self.columns:
            dist = col.dist(self.circle.middle)
            addNew = True
            for i in range(len(distCol)):
                if abs(distCol[i][0] - dist) < 10:
                    distCol[i] = ((distCol[i][0] + dist) / 2, distCol[i][1] + 1)
                    addNew = False
                    break
                
            if addNew:
                distCol.append((dist, 1))

        for d, c in distCol:
            if c > 1:
                self.curves.append(d)

    def drawArea(self, img, thickness = 1):
        for cur in self.curves:
            self.circle.drawCircleCurve(img, cur, thickness)

        self.circle.drawSecondLines(img, self.columns, thickness)
        self.circle.drawOutline(img, thickness)

    def linesIntersection(p1, p2, q1, q2):
        v1 = p1 - p2
        v2 = q1 - q2
        v3 = p1 - q1
        denom = v1.x * v2.y - v1.y * v2.x

        t = (v3.x * v2.y - v3.y * v2.x) / denom
        u = -(v1.x * v3.y - v1.y * v3.x) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 - v1*t

        return None

    def getCirclesAreas(img, columns, circles, fullCircle = False):
        areas = []
        for circle in circles:
            #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
            area = CircleArea(circle, fullCircle)
            area.testColumns(columns)
            #area.drawColumns(img, 3)
            area.findCurves(img, 3)
            areas.append(area)
        return areas

    def checkNeighboringCircleAreas(circleAreas, img):
        circleData = []

        for i in range(len(circleAreas)):
            c = circleAreas[i].circle
            if (c.fullCircle):
                continue

            ang = angle(c.start, c.end, c.middle)
            circleData.append([i, c.start, c.end, c.middle, ang])

        circlePairs = []


        for i in range(len(circleData)):
            for j in range(i+1, len(circleData)):

                # connect end with start
                if circleData[i][2] == circleData[j][1]:
                    ang = circleData[i][4] + circleData[j][4]
                    if ang < 0:
                        continue
                    ang -= angle(circleData[i][1], circleData[i][2], circleData[j][2])
                    circlePairs.append((i, j, ang))

                if circleData[j][2] == circleData[i][1]:
                    ang = circleData[i][4] + circleData[j][4]
                    if ang < 0:
                        continue

                    ang -= angle(circleData[j][1], circleData[j][2], circleData[i][2])
                    circlePairs.append((j, i, ang))

        circlePairs = sorted(circlePairs, key=lambda x: x[2])

        while len(circlePairs) > 0:
            val = circlePairs[0][2] / 2
            n1 = circlePairs[0][0]
            n2 = circlePairs[0][1]
            circleAreas[n1].circle.allignMiddle(val)
            circleAreas[n2].circle.allignMiddle(val)
            circlePairs.pop(0)
            CircleArea.checkPairs(circleAreas, img, circlePairs, val, n1, n2)


    def checkPairs(circleAreas, img, circlePairs, val, n1, n2=None):
        
        i = 0
        while i < len(circlePairs):

            if circlePairs[i][0] == n1 or circlePairs[i][0] == n2:
                val = circlePairs[i][2] - val
                n1 = circlePairs[i][1]
                circlePairs.pop(i)
                circleAreas[n1].circle.allignMiddle(val)
                CircleArea.checkPairs(circleAreas, img, circlePairs, val, n1)
                continue

            if circlePairs[i][1] == n1 or circlePairs[i][1] == n2:
                val = circlePairs[i][2] - val
                n1 = circlePairs[i][0]
                circleAreas[n1].circle.allignMiddle(val)
                circlePairs.pop(i)
                CircleArea.checkPairs(circleAreas, img, circlePairs, val, n1)
                continue

            i += 1

class PolygonArea:
    def __init__(self, points):
        self.points = points
