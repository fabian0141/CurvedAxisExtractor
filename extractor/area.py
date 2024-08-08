import cv2 as cv
import numpy as np
from extractor.vec import Vec2
from extractor.pointmath import PMath
from extractor.linewall import LineWall
from extractor.circlewall import CircleWall

class CircleArea:
    def __init__(self, circle, fullCircle):
        self.circle = circle
        self.columns = []
        self.circles = []
        self.lines = []
    
    def testColumns(self, columns):
        length = len(columns)
        i = 0
        while i < length:
            if self.circle.isInside(columns[i]):
                self.columns.append(columns[i])
                
            i += 1


    def drawColumns(self, dwg, thickness=1):
        for col in self.columns:
            dwg.add(dwg.circle(center=col.toArr(), r=thickness, fill="rgb(200,0,200)"))


    def findCurves(self, columns, walls):

        circles = []
        lines = []

        for col in columns:
            isLinePart = False
            isCirclePart = False

            for line in lines:
                isLinePart |= line.columnIsPart(col)

            for circle in circles:
                isCirclePart |= circle.columnIsPart(col)

            if not isLinePart:
                if not self.circle.isInside(col):
                    continue

                middle = self.circle.middle
                vec = col - middle
                d = col.dist(middle)
                d = self.circle.radius / d
                start = middle + vec * d
                
                #circles.append(CircleWall())
                lines.append(LineWall(col, start, self.circle.middle, LineWall.ADDED_WALL))

            if not isCirclePart:
                if not self.circle.isInside(col):
                    continue

                middle = self.circle.middle
                vec = col - middle
                d = col.dist(middle)
                d = self.circle.radius / d
                start = middle + vec * d
                
                #circles.append(CircleWall())
                circles.append(CircleWall(col, start, self.circle.middle, LineWall.ADDED_WALL))

        for circle in circles:
            circle.checkForIntersections(walls)

        for line in lines:
            line.checkForIntersections(walls)

        
        self.circles = circles
        self.lines = lines

    def drawArea(self, dwg, thickness = 1):
        for line in self.lines:
            line.drawWall(dwg, thickness)

        self.drawColumns(dwg, thickness)
        self.circle.drawOutline(dwg, thickness)

    def getCirclesAreas(img, columns, circles, fullCircle = False):
        areas = []
        for circle in circles:
            #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
            area = CircleArea(circle, fullCircle)
            area.testColumns(columns)
            #area.drawColumns(img, 3)
            areas.append(area)
        return areas

    def checkNeighboringCircleAreas(circleAreas, img):
        circleData = []

        for i in range(len(circleAreas)):
            c = circleAreas[i].circle
            if (c.fullCircle):
                continue

            ang = PMath.angle(c.start, c.end, c.middle)
            circleData.append([i, c.start, c.end, c.middle, ang])

        circlePairs = []


        for i in range(len(circleData)):
            for j in range(i+1, len(circleData)):

                # connect end with start
                if circleData[i][2] == circleData[j][1]:
                    ang = circleData[i][4] + circleData[j][4]
                    if ang < 0:
                        continue
                    ang -= PMath.angle(circleData[i][1], circleData[i][2], circleData[j][2])
                    circlePairs.append((i, j, ang))

                if circleData[j][2] == circleData[i][1]:
                    ang = circleData[i][4] + circleData[j][4]
                    if ang < 0:
                        continue

                    ang -= PMath.angle(circleData[j][1], circleData[j][2], circleData[i][2])
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

    def getWalls(self):
        walls = [
            CircleWall(self.circle, LineWall.GUIDE_WALL),
        ]
        if not self.circle.fullCircle:
            walls.append(LineWall(None, self.circle.start, self.circle.allignedMiddle, LineWall.GUIDE_WALL))
            walls.append(   LineWall(None, self.circle.end, self.circle.allignedMiddle, LineWall.GUIDE_WALL))
        return walls