import cv2 as cv
import numpy as np
from extractor.vec import Vec2
from extractor.forms import Circle

class CircleArea:
    def __init__(self, circle, fullCircle):
        self.circle = circle
        self.columns = []
        self.curvedWalls = []
    
    def testColumns(self, columns):
        length = len(columns)
        i = 0
        while i < length:
            if self.isInside(columns[i]):
                self.columns.append(columns.pop(i))
                length -= 1
                continue
                
            i += 1

    def isInside(self, p):
        dist = p.dist(self.circle.middle)
        if dist > self.circle.radius:
            return False    

        if self.circle.fullCircle:
            return True

        angle = Circle.getCircleAngle(p, self.circle.middle)
        angle += (2*np.pi) if self.circle.overflow and self.circle.startAngle > angle else 0
        if self.circle.startAngle < angle < self.circle.endAngle:
            return True
        return False
    
    def drawOutline(self, img, thickness=1):
        if self.circle.fullCircle:
            cv.circle(img, self.circle.middle.toIntArr(), int(self.circle.radius), (200, 0, 200), thickness)
            return

        startPoint = Vec2([np.cos(self.circle.startAngle), np.sin(self.circle.startAngle)]) * self.circle.radius + self.circle.middle
        endPoint = Vec2([np.cos(self.circle.endAngle), np.sin(self.circle.endAngle)]) * self.circle.radius + self.circle.middle

        cv.line(img, self.circle.middle.toIntArr(), startPoint.toIntArr(), (200, 0, 200), thickness)
        cv.line(img, self.circle.middle.toIntArr(), endPoint.toIntArr(), (200, 0, 200), thickness)

        self.drawCircleCurve(img, self.circle.radius, thickness)

        # angleRange = self.endAngle - self.startAngle if self.startAngle < self.endAngle else 2 * np.pi - self.startAngle + self.endAngle
        # rangeParts = int(angleRange * self.radius / 10)
        # for i in range(rangeParts):
        #    angle = self.startAngle + angleRange * i / rangeParts
        #    point = Vec2([np.cos(angle), np.sin(angle)]) * self.radius + self.middle
        #
        #    cv.line(img, startPoint.toIntArr(), point.toIntArr(), (200, 0, 200), thickness)
        #    startPoint = point

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
                self.drawCircleCurve(img, d, thickness)


    def drawCircleCurve(self, img, radius, thickness=1):
        if self.circle.fullCircle:
            cv.circle(img, self.circle.middle.toIntArr(), int(radius), (200, 0, 200), thickness)
            return

        startPoint = Vec2([np.cos(self.circle.startAngle), np.sin(self.circle.startAngle)]) * self.circle.radius + self.circle.middle
        
        angleRange = self.circle.endAngle - self.circle.startAngle if self.circle.startAngle < self.circle.endAngle else 2 * np.pi - self.circle.startAngle + self.circle.endAngle
        rangeParts = int(angleRange * radius / 10)
        for i in range(rangeParts):
            angle = self.circle.startAngle + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), np.sin(angle)]) * radius + self.circle.middle

            cv.line(img, startPoint.toIntArr(), point.toIntArr(), (200, 0, 200), thickness)
            startPoint = point

    def getCirclesAreas(img, columns, circles, fullCircle = False):
        areas = []
        for circle in circles:
            #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
            area = CircleArea(circle, fullCircle)
            area.drawOutline(img, 3)
            area.testColumns(columns)
            area.drawColumns(img, 3)
            area.findCurves(img, 3)
            areas.append(area)

        return areas

class PolygonArea:
    def __init__(self, points):
        self.points = points
