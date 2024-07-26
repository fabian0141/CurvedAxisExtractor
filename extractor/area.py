import cv2 as cv
import numpy as np
from extractor.vec import Vec2
from extractor.forms import Circle
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

        for col in self.columns:
            middle = self.circle.middle
            vec = col - middle
            d = col.dist(middle)
            d = self.circle.radius / d
            start = middle + vec * d

            alMiddle = self.circle.allignedMiddle
            if middle == alMiddle:
                end = middle
            else:
                end = CircleArea.linesIntersection(self.circle.start, alMiddle, start, middle)
                if end is None:
                    end = CircleArea.linesIntersection(self.circle.end, alMiddle, start, middle)

            cv.line(img, end.toIntArr(), start.toIntArr(), (200, 0, 200), thickness)

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

class PolygonArea:
    def __init__(self, points):
        self.points = points
