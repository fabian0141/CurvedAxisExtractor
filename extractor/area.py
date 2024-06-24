import cv2 as cv
import numpy as np
from extractor.vec import Vec2

class CircleArea:
    def __init__(self, p1, p2, p3, middle, radius):
        self.middle = middle
        self.radius = radius
        self.columns = []
        self.curvedWalls = []
        self.fullCircle = False

        if p1.dist(p3) < 10:
            self.fullCircle = True
            return

        angel1 = CircleArea.getCircleAngel(p1, middle)
        angel2 = CircleArea.getCircleAngel(p2, middle)
        angel3 = CircleArea.getCircleAngel(p3, middle)

        a = angel1 <= angel2
        b = angel2 <= angel3
        c = angel1 <= angel3

        if (a + b + c) % 2 == 1:
            self.startAngel = angel1
            self.endAngel = angel3
        else:
            self.startAngel = angel3
            self.endAngel = angel1

        if self.endAngel < self.startAngel:
            self.endAngel += 2 * np.pi
            self.overflow = True
        else:
            self.overflow = False

        print(self.startAngel * 180 / np.pi, self.endAngel * 180 / np.pi)

    def getCircleAngel(p, middle):
        unitPoint = p - middle
        unitPoint = unitPoint / abs(unitPoint)
        angel = np.arctan2(unitPoint.y, unitPoint.x)
        return angel if angel >= 0 else angel + 2 * np.pi
    
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
        dist = p.dist(self.middle)
        if dist > self.radius:
            return False    

        if self.fullCircle:
            return True

        angel = CircleArea.getCircleAngel(p, self.middle)
        angel += (2*np.pi) if self.overflow and self.startAngel > angel else 0
        if self.startAngel < angel < self.endAngel:
            return True
        return False
    
    def drawOutline(self, img, thickness=1):
        if self.fullCircle:
            cv.circle(img, self.middle.toIntArr(), int(self.radius), (200, 0, 200), thickness)
            return

        startPoint = Vec2([np.cos(self.startAngel), np.sin(self.startAngel)]) * self.radius + self.middle
        endPoint = Vec2([np.cos(self.endAngel), np.sin(self.endAngel)]) * self.radius + self.middle

        cv.line(img, self.middle.toIntArr(), startPoint.toIntArr(), (200, 0, 200), thickness)
        cv.line(img, self.middle.toIntArr(), endPoint.toIntArr(), (200, 0, 200), thickness)

        self.drawCircleCurve(img, self.radius, thickness)

        # angelRange = self.endAngel - self.startAngel if self.startAngel < self.endAngel else 2 * np.pi - self.startAngel + self.endAngel
        # rangeParts = int(angelRange * self.radius / 10)
        # for i in range(rangeParts):
        #    angel = self.startAngel + angelRange * i / rangeParts
        #    point = Vec2([np.cos(angel), np.sin(angel)]) * self.radius + self.middle
        #
        #    cv.line(img, startPoint.toIntArr(), point.toIntArr(), (200, 0, 200), thickness)
        #    startPoint = point

    def drawColumns(self, img, thickness=1):
        for col in self.columns:
            cv.circle(img, col.toIntArr(), 5, (200, 0, 200), thickness)

    def findCurves(self, img, thickness):
        distCol = []
        for col in self.columns:
            dist = col.dist(self.middle)
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
        if self.fullCircle:
            cv.circle(img, self.middle.toIntArr(), int(radius), (200, 0, 200), thickness)
            return

        startPoint = Vec2([np.cos(self.startAngel), np.sin(self.startAngel)]) * self.radius + self.middle
        
        angelRange = self.endAngel - self.startAngel if self.startAngel < self.endAngel else 2 * np.pi - self.startAngel + self.endAngel
        rangeParts = int(angelRange * radius / 10)
        for i in range(rangeParts):
            angel = self.startAngel + angelRange * i / rangeParts
            point = Vec2([np.cos(angel), np.sin(angel)]) * radius + self.middle

            cv.line(img, startPoint.toIntArr(), point.toIntArr(), (200, 0, 200), thickness)
            startPoint = point



class PolygonArea:
    def __init__(self, points):
        self.points = points
