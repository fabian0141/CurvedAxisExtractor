import cv2 as cv
import numpy as np
from extractor.vec import Vec2
from extractor.forms import Circle

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

        self.circle.drawOutline(img, thickness)



    def getCirclesAreas(img, columns, circles, fullCircle = False):
        areas = []
        for circle in circles:
            #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
            area = CircleArea(circle, fullCircle)
            area.testColumns(columns)
            #area.drawColumns(img, 3)
            area.findCurves(img, 3)
            area.drawArea(img, 3)
            areas.append(area)
        return areas

class PolygonArea:
    def __init__(self, points):
        self.points = points
