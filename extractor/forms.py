import numpy as np
from extractor.vec import Vec2
import cv2 as cv

class Segment:
    def __init__(self):
        self.parts = []

    def __iadd__(self, part):
        self.parts.append(part)
        return self
    
    def prepend(self, seg):
        self.parts = seg.parts + self.parts
        return self
    
    def __getitem__(self, i):
        return self.parts[i]
    
    def __len__(self):
        return len(self.parts)
    
    def getContour(self, start, end):
        contour = []
        for i in range(start, end):
            contour.extend(self.parts[i].points)

        return contour
    
    def dist(self, i1, i2):
        return self.parts[i1].first.dist(self.parts[i2-1].last)

class Line:
    def __init__(self, points):
        self.points = points
        self.first = points[0]
        self.last = points[-1]

    def __iadd__(self, line):
        self.points.extend(line.points[1:])
        self.last = self.points[-1]
        return self

    def __add__(self, line):
        self.points.extend(line.points[1:])
        self.last = self.points[-1]
        return self
    
    #def splitShare(intersection, before, between, after):
        
    #    before +

class Circle:
    def __init__(self, middle, radius, start, between, end):
        self.middle = middle
        self.radius = radius
        self.fullCircle = False
        self.allignedMiddle = middle

        if start.dist(end) < 30:
            self.fullCircle = True

        angle1 = Circle.getCircleAngle(start, middle)
        angle2 = Circle.getCircleAngle(between, middle)
        angle3 = Circle.getCircleAngle(end, middle)

        a = angle1 <= angle2
        b = angle2 <= angle3
        c = angle1 <= angle3

        if (a + b + c) % 2 == 1:
            self.startAngle = angle1
            self.endAngle = angle3
            self.start = start
            self.end = end

        else:
            self.startAngle = angle3
            self.endAngle = angle1
            self.start = end
            self.end = start

        if self.endAngle < self.startAngle:
            self.endAngle += 2 * np.pi
            self.overflow = True
        else:
            self.overflow = False

    def getCircleAngle(p, middle):
        unitPoint = p - middle
        unitPoint = unitPoint / abs(unitPoint)
        angle = np.arctan2(unitPoint.y, unitPoint.x)
        return angle if angle >= 0 else angle + 2 * np.pi

    def getCircle(seg, start, end):

        pointDist1 = seg.dist(start, end // 2)
        pointDist2 = seg.dist(start, end)

        m = 1


        if pointDist1 > pointDist2 * 1.5:
            m = 1.5

        p1 = seg[start].first
        p2 = seg[int((start + end) / (2*m))].first
        p3 = seg[int((end-1) / m)].last
        
        a = [[2*p1.x, 2*p1.y, 1], \
            [2*p2.x, 2*p2.y, 1], \
            [2*p3.x, 2*p3.y, 1]]
        
        b = [-p1.x**2 - p1.y**2, \
            -p2.x**2 - p2.y**2, \
            -p3.x**2 - p3.y**2]

        if np.linalg.det(a) == 0:
            return None

        x = np.linalg.solve(a, b)
        middlePoint = Vec2([-x[0], -x[1]])
        radius = np.sqrt(np.power(middlePoint.x, 2) + np.power(middlePoint.y, 2) - x[2])
        if radius < 100 or radius > 3000:
            return None

        return Circle(middlePoint, radius, p1, p2, seg[end-1].last) 

    def areBetweenPointsInside(self, between):

        for i in range(len(between)):
            dist = self.middle.dist(between[i].first)
            if abs(dist - self.radius) > 3:
                return False
        
        return True

    def isContourInside(self, contour):
        for p in contour:
            d = abs(self.middle.dist(p) - self.radius) 
            if d > 3:
                return False
            
        return True

    def isInside(self, p):
        dist = p.dist(self.middle)
        if dist > self.radius:
            return False    

        if self.fullCircle:
            return True

        angle = Circle.getCircleAngle(p, self.middle)
        angle += (2*np.pi) if self.overflow and self.startAngle > angle else 0
        if self.startAngle < angle < self.endAngle:
            return True
        return False
    
    def drawOutline(self, img, thickness=1):
        color = (0, 250, 150)
        if self.fullCircle:
            cv.circle(img, self.middle.toIntArr(), int(self.radius), color, thickness)
            return

        startPoint = Vec2([np.cos(self.startAngle), np.sin(self.startAngle)]) * self.radius + self.middle
        endPoint = Vec2([np.cos(self.endAngle), np.sin(self.endAngle)]) * self.radius + self.middle

        cv.line(img, self.allignedMiddle.toIntArr(), startPoint.toIntArr(), color, thickness)
        cv.line(img, self.allignedMiddle.toIntArr(), endPoint.toIntArr(), color, thickness)

        self.drawCircleCurve(img, self.radius, thickness, color)

    def drawCircleCurve(self, img, radius, thickness=1, color=None):
        if color is None:
            color = (200, 0, 200)

        if self.fullCircle:
            cv.circle(img, self.middle.toIntArr(), int(radius), color, thickness)
            return

        startPoint = Vec2([np.cos(self.startAngle), np.sin(self.startAngle)]) * self.radius + self.middle
        
        angleRange = self.endAngle - self.startAngle if self.startAngle < self.endAngle else 2 * np.pi - self.startAngle + self.endAngle
        rangeParts = int(angleRange * radius / 10)
        for i in range(rangeParts):
            angle = self.startAngle + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), np.sin(angle)]) * radius + self.middle

            cv.line(img, startPoint.toIntArr(), point.toIntArr(), color, thickness)
            startPoint = point
