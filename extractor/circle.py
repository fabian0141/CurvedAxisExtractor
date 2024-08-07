import numpy as np
from extractor.vec import Vec2
from extractor.pointmath import PMath
import svgwrite

class Circle:
    def __init__(self, middle, radius, start, between, end):
        self.middle = middle
        self.radius = radius
        self.fullCircle = False
        self.allignedMiddle = middle

        if start.dist(end) < 30:
            self.fullCircle = True

        angle1 = PMath.getAxisAngle(middle, start)
        angle2 = PMath.getAxisAngle(middle, between)
        angle3 = PMath.getAxisAngle(middle, end)

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

    def isInside(self, p, checkRadius=True):
        dist = p.dist(self.middle)
        if checkRadius and dist > self.radius:
            return False    

        if self.fullCircle:
            return True

        angle = PMath.getAxisAngle(self.middle, p)
        angle += (2*np.pi) if self.overflow and self.startAngle > angle else 0
        if self.startAngle < angle < self.endAngle:
            return True
        return False
    
    def drawOutline(self, dwg, thickness=1):
        color = "rgb(100,200,0)"
        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color))
            return

        startPoint = Vec2([np.cos(self.startAngle), -np.sin(self.startAngle)]) * self.radius + self.middle
        endPoint = Vec2([np.cos(self.endAngle), -np.sin(self.endAngle)]) * self.radius + self.middle

        dwg.add(dwg.line(start=self.allignedMiddle.toArr(), end=startPoint.toArr(), stroke=color))
        dwg.add(dwg.line(start=self.allignedMiddle.toArr(), end=endPoint.toArr(), stroke=color))

        self.drawCircleCurve(dwg, self.radius, thickness, color)

    def drawCircleCurve(self, dwg, radius, thickness=1, color=None):
        if color is None:
            color = "rgb(200,0,200)"

        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color))
            return

        startPoint = Vec2([np.cos(self.startAngle), -np.sin(self.startAngle)]) * self.radius + self.middle
        
        angleRange = self.endAngle - self.startAngle if self.startAngle < self.endAngle else 2 * np.pi - self.startAngle + self.endAngle
        rangeParts = int(angleRange * radius / 10)
        for i in range(rangeParts):
            angle = self.startAngle + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), -np.sin(angle)]) * radius + self.middle

            dwg.add(dwg.line(start=startPoint.toArr(), end=point.toArr(), stroke=color))
            startPoint = point

    
    def drawSecondLines(self, img, columns, thickness):
        for col in columns:
            middle = self.middle
            vec = col - middle
            d = col.dist(middle)
            d = self.radius / d
            start = middle + vec * d

            alMiddle = self.allignedMiddle
            if middle == alMiddle:
                end = middle
            else:
                end = PMath.segmentsIntersection(self.start, alMiddle, start, middle)
                if end is None:
                    end = PMath.segmentsIntersection(self.end, alMiddle, start, middle)

            cv.line(img, end.toIntArr(), start.toIntArr(), (200, 0, 200), thickness)


    def allignMiddle(self, ang):
        if self.allignedMiddle != self.middle:
            return

        ang = PMath.angle(self.start, self.end, self.middle) - ang 
        m = (self.start + self.end) / 2
        dir = self.end - self.start
        dir = Vec2([-dir.y, dir.x])
        h = np.tan(ang) / 2
        middle = m + dir*h
        self.allignedMiddle = middle