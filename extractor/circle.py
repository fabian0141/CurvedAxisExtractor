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

        a = int(angle1 <= angle2)
        b = int(angle3 < angle2)
        c = int(angle1 <= angle3)

        if (a + b + c) % 2 == 0:
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
            self.overflow = True
        else:
            self.overflow = False

    def getCircle(seg, start, end):

        pointDist1 = seg.dist(start, (end + start) // 2)
        pointDist2 = seg.dist(start, end)

        m = 1


        if pointDist1 > pointDist2 * 1.5:
            m = 1.5

        p1 = seg[start].first
        p2 = seg[int(start + (end-start) / (2*m))].first
        p3 = seg[int((start-1) + (end-start) / m)].last
        
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
        if radius < 300 or radius > 3000:
            return None

        return Circle(middlePoint, radius, p1, p2, seg[end-1].last) 
    
    def getCircle2(seg, start, end):

        p1 = Vec2(seg[start][:2])
        p2 = Vec2(seg[(end + start) // 2][:2])
        p3 = Vec2(seg[end][:2])

        pointDist1 = p1.dist(p2)
        pointDist2 = p1.dist(p3)
        m = 1

        if pointDist1 > pointDist2 * 1.5:
            m = 1.5

        p2 = Vec2(seg[int(start + (end-start) / (2*m))][:2])
        p3 = Vec2(seg[int((start-1) + (end-start) / m)][:2])
        
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
        if radius < 300 or radius > 3000:
            return None

        return Circle(middlePoint, radius, p1, p2, p3)

    def convArr(arr):
        circles = []
        for i in range(len(arr)):
            circles.append(Circle(Vec2(arr[i][6:8]), arr[i][8], Vec2(arr[i][0:2]), Vec2(arr[i][2:4]), Vec2(arr[i][4:6])))

        return circles

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

    def isOutsideAlligned(self, p):
        if self.allignedMiddle.dist(self.middle) < 0.1:
            return False

        a1 = self.end
        a2 = self.start
        b = self.middle
        c = self.allignedMiddle

        a1b = b - a1
        a2b = b - a2
        bc = c - b
        ca1 = a1 - c
        ca2 = a2 - c

        a1p = p - a1
        a2p = p - a2
        bp = p - b
        cp = p - c

        cr11 = a1b.cross(a1p)
        cr12 = a2b.cross(a2p)
        cr2 = bc.cross(bp)
        cr31 = ca1.cross(cp)
        cr32 = ca2.cross(cp)

        return (cr11 >= 0 and cr2 >= 0 and cr31 >= 0 and cr12 >= 0 and cr32 >= 0) or (cr11 <= 0 and cr2 <= 0 and cr31 <= 0 and cr12 <= 0 and cr32 <= 0)
    
    def isInside(self, p):
        dist = p.dist(self.middle)
        if dist > self.radius:
            return False    

        if self.fullCircle:
            return True
        
        if self.isOutsideAlligned(p):
            return False

        ang = PMath.getAxisAngle(self.middle, p)
        if (int(self.startAngle < ang) + int(self.endAngle < ang) + int(self.startAngle < self.endAngle)) % 2 == 0:
            return True
        return False
    
    def drawOutline(self, dwg, thickness=1):
        color = "rgb(100,200,0)"
        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color, stroke_width=thickness, fill="none"))
            return

        startPoint = Vec2([np.cos(self.startAngle), np.sin(self.startAngle)]) * self.radius + self.middle
        endPoint = Vec2([np.cos(self.endAngle), np.sin(self.endAngle)]) * self.radius + self.middle

        dwg.add(dwg.line(start=self.allignedMiddle.toArr(), end=startPoint.toArr(), stroke=color, stroke_width=thickness))
        dwg.add(dwg.line(start=self.allignedMiddle.toArr(), end=endPoint.toArr(), stroke=color, stroke_width=thickness))

        self.drawCircleCurve(dwg, self.radius, thickness, color)

    def drawCircleCurve(self, dwg, radius, thickness=1, color=None):
        if color is None:
            color = "rgb(200,0,200)"

        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color, stroke_width=thickness, fill="none"))
            return

        startPoint = Vec2([np.cos(self.startAngle), np.sin(self.startAngle)]) * self.radius + self.middle
        
        angleRange = self.endAngle - self.startAngle if self.startAngle < self.endAngle else 2 * np.pi - self.startAngle + self.endAngle
        rangeParts = int(angleRange * radius / 10)
        for i in range(rangeParts+1):
            angle = self.startAngle + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), np.sin(angle)]) * radius + self.middle

            dwg.add(dwg.line(start=startPoint.toArr(), end=point.toArr(), stroke=color, stroke_width=thickness))
            startPoint = point

    def allignMiddle(self, ang):
        if self.allignedMiddle != self.middle:
            return

        ang = PMath.angle(self.start, self.end, self.middle) - ang 
        m = (self.start + self.end) / 2
        dir = self.end - self.start
        dir = Vec2([-dir.y, dir.x])
        h = np.tan(ang) / 2
        middle = m - dir*h
        self.allignedMiddle = middle