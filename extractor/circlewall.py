from extractor.wall import Wall
from extractor.pointmath import PMath
from extractor.vec import Vec2
import numpy as np

class CircleWall(Wall):

    THRESHOLD = 5
    FULL_PERIOD = 2*np.pi

    def __init__(self, state, middle, radius=None, fullCircle=False, col=None, start=None, end=None):
        self.type = Wall.CIRCLE
        self.state = state
        self.middle = middle
        self.fullCircle = fullCircle
        if fullCircle:
            self.radius = radius
            return

        if col is not None:
            ang = PMath.getAxisAngle(middle, col)
            self.col = col
            self.ang = ang
            self.start = -CircleWall.FULL_PERIOD
            self.end = CircleWall.FULL_PERIOD
            self.maxStart = -CircleWall.FULL_PERIOD
            self.maxEnd = CircleWall.FULL_PERIOD
            self.radius = col.dist(middle)
            self.cols = [col]

        else:
            self.ang = start
            self.start = 0
            self.end = (end - start) % CircleWall.FULL_PERIOD
            self.radius = radius

        #self.outerHalf = ((maxAng1 + maxAng2) / 2 + (maxAng1 < maxAng2) * np.pi) % CircleWall.FULL_PERIOD

    def columnIsPart(self, col):
        d = self.middle.dist(col)
        if abs(d - self.radius) > CircleWall.THRESHOLD:
            return False

        ang = PMath.getAxisAngle(self.middle, col)
        
        if CircleWall.isBetweenAngles(self.outerHalf, self.maxAng1, ang): #before maxAng1
            self.maxAng1 = ang
            self.minAng1 = ang
            self.outerHalf = ((self.maxAng1 + self.maxAng2) / 2 + (self.maxAng1 < self.maxAng2) * np.pi) % CircleWall.FULL_PERIOD

        elif CircleWall.isBetweenAngles(self.maxAng2, self.outerHalf, ang): #after maxAng2
            self.maxAng2 = ang
            self.minAng2 = ang
            self.outerHalf = ((self.maxAng1 + self.maxAng2) / 2 + (self.maxAng1 < self.maxAng2) * np.pi) % CircleWall.FULL_PERIOD

        elif CircleWall.isBetweenAngles(self.maxAng1, self.minAng1, ang): #after maxAng1
            self.minAng1 = ang

        elif CircleWall.isBetweenAngles(self.minAng2, self.maxAng2, ang): #before maxAng2
            self.minAng2 = ang

        return True
    
    def combine(self, other):
        if abs(self.radius - other.radius) > 20:
            return False
        
        if self.col == other.col:
            print("WHUT?!")
        
        combinable = False

        a = (other.ang + other.end - self.ang + CircleWall.FULL_PERIOD) % CircleWall.FULL_PERIOD
        if 0 < a < self.maxEnd:
            self.end = max(self.end, a)
            combinable = True

        a = (other.ang + other.start - self.ang + CircleWall.FULL_PERIOD) % CircleWall.FULL_PERIOD - CircleWall.FULL_PERIOD # other direction
        if 0 > a > self.maxStart:
            self.start = min(self.start, a)
            combinable =  True

        if combinable:
            self.cols.append(other.col)

        return combinable
    
    def extend(self, other):
        if abs(self.middle.dist(other) - self.radius) > 3:
            return False
        
        if self.col == other:
            print("WHUT?!")

        extended = False

        ang = PMath.getAxisAngle(self.middle, other)
        a = (ang - self.ang + CircleWall.FULL_PERIOD) % CircleWall.FULL_PERIOD

        if 0 < a < self.maxEnd:
            self.end = max(self.end, a)
            extended = True

        a = a - CircleWall.FULL_PERIOD
        if 0 > a > self.maxStart:
            self.start = min(self.start, a)
            extended = True

        if extended:
            self.cols.append(other)

        return extended

    def isInside(self, p):
        if self.fullCircle:
            return True

        start = (self.ang + self.start + CircleWall.FULL_PERIOD) % CircleWall.FULL_PERIOD
        end = (self.ang + self.end) % CircleWall.FULL_PERIOD

        ang = PMath.getAxisAngle(self.middle, p)
        if (int(start < ang) + int(end < ang) + int(start < end)) % 2 == 0:
            return True
        return False
    
    def isBetweenAngles(start, end, ang):
        return (int(start < ang) + int(end < ang) + int(start < end)) % 2 == 0

    def checkIntersection(self, other):

        if other.type == Wall.LINE:

            angles = self.circleSegmentIntersection(other)
            for a in angles:
                if a is None:
                    continue

                a = (a - self.ang + CircleWall.FULL_PERIOD) % CircleWall.FULL_PERIOD
                if other.state == Wall.HARD_WALL and 0 < a < self.maxEnd:
                    self.maxEnd = a
                if 0 < a < self.end:
                    self.end = a

                a = a - CircleWall.FULL_PERIOD # other direction
                if other.state == Wall.HARD_WALL and 0 > a > self.maxStart:
                    self.maxStart = a
                if 0 > a > self.start:
                    self.start = a

        # TODO: circle circle


    def circleSegmentIntersection(self, other):
        v = other.origin - self.middle
        a = other.dir.dot(other.dir)
        b = 2*v.dot(other.dir)
        c = v.dot(v) - self.radius**2

        t1, t2 = PMath.quadraticSolver(a, b, c)
        if t1 is None:
            return [None, None]

        if other.start < t1 < other.end:
            p = other.origin + other.dir * t1
            a1 = PMath.getAxisAngle(self.middle, p)
        else:
            a1 = None
        
        if other.start < t2 < other.end:
            p = other.origin + other.dir * t2
            a2 = PMath.getAxisAngle(self.middle, p)
        else:
            a2 = None

        return [a1, a2]

    def rearangeMiddlePoint(self, other):
        m = (self.col + other.col) / 2
        d = other.col - self.col
        n = d.normal()
        r = (self.radius + other.radius) / 2
        t = np.sqrt(r**2 - (abs(d)/2)**2)

        p1 = m + n*t
        p2 = m - n*t
        self.radius = r

        if self.middle.dist(p1) < self.middle.dist(p2):
            self.middle = p1
        else:
            self.middle = p2

    def drawWall(self, dwg, thickness):
        color = "rgb(200,150,0)"

        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color, stroke_width=thickness, fill="none"))
            return
        
        angleRange = self.end - self.start if self.start < self.end else 2 * np.pi - self.start + self.end
        if angleRange > CircleWall.FULL_PERIOD:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color, stroke_width=thickness, fill="none"))
            return

        startPoint = Vec2([np.cos(self.ang + self.start), np.sin(self.ang + self.start)]) * self.radius + self.middle
        rangeParts = int(angleRange * self.radius / 20)
        for i in range(rangeParts+1):
            angle = self.ang + self.start + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), np.sin(angle)]) * self.radius + self.middle

            dwg.add(dwg.line(start=startPoint.toArr(), end=point.toArr(), stroke=color, stroke_width=thickness))
            startPoint = point
