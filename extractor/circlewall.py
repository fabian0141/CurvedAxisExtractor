from extractor.wall import Wall
from extractor.pointmath import PMath
from extractor.vec import Vec2
import numpy as np

class CircleWall(Wall):

    THRESHOLD = 5
    FULL_PERIOD = 2*np.pi

    def __init__(self, middle, radius, fullCircle, col, maxAng1, maxAng2, state):
        self.type = Wall.CIRCLE
        self.state = state
        self.middle = middle
        self.fullCircle = fullCircle

        if col is not None:
            ang = PMath.getAxisAngle(middle, col)
            self.minAng1 = (ang - 0.05) % CircleWall.FULL_PERIOD
            self.minAng2 = (ang + 0.05) % CircleWall.FULL_PERIOD
            self.radius = col.dist(middle)

        else:
            self.radius = radius

        self.maxAng1 = maxAng1
        self.maxAng2 = maxAng2
        self.outerHalf = ((maxAng1 + maxAng2) / 2 + (maxAng1 < maxAng2) * np.pi) % CircleWall.FULL_PERIOD

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
    
    def isBetweenAngles(start, end, ang):
        return (int(start < ang) + int(end < ang) + int(start < end)) % 2 == 0

    def checkIntersection(self, other):
        point1 = None
        point2 = None

        if other.type == Wall.LINE:
            if abs(self.minAng1 - self.maxAng1) > 1:
                point1 = Wall.circleLineIntersection(other.maxP1, other.maxP2, self.middle, self.radius, 
                                                     self.fullCircle, self.maxAng1, self.minAng1)

            if abs(self.minAng2 - self.maxAng2) > 1:
                point2 = Wall.circleLineIntersection(other.maxP1, other.maxP2, self.middle, self.radius, 
                                                     self.fullCircle, self.minAng2, self.maxAng2)
        #elif other.type == Wall.CIRCLE:
        #    point1 = Wall.circleLineIntersection(self.minP2, self.maxP1, other.circle)
        #    point2 = Wall.circleLineIntersection(self.minP1, self.maxP2, other.circle)

        if point1 is not None:
            self.maxAng1 = PMath.getAxisAngle(self.middle, point1)

        if point2 is not None:
            self.maxAng2 = PMath.getAxisAngle(self.middle, point2)

    def drawWall(self, dwg, thickness):
        color = "rgb(200,150,0)"

        if self.fullCircle:
            dwg.add(dwg.circle(center=self.middle.toArr(), r=self.radius, stroke=color, stroke_width=thickness, fill="none"))
            return

        startPoint = Vec2([np.cos(self.maxAng1), -np.sin(self.maxAng1)]) * self.radius + self.middle
        
        angleRange = self.maxAng2 - self.maxAng1 if self.maxAng1 < self.maxAng2 else 2 * np.pi - self.maxAng1 + self.maxAng2
        rangeParts = int(angleRange * self.radius / 20)
        for i in range(rangeParts+1):
            angle = self.maxAng1 + angleRange * i / rangeParts
            point = Vec2([np.cos(angle), -np.sin(angle)]) * self.radius + self.middle

            dwg.add(dwg.line(start=startPoint.toArr(), end=point.toArr(), stroke=color))
            startPoint = point
