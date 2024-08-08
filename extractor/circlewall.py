from extractor.wall import Wall
from extractor.pointmath import PMath
import numpy as np
class CircleWall(Wall):

    THRESHOLD = 2
    FULL_PERIOD = 2*np.pi

    def __init__(self, middle, radius, col, maxP1, maxP2, state):
        self.type = Wall.CIRCLE
        self.state = state
        self.middle = middle
        self.radius = radius

        ang = PMath.getAxisAngle(middle, col)
        self.minAng1 = (ang - 0.05) % CircleWall.FULL_PERIOD
        self.minAng2 = (ang + 0.05) % CircleWall.FULL_PERIOD
        self.maxAng1 = PMath.getAxisAngle(middle, maxP1)
        self.maxAng2 = PMath.getAxisAngle(middle, maxP2)

    def columnIsPart(self, col):
        d = self.middle.dist(col)
        if abs(d - self.radius) > CircleWall.THRESHOLD:
            return False

        ang = PMath.getAxisAngle(self.middle, col)
        if self.circle.overflow and self.start < ang < self.end:


        if t < 0:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP1)
            self.minP1 = col
            if t < maxT:
                self.maxP1 = col
        elif t > 1:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP2)
            self.minP2 = col
            if t > maxT:
                self.maxP2 = col

        return True

    def checkIntersection(self, other):
        if other.type == Wall.LINE:
            point1 = PMath.segmentsIntersection(self.minP1, self.maxP1, other.maxP1, other.maxP2)
            point2 = PMath.segmentsIntersection(self.minP2, self.maxP2, other.maxP1, other.maxP2)

        elif other.type == Wall.CIRCLE:
            point1 = Wall.circleLineIntersection(self.minP2, self.maxP1, other.circle)
            point2 = Wall.circleLineIntersection(self.minP1, self.maxP2, other.circle)

        if point1 is not None:
            self.maxP1 = point1

        if point2 is not None:
            self.maxP2 = point2

    def drawWall(self, dwg, thickness):
        dwg.add(dwg.line(start=self.maxP1.toArr(), end=self.maxP2.toArr(), stroke="rgb(200,150,0)"))