from extractor.wall import Wall
from extractor.pointmath import PMath
import cv2 as cv

class LineWall(Wall):

    THRESHOLD = 2

    def __init__(self, col, maxP1, maxP2, state):
        self.type = Wall.LINE
        self.state = state

        if col is not None:
            dir = maxP2 - maxP1
            dir /= abs(dir)
            self.minP1 = col - dir
            self.minP2 = col + dir
        self.maxP1 = maxP1
        self.maxP2 = maxP2
        #self.angle = PMath.getAxisAngle(minP1, minP2)

    def columnIsPart(self, col):
        point, t = PMath.closestPointOnLine(self.minP1, self.minP2, col)
        if point.dist(col) > LineWall.THRESHOLD:
            return False

        if t < 0:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP1)
            self.minP1 = col
            if t < maxT:
                #TODO: should be improved. dont just increase randomly
                self.maxP1 = col + (col - self.minP2)
        elif t > 1:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP2)
            self.minP2 = col
            if t > maxT:
                self.maxP2 = col + (col - self.minP1)

        return True

    def checkIntersection(self, other):
        if other.type == Wall.LINE:
            point1 = PMath.segmentsIntersection(self.minP1, self.maxP1, other.maxP1, other.maxP2)
            point2 = PMath.segmentsIntersection(self.minP2, self.maxP2, other.maxP1, other.maxP2)

        elif other.type == Wall.CIRCLE:
            point1 = Wall.circleLineIntersection(self.minP2, self.maxP1, other.middle, other.radius, 
                                                     other.fullCircle, other.maxAng1, other.maxAng2)
            point2 = Wall.circleLineIntersection(self.minP1, self.maxP2, other.middle, other.radius, 
                                                     other.fullCircle, other.maxAng1, other.maxAng2)

        if point1 is not None:
            self.maxP1 = point1

        if point2 is not None:
            self.maxP2 = point2

    def drawWall(self, dwg, thickness):
        dwg.add(dwg.line(start=self.maxP1.toArr(), end=self.maxP2.toArr(), stroke="rgb(200,150,0)"))
