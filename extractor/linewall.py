from extractor.wall import Wall
from extractor.pointmath import PMath
import cv2 as cv

class LineWall(Wall):

    THRESHOLD = 2

    def __init__(self, p1, maxP1, maxP2, state):
        self.type = Wall.LINE
        self.state = state

        if p1 is not None:
            dir = maxP2 - maxP1
            dir /= abs(dir)
            self.minP1 = p1 - dir
            self.minP2 = p1 + dir
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
                self.maxP1 = col
        elif t > 1:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP2)
            self.minP2 = col
            if t > maxT:
                self.maxP2 = col

        return True

    def checkIntersection(self, other):
        if other.type == Wall.LINE:
            point1 = PMath.linesIntersection(self.minP1, self.maxP1, other.maxP1, other.maxP2)
            point2 = PMath.linesIntersection(self.minP2, self.maxP2, other.maxP1, other.maxP2)

        elif other.type == Wall.CIRCLE:
            point1 = Wall.circleLineIntersection(self.minP2, self.maxP1, other.circle)
            point2 = Wall.circleLineIntersection(self.minP1, self.maxP2, other.circle)

        if point1 is not None:
            self.maxP1 = point1

        if point2 is not None:
            self.maxP2 = point2

    def drawWall(self, img, thickness):
        cv.line(img, self.maxP1.toIntArr(), self.maxP2.toIntArr(), (0, 150, 200), thickness)