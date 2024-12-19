from extractor.wall import Wall
from extractor.pointmath import PMath
import cv2 as cv

class LineWall(Wall):

    THRESHOLD = 2
    INFINITE_LINE = 10000

    def __init__(self, state, start=None, end=None, col=None, dir=None, extend=False):
        self.type = Wall.LINE
        self.state = state

        if col is not None:
            self.origin = col
            self.dir = dir
            self.start = -LineWall.INFINITE_LINE
            self.end = LineWall.INFINITE_LINE
            self.maxStart = -LineWall.INFINITE_LINE
            self.maxEnd = LineWall.INFINITE_LINE
        else: 
            self.dir = start.dir(end)
            self.origin = start
            self.start = 0
            self.end = LineWall.INFINITE_LINE if extend else start.dist(end)

    def columnIsPart(self, col):
        point, t = PMath.closestPointOnLine(self.minP1, self.minP2, col)
        if point.dist(col) > LineWall.THRESHOLD:
            return False

        if t < 0:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP1)
            self.minP1 = col
            if t < maxT:
                self.maxP1 = col + (col - self.minP2)
        elif t > 1:
            _, maxT = PMath.closestPointOnLine(self.minP1, self.minP2, self.maxP2)
            self.minP2 = col
            if t > maxT:
                self.maxP2 = col + (col - self.minP1)

        return True
    
    def combine(self, other):
        return PMath.isAlmostParallelDirs(self.dir, other.dir) #TODO: check that they are not seperated by a wall, and actually combine

        # diff = self.origin.dist(other.origin)
        # if self.maxStart < diff < self.maxEnd:
        #     self.start = min(self.start, other.start + diff)
        #     self.end = max(self.end, other.end + diff)
        #     return True

        # return False


    def checkIntersection(self, other):
        if other.type == Wall.LINE:
            t = self.segmentsIntersection(other)
            if t is None:
                return
            if other.state == Wall.HARD_WALL and 0 > t > self.maxStart:
                self.maxStart = t
            if 0 > t > self.start:
                self.start = t

            if other.state == Wall.HARD_WALL and 0 < t < self.maxEnd:
                self.maxEnd = t
            if 0 < t < self.end:
                self.end = t


        if other.type == Wall.CIRCLE:
            ts = self.segmentCircleIntersection(other)
            for t in ts:
                if t is None:
                    continue
                if other.state == Wall.HARD_WALL and 0 > t > self.maxStart:
                    self.maxStart = t
                if 0 > t > self.start:
                    self.start = t

                if other.state == Wall.HARD_WALL and 0 < t < self.maxEnd:
                    self.maxEnd = t
                if 0 < t < self.end:
                    self.end = t

    def segmentsIntersection(self, other):
        v = other.origin - self.origin
        denom = self.dir.cross(other.dir)

        t = v.cross(other.dir) / denom
        u = v.cross(self.dir) / denom
        if 0 <= u <= other.end:
            return t

        return None
    
    def segmentCircleIntersection(self, other):
        v = self.origin - other.middle
        a = self.dir.dot(self.dir)
        b = 2*v.dot(self.dir)
        c = v.dot(v) - other.radius**2

        t1, t2 = PMath.quadraticSolver(a, b, c)
        if t1 is None:
            return [None, None]

        p = self.origin + self.dir * t1
        if not other.isInside(p):
            t1 = None
        
        p = self.origin + self.dir * t2
        if not other.isInside(p):
            t2 = None
        return [t1, t2]

    def drawWall(self, dwg, thickness):
        dwg.add(dwg.line(start=(self.origin + self.dir*self.start).toArr(), end=(self.origin + self.dir*self.end).toArr(), stroke="rgb(200,150,0)", stroke_width=thickness))
