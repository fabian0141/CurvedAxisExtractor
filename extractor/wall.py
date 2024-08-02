from extractor.circle import Circle
from extractor.helper import quadraticSolver
import numpy as np

class Wall:
    LINE = 0
    CIRCLE = 1

    def __init__(self, data, type=-1):
        self.type = type
        self.line = data


    def getIntersection(self, other):
        if self.type == Wall.LINE:
            if other.type == Wall.LINE:
                return Wall.linesIntersection(self.data, other.data)
            
            if other.type == Wall.CIRCLE:
                return Wall.circleLineIntersection(self.data, other.data)
            
        if self.type == Wall.CIRCLE:
            if other.type == Wall.LINE:
                return Wall.circleLineIntersection(other.data, self.data)
            
            if other.type == Wall.CIRCLE:
                return Wall.circleIntersection(self.data, other.data)

    def linesIntersection(p, q):
        v1 = p[0] - p[1]
        v2 = q[0] - q[1]
        v3 = p[0] - q[0]

        denom = v1.x * v2.y - v1.y * v2.x

        t = (v3.x * v2.y - v3.y * v2.x) / denom
        u = -(v1.x * v3.y - v1.y * v3.x) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return p[0] - v1*t

        return None
    
    def circleLineIntersection(l, c):
        a = l[1] - l[0]
        b = l[0] - c.allignedMiddle

        k1 = a.dot(a)
        k2 = 2*a.dot(b)
        k3 = b.dot(b) - c.radius**2

        t1, t2 = quadraticSolver(k1, k2, k3)
        if t1 is None:
            return None

        # need only to check the closest
        # otherwise circle line will intersect
        if t1 < 0:
            if t2 > 1:
                return None
            t1 = t2
        elif t1 > t2:
            t1 = t2

        p1 = a * t1 + l[0]
        if not c.isInside(p1, False):
            return None
        
        return p1



    def circleIntersection(c1, c2):

        r1 = c1.radius
        r2 = c2.radius
        c1 = c1.allignedMiddle
        c2 = c2.allignedMiddle


        g = c1.dot(c1) - r1**2
        d = c1.x - c2.x
        e = c1.y - c2.y
        f = (c2.dot(c2) - r2**2 - g) / 2

        k = d/e
        l = f/e

        a = 1 + k**2
        b = 2*d*f/(e**2) - 2*c1.x - 2*c1.y*k
        c = l**2 - 2*l*c1.y + g

        t1, t2 = quadraticSolver(a, b, c)

        p1 = k*t1 + l
        p2 = k*t2 + l


