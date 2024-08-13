from extractor.circle import Circle
from extractor.pointmath import PMath
import numpy as np

class Wall:
    LINE = 0
    CIRCLE = 1

    HARD_WALL = 0
    GUIDE_WALL = 1
    ADDED_WALL = 2

    def checkForIntersections(self, walls):
        for other in walls:
            self.checkIntersection(other)

    
    def circleLineIntersection(l1, l2, middle, radius, fullCircle, start, end):
        a = l2 - l1
        b = l1 - middle

        k1 = a.dot(a)
        k2 = 2*a.dot(b)
        k3 = b.dot(b) - radius**2

        t1, t2 = PMath.quadraticSolver(k1, k2, k3)
        if t1 is None:
            return None

        # need only to check the closest
        # otherwise circle line will intersect
        if t1 < 0 or t1 > 1:
            if t2 < 0 or t2 > 1:
                return None
            t1 = t2
        elif t1 > t2:
            if t2 >= 0 and t2 <= 1:
                t1 = t2

        p1 = a * t1 + l1
        if not Wall.isInside(middle, fullCircle, start, end, p1):
            return None
        return p1
    

    def isInside(middle, fullCircle, start, end, p):
        if fullCircle:
            return True

        angle = PMath.getAxisAngle(middle, p)
        angle += (2*np.pi) if start > end and start > angle else 0
        if start < angle < end:
            return True
        return False



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


