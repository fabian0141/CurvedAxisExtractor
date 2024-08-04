import numpy as np

class PMath:
    def distancePointToLine(l1, l2, p):
        a = l2 - l1
        return abs(a.y * p.x - a.x * p.y + l2.x*l1.y - l2.y*l1.x) / abs(a)

    def closestPointOnLine(l1, l2, p):
        
        dir1 = l2 - l1
        dir2 = dir1.perp()

        t = dir2.cross(l1 - p)
        t /= dir1.cross(dir2)

        return l1 + dir1*t, t

    def angle(p1, p2, p3):
        line1 = p1-p2
        line2 = p3-p2

        d = line1.dot(line2)
        n = abs(line1)*abs(line2)
        return np.arccos(d / n)

    def quadraticSolver(a, b, c):
        disc = b**2 - 4*a*c
        if disc < 0:
            return None, None
        
        disc = np.sqrt(disc)
        div = a*2
        t1 = (-b + disc) / div 
        t2 = (-b - disc) / div
        return t1, t2

    def getAxisAngle(p1, p2):
        unitPoint = p2 - p1
        unitPoint = unitPoint / abs(unitPoint)
        angle = np.arctan2(unitPoint.y, unitPoint.x)
        return angle if angle >= 0 else angle + 2 * np.pi
    
    def linesIntersection(p1, p2, q1, q2):
        v1 = p1 - p2
        v2 = q1 - q2
        v3 = p1 - q1

        denom = v1.x * v2.y - v1.y * v2.x

        t = (v3.x * v2.y - v3.y * v2.x) / denom
        u = -(v1.x * v3.y - v1.y * v3.x) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 - v1*t

        return None