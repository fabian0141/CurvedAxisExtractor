import numpy as np

def distancePointToLine(l1, l2, p):
    a = l2 - l1
    return abs(a.y * p.x - a.x * p.y + l2.x*l1.y - l2.y*l1.x) / abs(a)

def angle(p1, p2, p3):
    line1 = p1-p2
    line2 = p3-p2

    d = line1.dot(line2)
    n = abs(line1)*abs(line2)
    return np.arccos(d / n)