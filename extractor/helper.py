import numpy as np

def distancePointToLine(l1, l2, p):
    a = l2 - l1
    return abs(a.y * p.x - a.x * p.y + l2.x*l1.y - l2.y*l1.x) / abs(a)

# TODO: something wrong here
def angle(start, middle, end):
    vec1 = (middle - start).toArr()
    vec2 = (end - middle).toArr()
    d = np.dot(vec1, vec2)
    n = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    ang = np.arccos(d / n)
    ang = np.rad2deg(ang)
    return ang