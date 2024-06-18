

def distancePointToLine(l1, l2, p):
    a = l2 - l1
    return abs(a.y * p.x - a.x * p.y + l2.x*l1.y - l2.y*l1.x) / abs(a)