from extractor.pointmath import PMath
from extractor.forms import Segment, Line
from extractor.circle import Circle

DEGREE = 57.2957795131
DEG150 = 2.61799387799

def findLines(parts):
    startPoint = parts[0].first
    lines = [parts[0]]

    for i in range(1, len(parts)):
        betweenPoint = parts[i].first
        endPoint = parts[i].last
        dist = PMath.distancePointToLine( startPoint, endPoint, betweenPoint)
        if dist <= 1:
            lines[-1] += parts[i] 
        else:
            #cv.line(img, startPoint.toArr(), betweenPoint.toArr(), (0,0,0), 1)
            #cv.circle(img, betweenPoint.toArr(), 3, (255, 100, 100), 2)
            startPoint = betweenPoint
            lines.append(parts[i])


    #cv.line(img, startPoint.toIntArr(), endPoint.toIntArr(), (0,0,0), 1)
    #cv.circle(img, endPoint.toIntArr(), 3, (255, 100, 100), 2)
    #lines.append(endPoint)

    # check if both ends are a corner
    dist = PMath.distancePointToLine(parts[-1].first, parts[0].last, parts[0].first)
    if dist <= 1:
        lines[0] = lines[-1] + lines[0]
        lines.pop(-1)
  
    leng = len(lines)
    # remove rounded corners
    # for idx in range(leng):
    #     i1 = idx % leng
    #     i2 = (idx + 1) % leng
    #     j1 = (idx + 2) % leng
    #     j2 = (idx + 3) % leng
    #     if lines[i2].first.dist(lines[j1].first) < 3:
    #         intersection = intersectionPoint(lines[i1].first, lines[i2].first, lines[j1].first, lines[j2].first)
    #         #cv.circle(img, intersection.toIntArr(), 3, (155, 50, 0), 2)
    #         Line.splitShare(intersection); lines[i2] = (intersection, (lines[i2][1] + lines[j1][1]) // 2)
    #         lines.pop(j1)
    #         leng -= 1

    return lines

def splitIntoSegments(img, lines):
    segments = [Segment()]


    for i in range(0, len(lines)-1):
        segments[-1] += lines[i]
        if PMath.angle(lines[i].first, lines[i].last, lines[i+1].last) < DEG150:
            segments.append(Segment())

    segments[-1] += lines[-1]

    if len(segments) == 1:
        return segments

    if PMath.angle(lines[-1].first, lines[-1].last, lines[0].last) > DEG150:
        segments[0].prepend(segments[-1])
        segments.pop(-1)
    return segments


# needs to be checked more if all circles are found
# especially if getcircle return null
def findCircles(seg):

    circle = None
    circles = []
    lines = []
    start = 0

    #startEndDist = 0

    for i in range(1, len(seg) - 2):
        if circle is None:
            circle = Circle.getCircle(seg, start, i+3)

            # no suitable circle found
            if circle is None:
                lines.append(seg[start])
                start = i

            # check if circle is valid
            elif circle.areBetweenPointsInside(seg[i:i+3]) and circle.isContourInside(seg.getContour(start,i+3)):
                circles.append(circle)

            # circle found but not valid
            else:
                circle = None
                lines.append(seg[start])
                start = i

        else:
            circle = Circle.getCircle(seg, start, i+3)

            # TODO: better check for if radius is smaller
            # remove circle because not suitable anymore
            if circle is None:
                circles = circles[:-1]
                lines.extend(seg[start:i-1])
                start = i
                continue

            if circle.areBetweenPointsInside(seg[start+1:i+3]) and circle.isContourInside(seg.getContour(start, i+3)):
                circles[-1] = circle
            else:
                circle = None
                lines.extend(seg[start:i-1])
                start = i

    #for circle in circles:
    #    cv.circle(img, circle.middle.toIntArr(), int(circle.radius), (0, 200, 0), 2)

    return circles, lines