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

    leng = len(lines)

    start = -1
    improveCorner = False
    i = 0
    while i < leng:
        if len(lines[i].points) < 4:
            lines.pop(i)
            improveCorner = True
            leng -= 1
            continue

        if improveCorner:
            intersection = PMath.linesIntersection(lines[start].first, lines[start].last, lines[i].first, lines[i].last)
            lines[start].push(intersection, Line.END)
            lines[i].push(intersection, Line.START)
            improveCorner = False

        start = i
        i += 1

    if improveCorner:
        intersection = PMath.linesIntersection(lines[start].first, lines[start].last, lines[0].first, lines[0].last)
        lines[start].push(intersection, Line.END)
        lines[0].push(intersection, Line.START)

    return lines

def findCorners(lines):

    leng = len(lines)
    # TODO: Need to check if start is valid
    start = -1
    improveCorner = False
    i = 0
    while i < leng:
        # if len(lines[(i-1)%leng].points) < 5 or len(lines[(i+1)%leng].points) < 5:
        #     if PMath.angle(lines[(i-1)%leng].first, lines[i].first, lines[i].last) < DEG150:
        #         print("cornerFound", lines[i].first.x, lines[i].first.y, PMath.angle(lines[(i-1)%leng].first, lines[i].first, lines[i].last))
        #         start = i
        #         i += 1
        #         continue

        if len(lines[i].points) < 4:
            #print("delete line", lines[i].first.x, lines[i].first.y)

            lines.pop(i)
            improveCorner = True
            leng -= 1
            continue

        if improveCorner:

            intersection = PMath.linesIntersection(lines[i-1].first, lines[i-1].last, lines[i].first, lines[i].last)
            if intersection is None or intersection.dist(lines[i-1].last) > 10: # lines are parallel
                middle = (lines[i-1].last + lines[i].first) / 2
                p1, _ = PMath.closestPointOnLine(lines[i-1].first, lines[i-1].last, middle)
                p2, _ = PMath.closestPointOnLine(lines[i].first, lines[i].last, middle)

                lines[i-1].replace(p1, Line.END)
                lines[i].replace(p2, Line.START)
                lines = lines[:i] + [Line([p1,p2])] + lines[i:]
                improveCorner = False
                leng += 1
                i += 1
            else:
                lines[i-1].push(intersection, Line.END)
                lines[i].push(intersection, Line.START)
                improveCorner = False

        i += 1

    if improveCorner:
        intersection = PMath.linesIntersection(lines[start].first, lines[start].last, lines[0].first, lines[0].last)
        lines[start].push(intersection, Line.END)
        lines[0].push(intersection, Line.START)

    return lines

def splitIntoSegments(img, lines):
    segments = [Segment()]

    minAngle = (4, 0)

    for i in range(0, len(lines)-1):
        segments[-1] += lines[i]
        ang = PMath.angle(lines[i].first, lines[i].last, lines[i+1].last)
        if ang < DEG150:
            segments.append(Segment())
        
        if ang < minAngle[0]:
            minAngle = (ang, i)

    segments[-1] += lines[-1]

    if len(segments) == 1:
        pos = minAngle[1]
        print(minAngle)
        return [Segment(lines[pos+1:] + lines[:pos+1])] 

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
    i = 0
    while i < len(seg)-3:
        if circle is None:
            circle = Circle.getCircle(seg, i, i+4)

            # no suitable circle found or not valid
            if circle is None or not (circle.areBetweenPointsInside(seg[i:i+3]) and circle.isContourInside(seg.getContour(i,i+4))):
                lines.append(seg[i])
                circle = None

            # circle found
            else:
                circles.append(circle)
                start = i


        else:
            circle = Circle.getCircle(seg, start, i+4)

            # remove circle because not suitable anymore
            if circle is None:
                circles = circles[:-1]
                i = i+2
                # lines.extend(seg[start:i+1]) TODO: check indices exactly

            elif circle.areBetweenPointsInside(seg[start+1:i+4]) and circle.isContourInside(seg.getContour(start, i+4)):
                circles[-1] = circle
            else:
                circle = None
                i = i+2
                lines.append(seg[i])
        i += 1

    #for circle in circles:
    #    cv.circle(img, circle.middle.toIntArr(), int(circle.radius), (0, 200, 0), 2)

    if len(circles) == 0:
        return [], seg.parts

    return circles, lines