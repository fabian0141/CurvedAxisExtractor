def splitContours(con, img):
    miniCons = []
    i = 0
    leng = len(con)
    firstIdx = None
    count = 0

    while True:
        max = 0

        #i10 = (i + 10) % leng 
        if firstIdx != None:
            outer = np.min([firstIdx - i, 20])
        else: 
            outer = 20

        i20 = (i + outer) % leng 
        idx = (i+10) % leng



        for j in range(outer):
            ij = (i+j) % leng
            dist = distancePointToLine(con[i], con[i20], con[ij])
            if dist > 1 and dist > max:
                #cv.circle(img, con[ij].toIntArr(), 0, (0, 0, 0), 1)
                idx = ij
                max = dist

        if outer < 20 and max == 0:
            break

        if idx >= i:
            count += idx - i
        else:
            count += idx + leng - i

        if firstIdx != None and count >= firstIdx:
            break



        miniCons.append(idx)
        #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
        #cv.line(img, con[i].toIntArr(), con[idx].toIntArr(), (150,150,150), 1)



        i = idx


        if firstIdx == None:
            firstIdx = leng + i
           #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
           #cv.line(img, con[idx].toIntArr(), con[i+10].toIntArr(), (150,150,150), 1)


    return miniCons

def findCornersFromContour(contour, img):


    lastAng = 0

    corners = []
    splitContours = []
    step = 5
    fullStep = step * 2
    firstCorner = -fullStep
    lastCorner = -fullStep


    for i in range(-fullStep, len(contour) - fullStep):
        
        idxs = [i, (i + step) % len(contour), (i + fullStep) % len(contour)]
        ang = angle(contour[i], contour[i+step], contour[i+fullStep])

        if ang > 30:
            if contour[lastCorner].dist(contour[idxs[1]]) < fullStep:
                if lastAng < ang:
                    lastCorner = idxs[1]
                    lastAng = ang
            else:
                if lastAng > 0:
                    if len(corners) == 0:
                        firstCorner = lastCorner

                    corners.append(contour[lastCorner])
                    cv.circle(img, contour[lastCorner].toIntArr(), 3, (255, 0, 0), 2)
                    if lastCorner < idxs[1]:
                        splitContours.append(contour[lastCorner:idxs[1]])
                    else:
                        splitContours.append(np.concatenate([contour[lastCorner:], contour[:idxs[1]]]))

                lastCorner = idxs[1]
                lastAng = ang

    if lastAng > 0:
        corners.append(contour[lastCorner])
        cv.circle(img, contour[lastCorner].toIntArr(), 3, (255, 0, 0), 2)
        if lastCorner < idxs[1]:
            splitContours.append(contour[lastCorner:idxs[1]])
        else:
            splitContours.append(np.concatenate([contour[lastCorner:], contour[:firstCorner]]))

    if len(corners) == 0:
        return None, contour

    print(len(corners))
    return corners, splitContours

def findLines(points, con, img):
    startPoint = points[con[0]]
    lines = [(startPoint, con[0])]

    for i in range(1, len(con) - 1):
        betweenPoint = points[con[i]]
        endPoint = points[con[i+1]]
        dist = distancePointToLine( startPoint, endPoint, betweenPoint)
        if dist > 1:
            #cv.line(img, startPoint.toArr(), betweenPoint.toArr(), (0,0,0), 1)
            #cv.circle(img, betweenPoint.toArr(), 3, (255, 100, 100), 2)
            startPoint = betweenPoint
            lines.append((betweenPoint, con[i]))

    #cv.line(img, startPoint.toIntArr(), endPoint.toIntArr(), (0,0,0), 1)
    #cv.circle(img, endPoint.toIntArr(), 3, (255, 100, 100), 2)
    #lines.append(endPoint)
    lines.append((endPoint, con[i+1]))

    leng = len(lines)
    # remove rounded corners
    for idx in range(leng):
        i1 = idx % leng
        i2 = (idx + 1) % leng
        j1 = (idx + 2) % leng
        j2 = (idx + 3) % leng
        if lines[i2][0].dist(lines[j1][0]) < 3:
            intersection = intersectionPoint(lines[i1][0], lines[i2][0], lines[j1][0], lines[j2][0])
            #cv.circle(img, intersection.toIntArr(), 3, (155, 50, 0), 2)
            lines[i2] = (intersection, (lines[i2][1] + lines[j1][1]) // 2)
            lines.pop(j1)
            leng -= 1

    for i in range(-1, len(lines)-1):
        #cv.line(img, lines[i].toIntArr(), lines[i+1].toIntArr(), (0,0,0), 1)
        cv.circle(img, lines[i+1][0].toIntArr(), 3, (255, 100, 100), 2)


    return lines

def splitIntoParts(img, points, corners):
    parts = []
    startIdx = None
    startCornerIdx = None
    firstCorner = None
    firstCornerIdx = None

    leng = len(corners)
    for i in range(0, leng):
        i1 = (i+1) % leng
        i2 = (i+2) % leng
        if angle(corners[i][0], corners[i1][0], corners[i2][0]) > 30:
            if firstCorner is None:
                firstCorner = corners[i1]
                firstCornerIdx = i1
            else:
                if startCornerIdx < i2:
                    corns = corners[startCornerIdx:i2]
                else:
                    corns = corners[startCornerIdx:] + corners[:i2]

                if startIdx < corners[i1][1]:
                    parts.append((points[startIdx:corners[i1][1]+1], corns))
                else:
                    parts.append((points[startIdx:] + points[:corners[i1][1]+1], corns))

                cv.line(img, points[startIdx].toIntArr(), points[corners[i1][1]].toIntArr(), (150,0,0), 1)
                cv.circle(img, corners[i1][0].toIntArr(), 5, (0, 100, 100), 3)

            startIdx = corners[i1][1]
            startCornerIdx = i1

    if startCornerIdx is None:
        return [(points, corners)]

    if startCornerIdx < firstCornerIdx:
        corns = corners[startCornerIdx:firstCornerIdx+1]
    else:
        corns = corners[startCornerIdx:] + corners[:firstCornerIdx+1]

    if startIdx < firstCorner[1]:
        parts.append((points[startIdx:firstCorner[1]+1], corns))
    else:
        parts.append((points[startIdx:] + points[:firstCorner[1]+1], corns))



    cv.line(img, points[startIdx].toIntArr(), firstCorner[0].toIntArr(), (150,0,0), 1)
    cv.circle(img, firstCorner[0].toIntArr(), 5, (0, 100, 100), 3)

    pointLeng = len(points)

    #update indexes of corners
    for part in parts:
        startIdx = part[1][0][1]
        
        for i in range(len(part[1])):
            part[1][i] = (part[1][i][0], np.mod(part[1][i][1] - startIdx, pointLeng))



    return parts

def intersectionPoint(p1, p2, q1, q2):

    v1 = p1 - p2
    v2 = q1 - q2
    v3 = p1 - q1

    denom = v1.x * v2.y - v1.y * v2.x

    t = (v3.x * v2.y - v3.y * v2.x) / denom
    #u = -(v1.x * v3.y - v1.x * v3.x) / denom

    return p1 - v1*t

def findCircles(part, img, columns):
    startPoint = part[1][0]
    middlePoint = None
    circles = []
    tresh = 5
    startIdx = 0
    firstCircleIdx = None
    lastCircleIdx = None

    #startEndDist = 0

    for i in range(1, len(part[1]) - 3):
        if middlePoint is None:
            middlePoint, radius = getCircle(startPoint, part[1][i+1], part[1][i+3])
            if middlePoint is None:
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0
            elif areBetweenPointsInside(middlePoint, radius, part[1][i:i+3]) and isContourInside(middlePoint, radius, part, startPoint[1], part[1][i+3][1]):
                circles.append((startPoint[0], part[1][i+1][0], part[1][i+3][0], middlePoint, radius))
                if firstCircleIdx == None:
                    firstCircleIdx = startIdx
                #startEndDist = middlePoint.dist(part[1][i+3])
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0
        else:
            pointDist1 = startPoint[0].dist(part[1][(startIdx+i+3) // 2][0])
            pointDist2 = startPoint[0].dist(part[1][i+3][0])

            if pointDist1 > pointDist2 * 1.5:
                middlePoint, radius = getCircle(startPoint, part[1][(startIdx+i+3) // 6], part[1][(startIdx+i+3) // 3])
            else:
                middlePoint, radius = getCircle(startPoint, part[1][(startIdx+i+3) // 2], part[1][i+3])

            if middlePoint is None:
                circles = circles[:-1]
                continue

            if areBetweenPointsInside(middlePoint, radius, part[1][startIdx+1:i+3]) and isContourInside(middlePoint, radius, part, startPoint[1], part[1][i+3][1]):
                circles[-1] = (startPoint[0], part[1][(startIdx+i+3) // 2][0], part[1][i+3][0], middlePoint, radius)
                #startEndDist = pointDist
                lastCircleIdx = i+3
            else:
                middlePoint = None
                startPoint = part[1][i]
                startIdx = i
                #startEndDist = 0

    if len(circles) > 0:
        print(len(circles))
        #better way to prove
        if part[1][0][0].dist(part[1][-1][0]) < 30 and circles[0][0] == part[1][0][0] and circles[0][2] == part[1][-1][0]:
            return getCirclesAreas(img, columns, circles, True)

        return getCirclesAreas(img, columns, circles)

    return []

def drawOutline(self, img, thickness=1):
    if self.circle.fullCircle:
        cv.circle(img, self.circle.middle.toIntArr(), int(self.circle.radius), (200, 0, 200), thickness)
        return

    startPoint = Vec2([np.cos(self.circle.startAngle), np.sin(self.circle.startAngle)]) * self.circle.radius + self.circle.middle
    endPoint = Vec2([np.cos(self.circle.endAngle), np.sin(self.circle.endAngle)]) * self.circle.radius + self.circle.middle

    cv.line(img, self.circle.middle.toIntArr(), startPoint.toIntArr(), (200, 0, 200), thickness)
    cv.line(img, self.circle.middle.toIntArr(), endPoint.toIntArr(), (200, 0, 200), thickness)

    self.drawCircleCurve(img, self.circle.radius, thickness)

    # angleRange = self.endAngle - self.startAngle if self.startAngle < self.endAngle else 2 * np.pi - self.startAngle + self.endAngle
    # rangeParts = int(angleRange * self.radius / 10)
    # for i in range(rangeParts):
    #    angle = self.startAngle + angleRange * i / rangeParts
    #    point = Vec2([np.cos(angle), np.sin(angle)]) * self.radius + self.middle
    #
    #    cv.line(img, startPoint.toIntArr(), point.toIntArr(), (200, 0, 200), thickness)
    #    startPoint = point








