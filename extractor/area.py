from extractor.pointmath import PMath
from extractor.linewall import LineWall
from extractor.circlewall import CircleWall

class CircleArea:
    def __init__(self, circle, fullCircle):
        self.circle = circle
        self.columns = []
        self.circles = []
        self.lines = []
    
    def testColumns(self, columns):
        length = len(columns)
        i = 0
        while i < length:
            if self.circle.isInside(columns[i]):
                self.columns.append(columns[i])
                
            i += 1


    def drawColumns(self, dwg, thickness=1):
        for col in self.columns:
            dwg.add(dwg.circle(center=col.toArr(), r=thickness, fill="rgb(200,0,200)"))


    def findCurves(self, columns, walls):

        circles = []
        lines = []

        restColumns = []

        for keepColumn in columns:
            if not keepColumn[1]:
                restColumns.append(keepColumn[0])
                continue

            col = keepColumn[0]

            minDist = min(300, self.circle.middle.dist(self.circle.allignedMiddle) + 100)
            if col.dist(self.circle.middle) > minDist and self.circle.isInside(col):
                dir = self.circle.middle.dir(col)
                lineWall = LineWall(LineWall.ADDED_WALL, col=col, dir=dir)

                circleWall = CircleWall(LineWall.ADDED_WALL, self.circle.middle, col=col)

                lineWall.checkForIntersections(walls)
                circleWall.checkForIntersections(walls)

                lines.append(lineWall)
                circles.append(circleWall)
                continue

            restColumns.append(keepColumn[0])


        # combine circles
        leng = len(circles)
        i = 0
        while i  < leng:
            j = i+1
            while j < leng:
                if circles[i].combine(circles[j]):
                    circles.pop(j) 
                    leng -= 1
                    continue
                j += 1
            i += 1

        #extend circles
        for i in range(len(circles)):
            for col in restColumns:
                if circles[i].extend(col):
                    dir = circles[i].middle.dir(col)   
                    lineWall = LineWall(LineWall.ADDED_WALL, col=col, dir=dir)
                    lineWall.checkForIntersections(walls)

                    lines.append(lineWall)

        # recenter middlepoints
        for circle in circles:
            if len(circle.cols) > 2:
                m, r = PMath.getCircle(circle.cols)
                circle.middle = m
                circle.radius = r

        # combine lines
        leng = len(lines)
        i = 0
        while i < leng:
            j = i+1
            while j < leng:
                if lines[i].combine(lines[j]):
                    lines.pop(j) 
                    leng -= 1
                    continue
                j += 1
            i += 1
        
        self.circles = circles
        self.lines = lines

    def drawArea(self, dwg, thickness = 1):
        for line in self.lines:
            line.drawWall(dwg, thickness)

        for circle in self.circles:
            circle.drawWall(dwg, thickness)

        #self.drawColumns(dwg, thickness)
        self.circle.drawOutline(dwg, thickness)

    def getCirclesAreas(columns, circles, fullCircle = False):
        areas = []
        for circle in circles:
            #cv.circle(img, c.toIntArr(), int(r), (255, 100, 100), 2)
            area = CircleArea(circle, fullCircle)
            area.testColumns(columns)
            #area.drawColumns(img, 3)
            areas.append(area)
        return areas

    def checkNeighboringCircleAreas(circleAreas): #TODO: maybe weird case where triangle flips
        circleData = []

        if len(circleAreas) == 3:
            c1 = circleAreas[0].circle
            c2 = circleAreas[1].circle
            c3 = circleAreas[2].circle
            if (c1.end == c2.start and c2.end == c3.start and c3.end == c1.start) or (c1.start == c2.end and c2.start == c3.end and c3.start == c1.end): #TODO: check if there are overlappings

                intersection = PMath.triangleCirclePoint(c1.start, c2.start, c3.start)
                c1.allignedMiddle = intersection
                c2.allignedMiddle = intersection
                c3.allignedMiddle = intersection
                return

        for i in range(len(circleAreas)):
            c = circleAreas[i].circle
            if (c.fullCircle):
                continue

            ang = PMath.angle(c.start, c.end, c.middle)
            circleData.append([i, c.start, c.end, c.middle, ang])

        circlePairs = []


        for i in range(len(circleData)):
            for j in range(i+1, len(circleData)):

                # connect end with start
                if circleData[i][2] == circleData[j][1]:
                    ang = circleData[i][4] + circleData[j][4]

                    ang -= PMath.angle(circleData[i][1], circleData[i][2], circleData[j][2])
                    if ang < 0:
                        continue
                    circlePairs.append((i, j, ang))

                if circleData[j][2] == circleData[i][1]:
                    ang = circleData[i][4] + circleData[j][4]
                    ang -= PMath.angle(circleData[j][1], circleData[j][2], circleData[i][2])
                    if ang < 0:
                        continue
                    circlePairs.append((j, i, ang))

        circlePairs = sorted(circlePairs, key=lambda x: x[2])

        while len(circlePairs) > 0:
            val = circlePairs[0][2] / 2
            n1 = circlePairs[0][0]
            n2 = circlePairs[0][1]
            circleAreas[n1].circle.allignMiddle(val)
            circleAreas[n2].circle.allignMiddle(val)
            circlePairs.pop(0)
            CircleArea.checkPairs(circleAreas, circlePairs, val, n1, n2)

    def checkPairs(circleAreas, circlePairs, val, n1, n2=None):
        
        i = 0
        while i < len(circlePairs):

            if circlePairs[i][0] == n1 or circlePairs[i][0] == n2:
                val = circlePairs[i][2] - val
                n1 = circlePairs[i][1]
                circlePairs.pop(i)
                circleAreas[n1].circle.allignMiddle(val)
                CircleArea.checkPairs(circleAreas, circlePairs, val, n1)
                continue

            if circlePairs[i][1] == n1 or circlePairs[i][1] == n2:
                val = circlePairs[i][2] - val
                n1 = circlePairs[i][0]
                circleAreas[n1].circle.allignMiddle(val)
                circlePairs.pop(i)
                CircleArea.checkPairs(circleAreas, circlePairs, val, n1)
                continue

            i += 1

    def getWalls(self):
        if self.circle.fullCircle:
            return [CircleWall(LineWall.HARD_WALL, self.circle.middle, radius=self.circle.radius, fullCircle=True)]
        else:
            return [
                CircleWall(LineWall.HARD_WALL, self.circle.middle, radius=self.circle.radius, start=self.circle.startAngle, end=self.circle.endAngle),
                LineWall(LineWall.GUIDE_WALL, start=self.circle.allignedMiddle, end=self.circle.start, extend=True),
                LineWall(LineWall.GUIDE_WALL, start=self.circle.allignedMiddle, end=self.circle.end, extend=True)
            ]