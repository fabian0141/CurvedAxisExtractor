from extractor.vec import Vec2
from extractor.helper import distancePointToLine
import cv2 as cv
from extractor.forms import Line


class Contour:
    def __init__(self, contour=None, vecContour=None):
        if contour is not None:
            self.cons = []
            for con in contour:
                self.cons.append(Vec2(con[0]))
        
        if vecContour is not None:
            self.cons = vecContour


        #self.cons.append(Vec2(contour[0][0]))

    def convertPixelContour(pixelContours):
        # remove first contour from image edge
        pixelContours = pixelContours[1:] 
        contours = []
        for pixCon in pixelContours:
            contours.append(Contour(contour=pixCon))

        return contours
    
    def convertContour(contour):
        contours = []
        vecContour = []
        i = 0
        print(contour[0], 0)
        for p in contour:
            i += 1
            if p[2] < 0:
                contours.append(Contour(vecContour=vecContour))
                vecContour = []
                print(contour[i-2], i-2)
                print(contour[i], i)
                continue

            vecContour.append(Vec2(p[:2]))

        return contours
                
    
    def __getitem__(self, idx):
        return self.cons[idx]
    
    def __len__(self):
        return len(self.cons)
    
    def getContourParts(contour, img):

        def checkForCorner():
            idx = min(start+10, len(contour)-1)
            end = min(start + 20, len(contour)-1)
            max = 0
            dist = None
            for i in range(start, end):
                dist = distancePointToLine(contour[start], contour[end], contour[i])
                if dist > 1 and dist > max:
                    idx = i
                    max = dist

            if max < 1 and end < start+20:
                idx = end
            return idx
        
        conParts = []
        start = 0

        # find first corner and shift previous part to end
        idx = checkForCorner()
        contour = contour[idx:] + contour[:idx]
        contour.append(contour[0])
        start = 0
        last = len(contour) - 10

        while start < len(contour)-1:
            idx = checkForCorner()
            conParts.append(Line(contour[start:idx+1]))
            start = idx

        # TODO: test if also check for corners here
        conParts[-1].points.extend(contour[start:])

        for con in conParts:
            cv.circle(img, con.first.toIntArr(), 1, (255, 0, 0), 1)
            #cv.line(img, con.first.toIntArr(), con.last.toIntArr(), (150,150,150), 1)

        return conParts

class ContourPart:
    def __init__(self, contour, start, end):
        self.contour = contour[start:end]

    def first(self):
        return self.contour[0]

    def last(self):
        return self.contour[-1]

    
    
