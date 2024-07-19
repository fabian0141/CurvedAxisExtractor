from extractor.vec import Vec2
from extractor.helper import distancePointToLine
import cv2 as cv

class Contour:
    def __init__(self, contour):
        self.cons = []
        for con in contour:
            self.cons.append(Vec2(con[0]))
        
        #self.cons.append(Vec2(contour[0][0]))

    def convertPixelContour(pixelContours):
        # remove first contour from image edge
        pixelContours = pixelContours[1:] 
        contours = []
        for pixCon in pixelContours:
            contours.append(Contour(pixCon))

        return contours
    
    def __getitem__(self, idx):
        return self.cons[idx]
    
    def __len__(self):
        return len(self.cons)
    

class ContourPart:
    def __init__(self, contour, start, end):
        self.contour = contour[start:end]

    def first(self):
        return self.contour[0]

    def last(self):
        return self.contour[-1]

    def getContourParts(contour, img):

        def checkForCorner():
            idx = min(start+10, len(contour)-1)
            end = min(start + 20, len(contour)-1)
            max = 0
            for i in range(start, end):
                dist = distancePointToLine(contour[start], contour[end], contour[i])
                if dist > 1 and dist > max:
                    idx = i
                    max = dist
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
            conParts.append(ContourPart(contour, start, idx+1))
            start = idx

        # TODO: test if also check for corners here
        conParts[-1].contour.extend(contour[start:])

        for con in conParts:
            cv.circle(img, con.first().toIntArr(), 1, (255, 0, 0), 1)
            cv.line(img, con.first().toIntArr(), con.last().toIntArr(), (150,150,150), 1)


        return conParts
    
