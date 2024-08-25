from extractor.vec import Vec2
from extractor.pointmath import PMath
import cv2 as cv
from extractor.forms import Line
import numpy as np


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
            # TODO: check for small vonour length
            if p[2] < 0 and len(vecContour) > 100:
                contours.append(Contour(vecContour=vecContour))
                vecContour = []
                continue

            vecContour.append(Vec2(p[:2] + [0.5, 0.5]))

        return contours
                
    
    def __getitem__(self, idx):
        return self.cons[idx]
    
    def __len__(self):
        return len(self.cons)

    def getFurthest2Points(contour):
        idx1 = 0
        max = 0
        for i in range(1, len(contour)-1):
            d = contour[0].dist(contour[i])
            if d > max:
                idx1 = i
                max = d

        idx2 = 0
        max = 0
        for i in range(0, len(contour)-1):
            if idx1 == i:
                continue

            d = contour[idx1].dist(contour[i])
            if d > max:
                idx2 = i
                max = d    


        if idx1 < idx2:
            return idx1, idx2
        
        return idx2, idx1
    
    def splitContourPart(part):

        idx = None
        max = 0
        for i in range(len(part)-1):
                dist = PMath.distancePointToLine(part[0], part[-1], part[i])
                if dist > 1 and dist > max:
                    idx = i
                    max = dist

        if idx is None:
            return [part]
        
        return Contour.splitContourPart(Line(part[:idx+1])) + Contour.splitContourPart(Line(part[idx:]))

    
    def getContourParts(contour, img):
        
        conParts = []

        idx1, idx2 = Contour.getFurthest2Points(contour)
        conParts.extend(Contour.splitContourPart(Line(contour[idx1:idx2+1])))
        conParts.extend(Contour.splitContourPart(Line(contour[idx2:] + contour[:idx1+1])))

        # TODO: better way to find parts
        # for i in range(1, len(contour)-1):
        #     #next = PMath.getAxisAngle(contour[i], contour[i+1])
        #     d = PMath.distancePointToLine(contour[i-10], contour[i+10], contour[i])

        #     if d > 1:
        #         conParts.append(Line(contour[start:i+1]))
        #         #conParts.extend(Contour.splitContourPart(Line(contour[start:i+1])))
        #         start = i

        #next = PMath.getAxisAngle(contour[i], contour[i+1])
        #angle += next - last
        #last = next
        #    if abs(angle) > 0.05:
        #conParts.append(Line(contour[start:] + contour[0:1]))
        #conParts.extend(Contour.splitContourPart(Line(contour[start:] + contour[0:1])))

        # TODO: test if also check for corners here
        #conParts[-1].points.extend(contour[start:])

        return conParts

class ContourPart:
    def __init__(self, contour, start, end):
        self.contour = contour[start:end]

    def first(self):
        return self.contour[0]

    def last(self):
        return self.contour[-1]

    
    
