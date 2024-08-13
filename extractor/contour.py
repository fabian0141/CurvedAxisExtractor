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
            if p[2] < 0:
                contours.append(Contour(vecContour=vecContour))
                vecContour = []
                continue

            vecContour.append(Vec2(p[:2] + [0.5, 0.5]))

        return contours
                
    
    def __getitem__(self, idx):
        return self.cons[idx]
    
    def __len__(self):
        return len(self.cons)
    
    def getContourParts(contour, img):
        
        conParts = []
        start = 0
        last = PMath.getAxisAngle(contour[0], contour[1])
        angle = 0

        # TODO: better way to find parts
        for i in range(1, len(contour)-1):
            next = PMath.getAxisAngle(contour[i], contour[i+1])
            a = (next - last) % (2*np.pi)
            angle += min(a, 2 * np.pi - a)
            last = next
            if abs(angle) > 0.2:
                conParts.append(Line(contour[start:i+1]))
                start = i
                angle = 0

        #next = PMath.getAxisAngle(contour[i], contour[i+1])
        #angle += next - last
        #last = next
        #    if abs(angle) > 0.05:
        conParts.append(Line(contour[start:] + contour[0:1]))

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

    
    
