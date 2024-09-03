import numpy as np
from extractor.vec import Vec2
import cv2 as cv
from extractor.pointmath import PMath

class Segment:
    def __init__(self, parts = None):
        self.parts = [] if parts == None else parts

    def __iadd__(self, part):
        self.parts.append(part)
        return self
    
    def prepend(self, seg):
        self.parts = seg.parts + self.parts
        return self
    
    def __getitem__(self, i):
        return self.parts[i]
    
    def __len__(self):
        return len(self.parts)
    
    def getContour(self, start, end):
        contour = []
        for i in range(start, end):
            contour.extend(self.parts[i].points)

        return contour
    
    def dist(self, i1, i2):
        return self.parts[i1].first.dist(self.parts[i2-1].last)

class Line:
    START = 0
    END = 1
    def __init__(self, points):
        self.points = points
        self.first = points[0]
        self.last = points[-1]

    def __iadd__(self, line):
        self.points.extend(line.points[1:])
        self.last = self.points[-1]
        return self

    def __add__(self, line):
        self.points.extend(line.points[1:])
        self.last = self.points[-1]
        return self
    
    def __getitem__(self, i):
        return self.points[i]
    
    def __len__(self):
        return len(self.points)

    def push(self, point, pos):
        if pos == Line.START:
            self.points = [point] + self.points
            self.first = point
        else:
            self.points.append(point)
            self.last = point
        return self

    def replace(self, point, pos):
        if pos == Line.START:
            self.points[0] = point
            self.first = point
        else:
            self.points[-1] = point
            self.last = point
        return self
    
    def length(self):
        return abs(self.last - self.first)

    #def splitShare(intersection, before, between, after):
        
    #    before +

