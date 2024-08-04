
import numpy as np

class Vec2:
    #def __init__(self, x, y):
    #    self.x = x
    #    self.y = y

    def __init__(self, arr):
        if len(arr) != 2:
            raise ValueError("Vec2 wrong values")
        self.x = arr[0]
        self.y = arr[1]
        self.arr = arr

    def __getitem__(self, item):
        if item == 0:
            return self.x
        else:
            return self.y
        
    def toArr(self):
        return self.arr
    
    def __add__(self, vec2):
        return Vec2([self.x + vec2.x, self.y + vec2.y])
    
    def __sub__(self, vec2):
        return Vec2(self.x - vec2.x, self.y - vec2.y)

    def __sub__(self, vec2):
        return Vec2([self.x - vec2.x, self.y - vec2.y])
    
    def __abs__(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)
    
    def __mul__(self, scalar):
        return Vec2([self.x * scalar, self.y * scalar])


    def __truediv__(self, scalar):
        return Vec2([self.x / scalar, self.y / scalar])
    
    def __eq__(self, vec2):
        return abs(self.x - vec2.x) < 5 and abs(self.y - vec2.y) < 5
    
    def dist(self, vec2):
        return abs(self - vec2)
    
    def dot(self, vec2):
        return self.x*vec2.x + self.y*vec2.y

    def cross(self, vec2):
        return self.x*vec2.y - self.y*vec2.x

    def perp(self):
        return Vec2([-self.y, self.x])

    def convertContour(contours):
        vecContour = []
        contours = contours[1:]
        #contours = contours[1:]

        for contour in contours:
            vecContour.append([])
            for con in contour:
                vecContour[-1].append(Vec2(con[0]))

            vecContour[-1].append(Vec2(contour[0][0]))

        return vecContour
    
    def toIntArr(self):
        return (int(self.x + 0.5), int(self.y + 0.5))

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def __unicode__(self):
        return 'u' + self.__str__()
