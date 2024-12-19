import cv2 as cv
from extractor.vec import Vec2

def getColumnCenter(imgFile):
    columnImg = cv.imread(imgFile, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(columnImg, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours[1:]
    columns = []

    for con in contours:
        outline = con[:,0]
        center, radius = cv.minEnclosingCircle(outline)
        center = Vec2(center)
        columns.append(center)

    return columns