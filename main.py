import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool
from extractor.vec import Vec2
from extractor.helper import distancePointToLine, angle
from extractor.area import CircleArea
from extractor.contour import Contour
from extractor.formfinder import findLines, splitIntoSegments, findCircles
from extractor.forms import Segment, Circle
from extractor.column import getColumnCenter


def extractPartsAndWalls(imgFile, columnImg, out = "test.png"):
    #load image
    img = cv.imread(imgFile)
    if img is None:
        return
    
    # convert to black white and get contour
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0,255,0), 1)

    # convert contour pixel array to contour list
    contours = Contour.convertPixelContour(contours)

    columns = getColumnCenter(columnImg)
    circleAreas = []

    for points in contours:
        #findCornersFromContour(points, img)
        miniCons = Contour.getContourParts(points, img)
        lines = findLines(miniCons)

        for i in range(-1, len(lines)-1):
            cv.line(img, lines[i].first.toIntArr(), lines[i+1].first.toIntArr(), (100,100,100), 1)
            cv.circle(img, lines[i+1].first.toIntArr(), 2, (255, 100, 100), 2)
        
        segments = splitIntoSegments(img, lines)
        for seg in segments:
            cv.line(img, seg.parts[0].first.toIntArr(), seg.parts[-1].last.toIntArr(), (0,0,0), 1)
            cv.circle(img, seg.parts[0].first.toIntArr(), 3, (0, 150, 250), 2)

        for seg in segments:
            circles = findCircles(seg)

            if len(circles) > 0:
                circleAreas.extend(CircleArea.getCirclesAreas(img, columns, circles))

    #checkNeighbourCircles(circles)

    for col in columns:
        cv.circle(img, col.toIntArr(), 3, (0, 0, 150), 5)

    cv.imwrite(out, img)

    if out == "test.png":
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', "test.png"))
        elif platform.system() == 'Windows':    # Windows
            os.startfile("test.png")
        else:                                   # linux variants
            subprocess.call(('xdg-open', "test.png"))


def checkNeighbourCircles(circles):
    neighbourCircles = []

    for i in range(len(circles)):
        for j in range(len(circles)):
            if i == j:
                continue

            if circles[i].endCorner == circles[i].startCorner:
                neighbourCircles.append

def test(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    #fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    extractPartsAndWalls(fn_slab, "testOutput/" + str(num) + ".png")

def selectedTest(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    extractPartsAndWalls(fn_slab, fn_column, "selectedOutput/" + str(num) + ".png")

    print("Finshed: " + str(num))

if __name__ == "__main__":
    #print("OpenCV version:", cv.__version__)
    #testHoughLine()
    #testHoughCircle()
    #testHarrisCorners()

    #nums = np.arange(1,1111)

    #with Pool(16) as p:
    #    p.map(test, nums)
    
    nums = [87, 94, 114, 177, 403, 476, 661, 673]

    with Pool(8) as p:
        p.map(selectedTest, nums)
    
    #extractPartsAndWalls("../Dataset/02_sl/ZB_0046_02_sl.png", "../Dataset/03_co/ZB_0046_03_co.png")


    # 403 673 
    #extractPartsAndWalls("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    extractPartsAndWalls("../Dataset/Selected/ZB_0094_02_sl.png", "../Dataset/Selected/ZB_0094_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0114_02_sl.png", "../Dataset/Selected/ZB_0114_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0177_02_sl.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0403_02_sl.png", "../Dataset/Selected/ZB_0403_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0476_02_sl.png", "../Dataset/Selected/ZB_0476_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0661_02_sl.png", "../Dataset/Selected/ZB_0661_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0673_02_sl.png")

