import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool
from extractor.vec import Vec2
from extractor.area import CircleArea
from extractor.contour import Contour
from extractor.formfinder import findLines, splitIntoSegments, findCircles
from extractor.forms import Segment
from extractor.circle import Circle

from extractor.column import getColumnCenter
#import svgwrite
#import contour


def extractPartsAndWalls(imgFile, columnImg, layoutImg, out = "test.png"):
    #load image
    img = cv.imread(imgFile)
    if img is None:
        return
    
    layout = cv.imread(layoutImg)
    if layout is None:
        return

    imgray = cv.cvtColor(layout, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    contours2, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contours = contours2[:2]
    if len(contours2) > 3:
        contours += contours2[4:5]
    cv.drawContours(img, contours, -1, (155,155,0), 1)

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

    CircleArea.checkNeighboringCircleAreas(circleAreas, img)

    for area in circleAreas:
        area.drawArea(img, 3)

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

def circleLinesIntersect(c1, c2):
    tests = [
        [c1.start,c2.start],
        [c1.end, c2.start],
        [c1.start, c2.end],
        [c1.end, c2.end],
    ]

    for test in tests:
        if linesIntersection(test[0], c1.allignedMiddle, test[1], c2.allignedMiddle) is not None:
            return True
        
    return False

def linesIntersection(p1, p2, q1, q2):
    v1 = p1 - p2
    v2 = q1 - q2
    v3 = p1 - q1

    denom = v1.x * v2.y - v1.y * v2.x

    t = (v3.x * v2.y - v3.y * v2.x) / denom
    u = -(v1.x * v3.y - v1.y * v3.x) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 - v1*t

    return None

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
    fn_layout = in_path + "07_os/" + prefix + str(num) + '_07_os.png'

    extractPartsAndWalls(fn_slab, fn_column, fn_layout, "selectedOutput/" + str(num) + ".png")

    print("Finshed: " + str(num))

def testContour(imgFile):
    imgCol = cv.imread(imgFile)
    if imgCol is None:
        return

    img = cv.cvtColor(imgCol, cv.COLOR_BGR2GRAY)
    dwg = svgwrite.Drawing('test.svg', size = img.shape)
    
    points = contour.testContour(img)
    group = dwg.g(id='content', transform='scale(3) translate(-1720, -1700)')
    dwg.add(group)

    #for point in points:
    #    c = int(point[2])
    #    col = "rgb({},{},{})".format(255,c,255)
    #    group.add(dwg.circle(center=point[:2], r=1.0, fill=col))

    points = contour.getContour(img)
    print(len(points))

    counter = 0
    contours = Contour.convertContour(points)

    for point in points:
        if point[2] < 0:
            continue
        c = int(point[2])

        #col = "rgb({},{},{})".format(counter,counter,counter)
        col = "rgb({},{},{})".format(c,c,c)

        counter = (counter+30)%256
        group.add(dwg.circle(center=point[:2], r=0.5, fill=col))


    # Save the SVG file
    dwg.save()

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

    extractPartsAndWalls("../Dataset/02_sl/ZB_0087_02_sl.png", "../Dataset/03_co/ZB_0087_03_co.png", "../Dataset/07_os/ZB_0087_07_os.png")
    #extractPartsAndWalls("../Dataset/02_sl/ZB_0403_02_sl.png", "../Dataset/03_co/ZB_0403_03_co.png", "../Dataset/07_os/ZB_0403_07_os.png")
    #extractPartsAndWalls("../Dataset/02_sl/ZB_0476_02_sl.png", "../Dataset/03_co/ZB_0476_03_co.png", "../Dataset/07_os/ZB_0476_07_os.png")


    # 403 673 
    #extractPartsAndWalls("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0094_02_sl.png", "../Dataset/Selected/ZB_0094_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0114_02_sl.png", "../Dataset/Selected/ZB_0114_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0177_02_sl.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0403_02_sl.png", "../Dataset/Selected/ZB_0403_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0476_02_sl.png", "../Dataset/Selected/ZB_0476_03_co.png", "../Dataset/Selected/ZB_0476_07_os.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0661_02_sl.png", "../Dataset/Selected/ZB_0661_03_co.png", "../Dataset/Selected/ZB_0661_07_os.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0673_02_sl.png", "../Dataset/Selected/ZB_0673_03_co.png", "../Dataset/Selected/ZB_0673_07_os.png")

    #testContour("../Dataset/Selected/ZB_0673_07_os.png")
    #testContour("../Dataset/07_os/ZB_0087_07_os.png")
    #testContour("../Dataset/07_os/ZB_0476_07_os.png")

    #arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    #arr2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    #result = contour.add(arr1, arr2)
    #print(f"Result of addition: {result}")