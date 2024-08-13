import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool
from extractor.vec import Vec2
from extractor.area import CircleArea
from extractor.contour import Contour
from extractor.formfinder import findLines, splitIntoSegments, findCircles, findCorners
from extractor.forms import Segment
from extractor.circle import Circle
from extractor.pointmath import PMath

from extractor.column import getColumnCenter
import svgwrite
import contour


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
    walls = []

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
            circles, lin = findCircles(seg)
            lines.extend(lin)

            if len(circles) > 0:
                circleAreas.extend(CircleArea.getCirclesAreas(img, columns, circles))

    CircleArea.checkNeighboringCircleAreas(circleAreas, img)
    
    for area in circleAreas:
        walls.extend(area.getWalls())

    for area in circleAreas:
        area.findCurves(columns, walls)


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
        if PMath.segmentsIntersection(test[0], c1.allignedMiddle, test[1], c2.allignedMiddle) is not None:
            return True
        
    return False

def test(num):
    print(num)
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    #fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'

    extractPartsAndWalls(fn_slab, "testOutput/" + str(num) + ".png")

def selectedTest(num, svg=True):
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_slab = in_path + "02_sl/" + prefix + str(num) + '_02_sl.png'
    fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'
    fn_layout = in_path + "07_os/" + prefix + str(num) + '_07_os.png'

    if svg:
        extractPartsAndWalls2(fn_layout, fn_column, "selectedOutput/" + str(num) + ".svg")
    else:
        extractPartsAndWalls(fn_slab, fn_column, fn_layout, "selectedOutput/" + str(num) + ".png")

    print("Finshed: " + str(num))

def testContour(imgFile):
    imgCol = cv.imread(imgFile)
    if imgCol is None:
        return

    img = cv.cvtColor(imgCol, cv.COLOR_BGR2GRAY)
    dwg = svgwrite.Drawing('test.svg', size = img.shape)
    dwg.viewbox(minx=0, miny=0, width=img.shape[0], height=img.shape[1])

    dwg.add(dwg.image(imgFile, insert=(0, 0), size=img.shape))


    points = contour.testContour(img)

    #for point in points[:5000]:
    #    c = int(point[2])
    #    col = "rgb({},{},{})".format(255,c,255)
    #    dwg.add(dwg.circle(center=point[:2], r=0.5, fill=col))

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
        dwg.add(dwg.circle(center=point[:2] + [0.5, 0.5], r=1, fill="rgb(0,200,200)"))
        

    zoom_script = """
        var svgElement = document.documentElement;
        var zoomLevel = 1;
        var viewBox = [0, 0, 5905, 5905];

        svgElement.addEventListener('wheel', function(event) {
            event.preventDefault();

            var mouseX = event.clientX;
            var mouseY = event.clientY;
            var svgRect = svgElement.getBoundingClientRect();
            var svgX = mouseX - svgRect.left;
            var svgY = mouseY - svgRect.top;

            var viewBoxX = viewBox[0];
            var viewBoxY = viewBox[1];
            var viewBoxWidth = viewBox[2];
            var viewBoxHeight = viewBox[3];

            var scale = (event.deltaY > 0) ? 1.1 : 0.9;

            var newViewBoxWidth = viewBoxWidth * scale;
            var newViewBoxHeight = viewBoxHeight * scale;

            var dx = (svgX / svgRect.width) * (newViewBoxWidth - viewBoxWidth);
            var dy = (svgY / svgRect.height) * (newViewBoxHeight - viewBoxHeight);

            viewBox[0] = viewBoxX - dx;
            viewBox[1] = viewBoxY - dy;
            viewBox[2] = newViewBoxWidth;
            viewBox[3] = newViewBoxHeight;

            svgElement.setAttribute('viewBox', viewBox.join(' '));
        }, { passive: false });
    """
    dwg.add(dwg.script(content=zoom_script, type="text/ecmascript"))

    # Save the SVG file
    dwg.save()

def extractPartsAndWalls2(imgFile, columnImg, out="test.svg"):
    imgCol = cv.imread(imgFile)
    if imgCol is None:
        return

    img = cv.cvtColor(imgCol, cv.COLOR_BGR2GRAY)
    dwg = svgwrite.Drawing(out, size = img.shape)
    dwg.viewbox(minx=0, miny=0, width=img.shape[0], height=img.shape[1])

    absPath = os.path.abspath(imgFile)
    dwg.add(dwg.image(absPath, insert=(0, 0), size=img.shape))

    points = contour.getContour(img)
    print(len(points))
    for point in points:
        if point[2] < 0:
            continue

        col = int(255 - point[2])

        c = "rgb(255,{},{})".format(col, col)

        dwg.add(dwg.circle(center=point[:2] + [0.5, 0.5], r=1, fill=c)) #fill="rgb(0,200,200)"))

    # columns = getColumnCenter(columnImg)
    # circleAreas = []
    # walls = []

    # contours = Contour.convertContour(points)
    # for points in contours:
    #     miniCons = Contour.getContourParts(points, img)
    #     for con in miniCons:
    #         dwg.add(dwg.circle(center=con.first.toArr(), r=1.5, fill="rgb(150,150,150)"))
        
    #     lines = findCorners(miniCons)
    #     for i in range(-1, len(lines)-1):
    #         dwg.add(dwg.circle(center=lines[i+1].first.toArr(), r=3, fill="rgb(150,200,250)"))

    #     segments = splitIntoSegments(img, lines)
    #     for seg in segments:
    #         dwg.add(dwg.circle(center=seg.parts[0].first.toArr(), r=1, fill="rgb(150,0,0)"))

    #     for seg in segments:
    #         circles, lin = findCircles(seg)
    #         lines.extend(lin)
    #         #for c in circles:
    #         #    c.drawOutline(dwg, 1)

    #         if len(circles) > 0:
    #             circleAreas.extend(CircleArea.getCirclesAreas(img, columns, circles))

    # CircleArea.checkNeighboringCircleAreas(circleAreas, img)
    
    # for area in circleAreas:
    #     walls.extend(area.getWalls())

    # for area in circleAreas:
    #     area.findCurves(columns, walls)


    # for area in circleAreas:
    #     area.drawArea(dwg, 3)

    # for col in columns:
    #     dwg.add(dwg.circle(center=col.toArr(), r=5, fill="rgb(150,0,0)"))

    zoom_script = """
        var svgElement = document.documentElement;
        var zoomLevel = 1;
        var viewBox = [0, 0, 5905, 5905];

        svgElement.addEventListener('wheel', function(event) {
            event.preventDefault();

            var mouseX = event.clientX;
            var mouseY = event.clientY;
            var svgRect = svgElement.getBoundingClientRect();
            var svgX = mouseX - svgRect.left;
            var svgY = mouseY - svgRect.top;

            var viewBoxX = viewBox[0];
            var viewBoxY = viewBox[1];
            var viewBoxWidth = viewBox[2];
            var viewBoxHeight = viewBox[3];

            var scale = (event.deltaY > 0) ? 1.1 : 0.9;

            var newViewBoxWidth = viewBoxWidth * scale;
            var newViewBoxHeight = viewBoxHeight * scale;

            var dx = (svgX / svgRect.width) * (newViewBoxWidth - viewBoxWidth);
            var dy = (svgY / svgRect.height) * (newViewBoxHeight - viewBoxHeight);

            viewBox[0] = viewBoxX - dx;
            viewBox[1] = viewBoxY - dy;
            viewBox[2] = newViewBoxWidth;
            viewBox[3] = newViewBoxHeight;

            svgElement.setAttribute('viewBox', viewBox.join(' '));
        }, { passive: false });
    """
    dwg.add(dwg.script(content=zoom_script, type="text/ecmascript"))

    # Save the SVG file
    dwg.save()

if __name__ == "__main__":
    
    #nums = [87, 94, 114, 177, 403, 476, 661, 673]

    #with Pool(8) as p:
    #    p.map(selectedTest, nums)

    #extractPartsAndWalls("../Dataset/02_sl/ZB_0087_02_sl.png", "../Dataset/03_co/ZB_0087_03_co.png", "../Dataset/07_os/ZB_0087_07_os.png")
    #extractPartsAndWalls("../Dataset/02_sl/ZB_0403_02_sl.png", "../Dataset/03_co/ZB_0403_03_co.png", "../Dataset/07_os/ZB_0403_07_os.png")
    #extractPartsAndWalls("../Dataset/02_sl/ZB_0476_02_sl.png", "../Dataset/03_co/ZB_0476_03_co.png", "../Dataset/07_os/ZB_0476_07_os.png")


    #extractPartsAndWalls("../Dataset/Selected/ZB_0087_02_sl.png", "../Dataset/Selected/ZB_0087_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0094_02_sl.png", "../Dataset/Selected/ZB_0094_03_co.png", "../Dataset/07_os/ZB_0094_07_os.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0114_02_sl.png", "../Dataset/Selected/ZB_0114_03_co.png")
    # extractPartsAndWalls("../Dataset/Selected/ZB_0177_02_sl.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0403_02_sl.png", "../Dataset/Selected/ZB_0403_03_co.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0476_02_sl.png", "../Dataset/Selected/ZB_0476_03_co.png", "../Dataset/Selected/ZB_0476_07_os.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0661_02_sl.png", "../Dataset/Selected/ZB_0661_03_co.png", "../Dataset/Selected/ZB_0661_07_os.png")
    #extractPartsAndWalls("../Dataset/Selected/ZB_0673_02_sl.png", "../Dataset/Selected/ZB_0673_03_co.png", "../Dataset/Selected/ZB_0673_07_os.png")

    #extractPartsAndWalls2("../Dataset/07_os/ZB_0087_07_os.png", "../Dataset/03_co/ZB_0087_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0094_07_os.png", "../Dataset/03_co/ZB_0094_03_co.png")
    extractPartsAndWalls2("../Dataset/07_os/ZB_0114_07_os.png", "../Dataset/03_co/ZB_0114_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0177_07_os.png", "../Dataset/03_co/ZB_0177_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0403_07_os.png", "../Dataset/03_co/ZB_0403_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0476_07_os.png", "../Dataset/03_co/ZB_0476_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0661_07_os.png", "../Dataset/03_co/ZB_0661_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0673_07_os.png", "../Dataset/03_co/ZB_0673_03_co.png")

    #nums = [87, 94, 114, 177, 403, 476, 661, 673]
    #nums = [87]
    #with Pool(8) as p:
    #   p.map(selectedTest, nums)