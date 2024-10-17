import cv2 as cv
import numpy as np
import math
import subprocess, os, platform
from multiprocessing import Pool
from extractor.vec import Vec2
from extractor.area import CircleArea
from extractor.contour import Contour
from extractor.formfinder import findLines, splitIntoSegments, findCircles, findCorners
from extractor.forms import Segment, Line
from extractor.circle import Circle
from extractor.pointmath import PMath
from extractor.linewall import LineWall

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
    dwg.viewbox(minx=2900, miny=2900, width=img.shape[0], height=img.shape[1])

    absPath = os.path.abspath(imgFile)
    dwg.add(dwg.image(absPath, insert=(-0.5, -0.5), size=img.shape))

    columns = getColumnCenter(columnImg)
    lineWalls = []
    areas = []

    contours = contour.getContour(img)
    # for con in contours:
    #     print(len(con))
    #     col = 0
    #     for point in con:
    #         #col = int(255 - point[2])
    #         c = "rgb({},{},{})".format(col, col, col)
    #         col = (col + 20) % 250
    #         dwg.add(dwg.circle(center=point[:2], r=1, fill=c)) #fill="rgb(0,200,200)"))

    for con in contours:
        parts = contour.getContourParts(con)
        for part in parts:
            dwg.add(dwg.circle(center=con[part[0]][:2], r=2, fill="rgb(250,100,100)"))
        #    dwg.add(dwg.line(start=con[part[0]][:2], end=con[part[1]][:2], stroke="rgb(0,200,0)", stroke_width=0.5))

        parts = contour.fixCorners(con, parts)
        #for part in parts:
        #    dwg.add(dwg.circle(center=con[part[0]][:2], r=0.5, fill="rgb(0,154,178)"))

        segments = contour.splitIntoSegments(con, parts)
        #for seg in segments:
        #    dwg.add(dwg.circle(center= con[parts[seg[0]][0]][:2], r=1, fill="rgb(150,0,0)"))

        circleAreas = []
        for seg in segments:
            circles, lines = contour.findCirclesAndLines(con, parts, seg[0], seg[1])

            #for lin in lines:
            #    dwg.add(dwg.line(start=lin[:2], end=lin[2:4], stroke="rgb(200,0, 200)", stroke_width=2))
            
            #for c in circles:
            #    dwg.add(dwg.circle(center=c[6:8], r=c[8], stroke="rgb(0,154,178)", stroke_width=1, fill="none"))

            lines = Line.convArr(lines)
            lineWalls.extend(lines)

            circles = Circle.convArr(circles)
            if len(circles) > 0:
                circleAreas.extend(CircleArea.getCirclesAreas(columns, circles))

        CircleArea.checkNeighboringCircleAreas(circleAreas)
        areas.extend(circleAreas)


    #for lin in lineWalls:
    #    dwg.add(dwg.line(start=lin.first.toArr(), end=lin.last.toArr(), stroke="rgb(150,0,150)", stroke_width=5))

    paraLines = []
    walls = []

    for lin in lineWalls:
        if lin.distinctiveWall or lin.length() > 500:
            a = True
            for i in range(len(paraLines)):
                if PMath.isAlmostParallel(lin.first, lin.last, paraLines[i].first, paraLines[i].last):
                    a = False
            
            if a:
                paraLines.append(lin)
                #dwg.add(dwg.line(start=lin.first.toArr(), end=lin.last.toArr(), stroke="rgb(50,100,250)", stroke_width=20))


        #dwg.add(dwg.line(start=lin.first.toArr(), end=lin.last.toArr(), stroke="rgb(150,150,150)", stroke_width=1))
        walls.append(LineWall(LineWall.HARD_WALL, start=lin.first, end=lin.last))




    columnLine = []
    keepColumn = [[col, True] for col in columns]

    for i in range(len(columns)):
        for j in range(i+1, len(columns)):

            line = (columns[i], columns[j], i, j)
            for para in paraLines:
                if PMath.isAlmostParallel(para.first, para.last, line[0], line[1]): 
                    columnLine.append(line) 
                    #dwg.add(dwg.line(start=line[0].toArr(), end=line[1].toArr(), stroke="rgb(150,50,50)", stroke_width=10))



    for i in range(len(columnLine)):
        for j in range(i+1, len(columnLine)):

            if PMath.distancePointToLine(columnLine[i][0], columnLine[i][1], columnLine[j][0]) < 3 and PMath.distancePointToLine(columnLine[i][0], columnLine[i][1], columnLine[j][1]) < 3: 
                if columnLine[i][0] == columnLine[j][0] or columnLine[i][1] == columnLine[j][1] or columnLine[i][0] == columnLine[j][1] or columnLine[i][1] == columnLine[j][0]:
                    keepColumn[columnLine[i][2]][1] = False
                    keepColumn[columnLine[i][3]][1] = False
                    keepColumn[columnLine[j][2]][1] = False
                    keepColumn[columnLine[j][3]][1] = False

                    #dwg.add(dwg.line(start=columnLine[i][0].toArr(), end=columnLine[i][1].toArr(), stroke="rgb(0,100,150)", stroke_width=10))
                    #dwg.add(dwg.line(start=columnLine[j][0].toArr(), end=columnLine[j][1].toArr(), stroke="rgb(0,100,150)", stroke_width=10))




    for area in areas:
        walls.extend(area.getWalls())

    for area in areas:
        area.findCurves(keepColumn, walls)

    for area in areas:
        area.drawArea(dwg, 1)


    for col in keepColumn:
        if col[1]:
            dwg.add(dwg.circle(center=col[0].toArr(), r=5, fill="rgb(150,0,0)"))
        else:
            dwg.add(dwg.circle(center=col[0].toArr(), r=5, fill="rgb(100,0,0)"))

    zoom_script = """
        var svgElement = document.documentElement;
        var zoomLevel = 1;
        var viewBox = [2900, 2900, 5905, 5905];

        var isDragging = false;
        var dragStart = { x: 0, y: 0 };
        var viewBoxStart = [0, 0];

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

        // Handle mouse down event to start dragging
        svgElement.addEventListener('mousedown', function(event) {
            isDragging = true;
            dragStart.x = event.clientX;
            dragStart.y = event.clientY;
            viewBoxStart = [...viewBox];
        });

        // Handle mouse move event to drag the viewBox
        svgElement.addEventListener('mousemove', function(event) {
            if (isDragging) {
                var dx = (event.clientX - dragStart.x) * (viewBoxStart[2] / svgElement.clientWidth) / 5;
                var dy = (event.clientY - dragStart.y) * (viewBoxStart[3] / svgElement.clientHeight) / 5;

                viewBox[0] = viewBoxStart[0] - dx;
                viewBox[1] = viewBoxStart[1] - dy;

                svgElement.setAttribute('viewBox', viewBox.join(' '));
            }
        });

        // Handle mouse up event to stop dragging
        svgElement.addEventListener('mouseup', function() {
            isDragging = false;
        });

        // Handle mouse leave event to stop dragging if mouse leaves the SVG area
        svgElement.addEventListener('mouseleave', function() {
            isDragging = false;
        });
    """
    dwg.add(dwg.script(content=zoom_script, type="text/ecmascript"))

    # Save the SVG file
    dwg.save()
    print("Done.")

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
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0114_07_os.png", "../Dataset/03_co/ZB_0114_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0177_07_os.png", "../Dataset/03_co/ZB_0177_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0403_07_os.png", "../Dataset/03_co/ZB_0403_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0476_07_os.png", "../Dataset/03_co/ZB_0476_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0661_07_os.png", "../Dataset/03_co/ZB_0661_03_co.png")
    #extractPartsAndWalls2("../Dataset/07_os/ZB_0673_07_os.png", "../Dataset/03_co/ZB_0673_03_co.png")

    #extractPartsAndWalls2("../Dataset/Selected/Test.png", "../Dataset/Selected/ZB_0087_03_co.png")

    #num = "0005"
    #extractPartsAndWalls2("../Dataset/07_os/ZB_{}_07_os.png".format(num), "../Dataset/03_co/ZB_{}_03_co.png".format(num))


    nums = [87, 94, 114, 177, 403, 476, 661, 673]
    #nums = [403, 661, 673]

    with Pool(8) as p:
        p.map(selectedTest, nums)