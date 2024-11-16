"""
This script demonstrates how to use OpenCV (cv2) for various image processing tasks.
"""
from multiprocessing import Pool
import os
import cv2

import svgwrite
import contour

from extractor.area import CircleArea
from extractor.forms import Line
from extractor.circle import Circle
from extractor.pointmath import PMath
from extractor.linewall import LineWall
from extractor.column import getColumnCenter


def selected_test(num):
    in_path = "../Dataset/"
    prefix = 'ZB_' if num > 999 else 'ZB_0' if num > 99 else 'ZB_00' if num > 9 else 'ZB_000'
    fn_column = in_path + "03_co/" + prefix + str(num) + '_03_co.png'
    fn_layout = in_path + "07_os/" + prefix + str(num) + '_07_os.png'
    extract_walls(fn_layout, fn_column, "selectedOutput/" + str(num) + ".svg")
    print("Finshed: " + str(num))


def extract_walls(img_file, column_img, out="test.svg"):
    img_col = cv2.imread(img_file)
    if img_col is None:
        return

    img = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    dwg = svgwrite.Drawing(out, size=img.shape)
    dwg.viewbox(minx=2900, miny=2900, width=img.shape[0], height=img.shape[1])

    abs_path = os.path.abspath(img_file)
    dwg.add(dwg.image(abs_path, insert=(-0.5, -0.5), size=img.shape))

    columns = getColumnCenter(column_img)
    line_walls = []
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
        # for part in parts:
        #    dwg.add(dwg.circle(center=con[part[0]][:2], r=0.5, fill="rgb(0,154,178)"))

        segments = contour.splitIntoSegments(con, parts)
        # for seg in segments:
        #    dwg.add(dwg.circle(center= con[parts[seg[0]][0]][:2], r=1, fill="rgb(150,0,0)"))

        circle_areas = []
        for seg in segments:
            circles, lines = contour.findCirclesAndLines(con, parts, seg[0], seg[1])

            # for lin in lines:
            #    dwg.add(dwg.line(start=lin[:2], end=lin[2:4], stroke="rgb(200,0, 200)", stroke_width=2))

            # for c in circles:
            #    dwg.add(dwg.circle(center=c[6:8], r=c[8], stroke="rgb(0,154,178)", stroke_width=1, fill="none"))

            lines = Line.convArr(lines)
            line_walls.extend(lines)

            circles = Circle.convArr(circles)
            if len(circles) > 0:
                circle_areas.extend(CircleArea.getCirclesAreas(columns, circles))

        CircleArea.checkNeighboringCircleAreas(circle_areas)
        areas.extend(circle_areas)

    # for lin in lineWalls:
    #    dwg.add(dwg.line(start=lin.first.toArr(), end=lin.last.toArr(), stroke="rgb(150,0,150)", stroke_width=5))

    para_lines = []
    walls = []

    for lin in line_walls:
        if lin.distinctiveWall or lin.length() > 500:
            a = True
            for plin in para_lines:
                if PMath.isAlmostParallel(lin.first, lin.last, plin.first, plin.last):
                    a = False

            if a:
                para_lines.append(lin)
                dwg.add(dwg.line(lin.first.toArr(), lin.last.toArr(), stroke="rgb(50,100,250)", stroke_width=20))

        # dwg.add(dwg.line(start=lin.first.toArr(), end=lin.last.toArr(), stroke="rgb(150,150,150)", stroke_width=1))
        walls.append(LineWall(LineWall.HARD_WALL,
                     start=lin.first, end=lin.last))

    column_line = []
    keep_column = [[col, True] for col in columns]

    for coli, i in enumerate(columns):
        for colj, j in enumerate(columns, i+1):

            line = (coli, colj, i, j)
            for para in para_lines:
                if PMath.isAlmostParallel(para.first, para.last, line[0], line[1]):
                    column_line.append(line)
                    # dwg.add(dwg.line(start=line[0].toArr(), end=line[1].toArr(), stroke="rgb(150,50,50)", stroke_width=10))

    for coli, i in enumerate(column_line):
        for colj, j in enumerate(column_line, i+1):
            if PMath.distancePointToLine(coli[0], coli[1], colj[0]) < 3 \
                and PMath.distancePointToLine(coli[0], coli[1], colj[1]) < 3:

                if coli[0] == colj[0] or coli[1] == colj[1] or coli[0] == colj[1] or coli[1] == colj[0]:
                    keep_column[column_line[i][2]][1] = False
                    keep_column[column_line[i][3]][1] = False
                    keep_column[column_line[j][2]][1] = False
                    keep_column[column_line[j][3]][1] = False

                    # dwg.add(dwg.line(start=columnLine[i][0].toArr(), end=columnLine[i][1].toArr(), stroke="rgb(0,100,150)", stroke_width=10))
                    # dwg.add(dwg.line(start=columnLine[j][0].toArr(), end=columnLine[j][1].toArr(), stroke="rgb(0,100,150)", stroke_width=10))

    for area in areas:
        walls.extend(area.getWalls())

    for area in areas:
        area.findCurves(keep_column, walls)

    for area in areas:
        area.drawArea(dwg, 1)

    for col in keep_column:
        if col[1]:
            dwg.add(dwg.circle(
                center=col[0].toArr(), r=5, fill="rgb(150,0,0)"))
        else:
            dwg.add(dwg.circle(
                center=col[0].toArr(), r=5, fill="rgb(100,0,0)"))

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

    # extract_walls("../Dataset/07_os/ZB_0087_07_os.png", "../Dataset/03_co/ZB_0087_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0094_07_os.png", "../Dataset/03_co/ZB_0094_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0114_07_os.png", "../Dataset/03_co/ZB_0114_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0177_07_os.png", "../Dataset/03_co/ZB_0177_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0403_07_os.png", "../Dataset/03_co/ZB_0403_03_co.png")
    extract_walls("../Dataset/07_os/ZB_0476_07_os.png", "../Dataset/03_co/ZB_0476_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0661_07_os.png", "../Dataset/03_co/ZB_0661_03_co.png")
    # extract_walls("../Dataset/07_os/ZB_0673_07_os.png", "../Dataset/03_co/ZB_0673_03_co.png")

    # extract_walls("../Dataset/Selected/Test.png", "../Dataset/Selected/ZB_0087_03_co.png")

    # num = "0005"
    # extract_walls("../Dataset/07_os/ZB_{}_07_os.png".format(num), "../Dataset/03_co/ZB_{}_03_co.png".format(num))

    # nums = [87, 94, 114, 177, 403, 476, 661, 673]
    # nums = [403, 661, 673]

    # with Pool(8) as p:
    #    p.map(selected_test, nums)
