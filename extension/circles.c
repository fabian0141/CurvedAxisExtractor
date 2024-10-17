#include "circles.h"
#include <numpy/arrayobject.h>

#include "line.h"
#include "circle.h"

int getCircle(double *data, int* parts, int start, int end, Circle* circle) {
    double pointDist1 = pDist(data, parts, start*3, (end + start) / 2 * 3);
    double pointDist2 = pDist(data, parts, start*3, end*3);
    double m = 1;
    if (pointDist1 > pointDist2 * 1.5)
        m = 1.5;

    Point p1 = toPoint(data, parts[start*3]*3);
    Point p2 = toPoint(data, parts[(int)(start + (end-start) / (2*m))*3]*3);
    Point p3 = toPoint(data, parts[(int)(start + (end-start) / m)*3+1]*3);

    double a[3][3] = {
        {2*p1.x, 2*p1.y, 1},
        {2*p2.x, 2*p2.y, 1},
        {2*p3.x, 2*p3.y, 1}
    };

    double b[3] = {
        -(p1.x*p1.x) - p1.y*p1.y,
        -(p2.x*p2.x) - p2.y*p2.y,
        -(p3.x*p3.x) - p3.y*p3.y
    };

    if (fabs(determinant(a)) < 0.0000001)
        return 0;

    Point middlePoint;
    double radius;
    circleLGS(a, b, &middlePoint, &radius);

    if (radius < 300 || radius > 3000)
        return 0;

    circle->start = p1;
    circle->between = p2;
    circle->end = toPoint(data, parts[end*3+1]*3);
    circle->middle = middlePoint;
    circle->radius = radius;
    return 1;
}

void findPrimitives(double *data, int* parts, int partsLength, int firstPart, int lastPart, Circle* circles, int *circlesLength, LineSegment* lines, int* linesLength) {

    Circle circle;
    int isSuitable = 0;
    *circlesLength = 0;
    *linesLength = 0;
    int start;
    int i = firstPart;

    for (; i < lastPart-3; i++) {
        if (!isSuitable) {

            isSuitable = getCircle(data, parts, i, i+4, &circle);
            if (!isSuitable || !isCircleValid(&circle, data, parts, i, i+4)) { // no suitable circle found or not valid
                addLine(lines, (*linesLength)++, data, parts, i);
                isSuitable = 0;

            } else { // circle found
                circles[(*circlesLength)++] = circle;
                start = i;
            }
        } else {
            isSuitable = getCircle(data, parts, start, i+4, &circle);
            if (!isSuitable) { // remove circle because not suitable anymore
                (*circlesLength)--;
                i += 2;
                addLines(lines, linesLength, data, parts, start, i, linesLength);

            } else if (isCircleValid(&circle, data, parts, start, i+4)) {
                circles[(*circlesLength)-1] = circle;
            } else {
                isSuitable = 0;
                i += 2;
                addLine(lines, (*linesLength)++, data, parts, i);
            }
        }
    }

    if (circlesLength == 0) {
        if (linesLength == 1 && lineLength(data, parts, 0) > 100) {
            lines[0].distincitveWall = 1;
        }
        return;
    }
    addLines(lines, linesLength, data, parts, i, lastPart);
}

PyObject* primitivesToPyArray(Circle* circles, int circlesLength, LineSegment* lines, int linesLength) {
    
    npy_intp dims[] = {circlesLength, 9};
    PyObject* pyCircles = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    double* data = (double*)PyArray_DATA((PyArrayObject*)pyCircles);

    int idx = 0;
    for (int i = 0; i < circlesLength; i++) {
        data[idx++] = circles[i].start.x;
        data[idx++] = circles[i].start.y;
        data[idx++] = circles[i].between.x;
        data[idx++] = circles[i].between.y;
        data[idx++] = circles[i].end.x;
        data[idx++] = circles[i].end.y;
        data[idx++] = circles[i].middle.x;
        data[idx++] = circles[i].middle.y;
        data[idx++] = circles[i].radius;
    }

    dims[0] = linesLength;
    dims[1] = 5;
    PyObject* pyLines = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    data = (double*)PyArray_DATA((PyArrayObject*)pyLines);

    idx = 0;
    for (int i = 0; i < linesLength; i++) {
        data[idx++] = lines[i].p1.x;
        data[idx++] = lines[i].p1.y;
        data[idx++] = lines[i].p2.x;
        data[idx++] = lines[i].p2.y;
        data[idx++] = lines[i].distincitveWall;
    }
    
    free(circles);
    free(lines);
    return PyTuple_Pack(2, pyCircles, pyLines);
}

PyObject* findCirclesAndLines(PyObject *self, PyObject *args) {
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    int firstPart;
    int lastPart;

    if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &arr1, &PyArray_Type, &arr2, &firstPart, &lastPart)) {
        PyErr_SetString(PyExc_TypeError, "Expected two numpy arrays and two ints");
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 2 || PyArray_TYPE(arr1) != NPY_FLOAT64 || PyArray_NDIM(arr2) != 2 || PyArray_TYPE(arr2) != NPY_INT) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type double.");
        return NULL;
    }

    double *data = PyArray_DATA(arr1);
    int dataSize = PyArray_SHAPE(arr1)[0] * 3;

    int *parts = PyArray_DATA(arr2);
    int partsLength = PyArray_SHAPE(arr2)[0];

    Circle* circles = malloc(partsLength * sizeof(Circle));
    int circlesLength;

    LineSegment* lines = malloc(partsLength * sizeof(LineSegment));
    int linesLength;

    findPrimitives(data, parts, partsLength, firstPart, lastPart, circles, &circlesLength, lines, &linesLength);
    return primitivesToPyArray(circles, circlesLength, lines, linesLength);
}

void initCircles() {
    import_array();
}