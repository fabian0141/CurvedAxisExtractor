#include "findcorner.h"
#include <numpy/arrayobject.h>

#include "line.h"
#include "point.h"

double dist3(double* data, int idx, Point p) {
    double x = data[idx] - p.x;
    double y = data[idx+1] - p.y;
    return sqrt(x*x + y*y);
}

void findCorners(double *data, int dataSize, Line** root, int* partLength) {
    int improveCorner = 0;
    Line* part = *root;

    do {
        if (part->last - part->first <= 6) {
            improveCorner = part != *root;
            Line* next = part->next;
            *root = removeLine(*root, part);

            part = next;
            (*partLength)--;
            continue;
        }


        if (improveCorner) {
            Line* prev = part->prev;
            Point inter = linesIntersection(data, prev, part);

            if (inter.x < 0 || dist3(data, prev->last, inter) > 10) { // lines are parallel
                Point middle = pMiddle(data, prev->last, part->first);

                Point p = closestPointOnLine(data, prev->first, prev->last, middle);
                updateData(data, prev->last, p);
                p = closestPointOnLine(data, part->first, part->last, middle);
                updateData(data, part->first, p);

                insertLine(prev);
                (*partLength)++;
                improveCorner = 0;
                part = part->next;

            } else {
                for (int i = prev->last; i <= part->first; i += 3) {
                    updateData(data, i, inter);
                }
                improveCorner = 0;
            }
        }
        part = part->next;
    } while (part != NULL && part->next != NULL);

    if (part != NULL && part->last - part->first <= 6) {
        Line* prev = part->prev;
        *root = removeLine(*root, part);
        (*partLength)--;


        part = *root;
        Point inter = linesIntersection(data, prev, part);

        if (inter.x < 0 || dist3(data, prev->last, inter) > 10) { // lines are parallel
            Point middle = pMiddle(data, prev->last, part->first);

            Point p = closestPointOnLine(data, prev->first, prev->last, middle);
            updateData(data, prev->last, p);
            p = closestPointOnLine(data, part->first, part->last, middle);
            updateData(data, part->first, p);

            insertLine(prev);
            (*partLength)++;
            part = part->next;

        } else {
            for (int i = prev->last; i <= part->first; i += 3)
            {
                updateData(data, i, inter);
            }
        }
    }
}

PyObject* updateParts(Line* part, int partLength) {

    npy_intp shape[2] = {partLength, 3};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(2, shape, NPY_INT);
    int *parts = (int*) PyArray_DATA(result);

    int idx = 0;
    do {
        parts[idx] = part->first / 3;
        parts[idx+1] = part->last / 3;
        parts[idx+2] = 0;

        idx += 3;
        part = part->next;
    } while (part != NULL);

    return result;
}

PyObject* fixCorners(PyObject *self, PyObject *args) {
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arr1, &PyArray_Type, &arr2)) {
        PyErr_SetString(PyExc_TypeError, "Expected two numpy arrays.");
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 2 || PyArray_TYPE(arr1) != NPY_FLOAT64 || PyArray_NDIM(arr2) != 2 || PyArray_TYPE(arr2) != NPY_INT) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type double.");
        return NULL;
    }

    double *data = PyArray_DATA(arr1);
    int dataSize = PyArray_SHAPE(arr1)[0] * 3;

    int *parts = PyArray_DATA(arr2);
    int partLength = PyArray_SHAPE(arr2)[0];

    Line* root = partsToLines(parts, partLength);    
    findCorners(data, dataSize, &root, &partLength);    
    return updateParts(root, partLength);
}


void initFindCorner() {
    _import_array();
}