#include "segment.h"
#include <numpy/arrayobject.h>

#include "line.h"

const float DEG150 = 2.61799387799;

void shiftElementsI(int *arr, int size, int n) {
    int* temp = malloc(size * sizeof(int));
    memmove(temp, arr + n, (size - n) * sizeof(int));
    memmove(temp + (size - n), arr, n * sizeof(int));
    memmove(arr, temp, size * sizeof(int));
    free(temp);
}

void shiftParts(int shift, int** segments, int segmentsLength, int* parts, int partsSize) {
    if (segmentsLength > 2) {
        (*segments)[1] += shift;

        for (int i = 2; i < segmentsLength; i += 2) {
            (*segments)[i] += shift;
            (*segments)[i+1] += shift;
        }
    }
    shiftElementsI(parts, partsSize, (*segments)[segmentsLength]*3);
}

int split(int** segments, double* data, int* parts, int partsSize) {
    int segmentsLength = 0;
    double minAngle = 4;
    double minPos = 0;
    (*segments)[0] = 0;

    for (int i = 0; i < partsSize-3; i += 3)
    {
        double ang = lineAngle(data, parts[i], parts[i + 1], parts[i+4]);

        if (ang < DEG150) {
            segmentsLength += 2;
            (*segments)[segmentsLength-1] = i/3;
            (*segments)[segmentsLength] = i/3 + 1;
        }

        if (ang < minAngle) {
            minAngle = ang;
            minPos = i;
        }
    }

    (*segments)[segmentsLength+1] = partsSize/3 - 1;
    if (segmentsLength == 0) {
        int shift = (partsSize - minPos) / 3;
        shiftParts(shift, segments, segmentsLength, parts, partsSize);

    } else {
        double ang = lineAngle(data, parts[partsSize-2], parts[partsSize-1], parts[1]);
        if (ang >= DEG150) {
            int shift = partsSize / 3 - (*segments)[segmentsLength];
            shiftParts(shift, segments, segmentsLength, parts, partsSize);
            segmentsLength -= 2;
        }
    }

    *segments = realloc(*segments, (segmentsLength + 2) * sizeof(int));
    return segmentsLength/2 + 1;
}

PyObject* toPyArray(int* segments, int segmentsLength) {
    npy_intp dims[] = {segmentsLength, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_INT);
    int* data = (int*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, segments, segmentsLength * 2 * sizeof(int));
    free(segments);
    return result;
}

PyObject* splitIntoSegments(PyObject *self, PyObject *args) {
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
    int partsLength = PyArray_SHAPE(arr2)[0];

    int* segments = malloc(partsLength * 2 * sizeof(int));
    int segmentsLength = split(&segments, data, parts, partsLength*3);
    return toPyArray(segments, segmentsLength);
}


void initSegments() {
    _import_array();
}