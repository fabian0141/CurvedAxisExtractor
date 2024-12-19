#include "splitcontour.h"
#include "usenumpy.h"

#include "line.h"

void shiftElementsD(double *arr, int size, int n) {
    double* temp = malloc(size * sizeof(double));
    memmove(temp, arr + n, (size - n) * sizeof(double));
    memmove(temp + (size - n), arr, n * sizeof(double));
    memmove(arr, temp, size * sizeof(double));
    free(temp);
}

double dist2(double *data, int idx1, int idx2) {
    double x = data[idx1] - data[idx2];
    double y = data[idx1+1] - data[idx2+1];
    return sqrt(x*x + y*y);
}

// improve performance only repeat part for middle
double distancePointToLine(double *data, int first, int last, int middle) {
    double lX = data[last] - data[first];
    double lY = data[last+1] - data[first+1];

    double a = sqrt(lX*lX + lY*lY);
    double d = lY * data[middle] - lX * data[middle+1] + data[last]*data[first+1] - data[last+1]*data[first];
    double res = fabs(d) / a;
    return res;
}



void getFurthest2Points(double *data, int size, int *idx1, int *idx2) {
    double max = 0;
    for (int i = 3; i < size; i += 3) {
        double d =  dist2(data, 0, i);
        if (d > max) {
            *idx1 = i;
            max = d;
        }
    }

    max = 0;
    for (int i = 0; i < size; i += 3) {
        if (*idx1 == i)
            continue;

        double d =  dist2(data, *idx1, i);
        if (d > max) {
            *idx2 = i;
            max = d;
        }
    }

    if (*idx2 < *idx1) {
        int tmp = *idx1;
        *idx1 = *idx2;
        *idx2 = tmp;
    }    
}

Line* splitContourPart(double *data, int idx1, int idx2, int *counter) {
    int idx = -1;
    double max = 0;
        for (int i = idx1; i < idx2 + 3; i += 3)
        {
            double dist = distancePointToLine(data, idx1, idx2, i);
            if (dist > 1 && dist > max) {
                idx = i;
                max = dist;
            }
        }

    if (idx == -1) {
        Line* part = malloc(sizeof(Line));
        part->first = idx1;
        part->last = idx2;
        part->next = NULL;
        part->end = part;
        (*counter)++;
        return part;
    }

    Line* part = splitContourPart(data, idx1, idx, counter);
    Line* next = splitContourPart(data, idx, idx2, counter);
    part->end->next = next;
    part->end = next->end;
    return part;
}

Line* getParts(double *data, int size, int *counter) {
    int idx1 = 0, idx2 = 0;
    getFurthest2Points(data, size, &idx1, &idx2);
    shiftElementsD(data, size, idx1);

    int n = idx2 - idx1;
    Line *part = splitContourPart(data, 0, n, counter);
    Line* next = splitContourPart(data, n, size-3, counter);
    part->end->next = next;
    part->end = next->end;
    return part;
}


PyObject* getContourParts(PyObject *self, PyObject *args) {
    PyArrayObject *arr1 = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array.");
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 2 || PyArray_TYPE(arr1) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type double.");
        return NULL;
    }

    double *data = (double*) PyArray_DATA(arr1);
    int size = PyArray_SHAPE(arr1)[0] * 3;

    int counter = 0;
    Line *part = getParts(data, size, &counter);

    npy_intp shape[2] = {counter, 3};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(2, shape, NPY_INT);
    int *parts = (int*) PyArray_DATA(result);

    int idx = 0;
    do {

        parts[idx] = part->first / 3;
        parts[idx + 1] = part->last / 3;
        parts[idx + 2] = 0;
        idx += 3;

        part = part->next;
    } while (part != NULL);

    return result;
}