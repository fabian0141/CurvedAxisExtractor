#include "angle.h"
#include "usenumpy.h"

#include "point.h"

const static double PI = 3.14159265359; 

int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

PyObject* getTangentAngles(PyObject *self, PyObject *args) {
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
    int size = PyArray_SHAPE(arr1)[0];

    npy_intp shape[1] = {size};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_FLOAT64);
    double *angs = (int*) PyArray_DATA(result);

    double lastAng = 0.0;
    double ang = 0.0;

    for (int i = 0; i < size; i++) {

        Point vec1 = pDir(data, i * 3, mod(i-3, size) * 3);
        Point vec2 = pDir(data, i * 3, mod(i+3, size) * 3);
        Point vec3 = pSub(vec2, vec1);


        double angle = atan2(vec3.y, vec3.x);
        if (angle < 0) angle = angle + 2*PI;

        double delta = angle - lastAng;
        if (fabs(delta) > PI) delta = 2*PI - fabs(delta);

        ang += delta;
        //printf("(%f %f) (%f %f) (%f %f) %f %f %f %f\n", vec1.x, vec1.y, vec2.x, vec2.y, vec3.x, vec3.y, angle, delta, lastAng, ang);

        lastAng = angle;
        angs[i] = ang;
    }
    return result;
}

PyObject* smoothValues(PyObject *self, PyObject *args) {
    PyArrayObject *arr1 = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array.");
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 1 || PyArray_TYPE(arr1) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be one-dimensional and of type double.");
        return NULL;
    }

    double *data = (double*) PyArray_DATA(arr1);
    int size = PyArray_SHAPE(arr1)[0];

    npy_intp shape[1] = {size};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_FLOAT64);
    double *smoothed = (int*) PyArray_DATA(result);

    for (int i = 0; i < size; i++) {

        double c = data[i];
        for (int j = 0; j < 5; j++) {
            c += data[mod(i - j, size)] + data[mod(i + j, size)];
        }
        c /= 11;
        smoothed[i] = c;
    }
    return result;
}

PyObject* getSlopesFromAngs(PyObject *self, PyObject *args) {
    PyArrayObject *arr1 = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array.");
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 1 || PyArray_TYPE(arr1) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be one-dimensional and of type double.");
        return NULL;
    }

    double *data = (double*) PyArray_DATA(arr1);
    int size = PyArray_SHAPE(arr1)[0];

    npy_intp shape[1] = {size};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_FLOAT64);
    double *slopes = (int*) PyArray_DATA(result);

    for (int i = 0; i < size; i++) {
        double c = data[mod(i + 5, size)] - data[mod(i - 5, size)];
        slopes[i] = c;
    }
    return result;
}