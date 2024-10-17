#include "contour.h"
#include "splitcontour.h"
#include "findcorner.h"
#include "segment.h"
#include "circles.h"

static PyMethodDef MyMethods[] = {
    {"getContour", getContour, METH_VARARGS, "Get all contours from an image."},
    {"getContourParts", getContourParts, METH_VARARGS, "Split contour into small parts."},
    {"fixCorners", fixCorners, METH_VARARGS, "Fix corners to accurate position."},
    {"splitIntoSegments", splitIntoSegments, METH_VARARGS, "Split contour into segments."},
    {"findCirclesAndLines", findCirclesAndLines, METH_VARARGS, "Split contour into segments."},

    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef contour = {
    PyModuleDef_HEAD_INIT,
    "contour", // Name of the module
    NULL,       // Module documentation (could be a docstring)
    -1,         // Size of per-interpreter state of the contour
    MyMethods
};


PyMODINIT_FUNC PyInit_contour(void) {

    PyObject* m;
    

    //import_array();
    initContour();
    initSplitContour();
    initFindCorner();
    initSegments();
    initCircles();

    if (PyErr_Occurred()) {
        printf("Shit");
        return NULL;
    }
    m = PyModule_Create(&contour);
    if (m == NULL)
        return NULL;

    //if (PyType_Ready(&quadtree) < 0)
    //   return NULL;

    //Py_INCREF(&quadtree);
    //PyModule_AddObject(m, "Contour", (PyObject *)&quadtree);

    return m;
}