
#include "contour.h"
#include "quadtree.h"


static PyMethodDef MyMethods[] = {
    {"getContour", getContour, METH_VARARGS, "Add two arrays element-wise"},
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
    //import_array();

    PyObject* m;
    
    if (PyType_Ready(&quadtree) < 0)
        return NULL;

    m = PyModule_Create(&contour);
    if (m == NULL)
        return NULL;

    Py_INCREF(&quadtree);
    PyModule_AddObject(m, "Contour", (PyObject *)&quadtree);

    return m;
}