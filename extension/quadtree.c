#include "quadtree.h"

#include <stdlib.h>
#include <math.h>

// Define a point struct with x, y coordinates
typedef struct {
    float x, y;
} Point;

// Define a quadtree node struct
typedef struct QuadtreeNode {
    float min_x, max_x, min_y, max_y;  // Boundaries
    Point *points;  // Array of points in this node
    int capacity;   // Maximum number of points before splitting
    int count;      // Current number of points
    struct QuadtreeNode *children[4];  // Pointers to child nodes (if split)
} QuadtreeNode;

// Function to create a new Quadtree node
QuadtreeNode* QuadtreeNode_new(float min_x, float max_x, float min_y, float max_y, int capacity) {
    QuadtreeNode *node = (QuadtreeNode*)malloc(sizeof(QuadtreeNode));
    node->min_x = min_x;
    node->max_x = max_x;
    node->min_y = min_y;
    node->max_y = max_y;
    node->capacity = capacity;
    node->count = 0;
    node->points = (Point*)malloc(capacity * sizeof(Point));
    for (int i = 0; i < 4; i++) node->children[i] = NULL;
    return node;
}

// Function to check if a point is within the bounds of a node
int QuadtreeNode_contains(QuadtreeNode *node, Point p) {
    return (p.x >= node->min_x && p.x <= node->max_x && p.y >= node->min_y && p.y <= node->max_y);
}

// Function to split a node into 4 children
void QuadtreeNode_split(QuadtreeNode *node) {
    float mid_x = (node->min_x + node->max_x) / 2;
    float mid_y = (node->min_y + node->max_y) / 2;

    // Create four child nodes
    node->children[0] = QuadtreeNode_new(node->min_x, mid_x, node->min_y, mid_y, node->capacity);
    node->children[1] = QuadtreeNode_new(mid_x, node->max_x, node->min_y, mid_y, node->capacity);
    node->children[2] = QuadtreeNode_new(node->min_x, mid_x, mid_y, node->max_y, node->capacity);
    node->children[3] = QuadtreeNode_new(mid_x, node->max_x, mid_y, node->max_y, node->capacity);
}

// Function to insert a point into the quadtree
int QuadtreeNode_insert(QuadtreeNode *node, Point p) {
    if (!QuadtreeNode_contains(node, p)) return 0;  // Point is out of bounds

    // If the node has space, add the point
    if (node->count < node->capacity) {
        node->points[node->count++] = p;
        return 1;
    }

    // If the node is full, split and distribute points
    if (node->children[0] == NULL) QuadtreeNode_split(node);

    // Recursively try inserting the point into a child node
    for (int i = 0; i < 4; i++) {
        if (QuadtreeNode_insert(node->children[i], p)) return 1;
    }

    return 0;
}

// Function to search for points within a radius
void QuadtreeNode_query(QuadtreeNode *node, Point center, float radius, Point **result, int *result_count, int *result_capacity) {
    // Check if the node is within the query range
    if (center.x - radius > node->max_x || center.x + radius < node->min_x ||
        center.y - radius > node->max_y || center.y + radius < node->min_y) {
        return;
    }

    // Check points in the current node
    for (int i = 0; i < node->count; i++) {
        float dx = node->points[i].x - center.x;
        float dy = node->points[i].y - center.y;
        if (sqrt(dx * dx + dy * dy) <= radius) {
            if (*result_count >= *result_capacity) {
                *result_capacity *= 2;
                *result = (Point*)realloc(*result, (*result_capacity) * sizeof(Point));
            }
            (*result)[(*result_count)++] = node->points[i];
        }
    }

    // Recursively query child nodes
    if (node->children[0] != NULL) {
        for (int i = 0; i < 4; i++) {
            QuadtreeNode_query(node->children[i], center, radius, result, result_count, result_capacity);
        }
    }
}

// Python bindings

// A Python object to represent the Quadtree
typedef struct {
    PyObject_HEAD
    QuadtreeNode *root;
} Quadtree;

// Destructor for the Quadtree object
void Quadtree_dealloc(Quadtree *self) {
    free(self->root);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *Quadtree_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    Quadtree *self;
    self = (Quadtree *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->root = NULL;  // Initialize the root pointer to NULL
    }
    return (PyObject *)self;
}

// Initialize a new Quadtree object
int Quadtree_init(Quadtree *self, PyObject *args, PyObject *kwds) {
    float min_x, max_x, min_y, max_y;
    int capacity;

    if (!PyArg_ParseTuple(args, "ffffi", &min_x, &max_x, &min_y, &max_y, &capacity)) {
        return -1;
    }

    self->root = QuadtreeNode_new(min_x, max_x, min_y, max_y, capacity);
    return 0;
}

// Method to insert a point
PyObject* Quadtree_insert(Quadtree *self, PyObject *args) {
    float x, y;
    if (!PyArg_ParseTuple(args, "ff", &x, &y)) {
        return NULL;
    }

    Point p = {x, y};
    if (QuadtreeNode_insert(self->root, p)) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// Method to query points within a radius
PyObject* Quadtree_query(Quadtree *self, PyObject *args) {
    float x, y, radius;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &radius)) {
        return NULL;
    }

    Point center = {x, y};
    Point *result = (Point*)malloc(10 * sizeof(Point));
    int result_count = 0;
    int result_capacity = 10;

    QuadtreeNode_query(self->root, center, radius, &result, &result_count, &result_capacity);

    PyObject *py_result = PyList_New(result_count);
    for (int i = 0; i < result_count; i++) {
        PyObject *point_tuple = PyTuple_Pack(2, PyFloat_FromDouble(result[i].x), PyFloat_FromDouble(result[i].y));
        PyList_SetItem(py_result, i, point_tuple);
    }

    free(result);
    return py_result;
}

void QuadtreeNode_repr(QuadtreeNode *node, PyObject *list) {
    if (node == NULL) return;

    // Add the node's boundaries and point count to the list
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "Node: [min_x=%.2f, max_x=%.2f, min_y=%.2f, max_y=%.2f, points=%d]",
             node->min_x, node->max_x, node->min_y, node->max_y, node->count);
    PyList_Append(list, PyUnicode_FromString(buffer));

    // Add the points in this node to the list
    for (int i = 0; i < node->count && i < 500; i++) {
        snprintf(buffer, sizeof(buffer), "(%.2f, %.2f)", node->points[i].x, node->points[i].y);
        PyList_Append(list, PyUnicode_FromString(buffer));
    }

    // Recursively add child nodes to the list
    for (int i = 0; i < 4; i++) {
        if (node->children[i] != NULL) {
            QuadtreeNode_repr(node->children[i], list);
        }
    }
}

// Python method to return the string representation of the Quadtree
PyObject* Quadtree_repr(Quadtree *self) {
    PyObject *list = PyList_New(0);
    QuadtreeNode_repr(self->root, list);

    // Join the list of strings into one string
    PyObject *separator = PyUnicode_FromString("\n");
    PyObject *result = PyUnicode_Join(separator, list);

    Py_DECREF(list);
    Py_DECREF(separator);
    return result;
}

// Python type definition
PyMethodDef Quadtree_methods[] = {
    {"insert", (PyCFunction)Quadtree_insert, METH_VARARGS, "Insert a point (x, y)"},
    {"query", (PyCFunction)Quadtree_query, METH_VARARGS, "Query points within radius"},
    {NULL}  // Sentinel
};

PyTypeObject quadtree = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "contour.Contour",
    .tp_basicsize = sizeof(Quadtree),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Quadtree_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Quadtree_new,
    .tp_init = (initproc)Quadtree_init,
    .tp_methods = Quadtree_methods,
    .tp_repr = (reprfunc)Quadtree_repr,
};