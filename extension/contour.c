#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double x;
    double y;
    double val;
} Point;

typedef struct {
    int length;
    Point points[50];
} Bucket;


//python setup.py build && python setup.py install
// TODO: sort points by value and check then for biggest value
// Function to add two numbers
static PyObject* add(PyObject* self, PyObject* args) {
    PyArrayObject *arr1, *arr2, *result;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arr1, &PyArray_Type, &arr2)) {
        return NULL;
    }

    // Ensure both arrays are of the same shape and type
    if (PyArray_NDIM(arr1) != 1 || PyArray_NDIM(arr2) != 1 ||
        PyArray_TYPE(arr1) != NPY_DOUBLE || PyArray_TYPE(arr2) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be one-dimensional and of type double.");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(arr1);
    if (size != PyArray_SIZE(arr2)) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same size.");
        return NULL;
    }

    // Create a new array for the result
    result = (PyArrayObject*) PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!result) return NULL;

    double *data1 = (double*) PyArray_DATA(arr1);
    double *data2 = (double*) PyArray_DATA(arr2);
    double *data_result = (double*) PyArray_DATA(result);

    // Perform element-wise addition
    for (npy_intp i = 0; i < size; i++) {
        data_result[i] = data1[i] + data2[i];
    }

    return (PyObject*) result;
}

static double dist(Point p1, Point p2) {
    //printf("L: %f %f %f %f\n", p1.x, p1.y, p2.x, p2.y);
    double x = p1.x - p2.x;
    double y = p1.y - p2.y;
    return sqrt(x*x + y*y);
}

static int checkIfBiggestValue(Bucket *buckets, Point p, int buIdx, int buWidth) {
    for (int y = -1; y < 2; y++) {
        for (int x = -1; x < 2; x++) {
            int idx = buIdx + y * buWidth + x;
            Bucket bu = buckets[idx];

            for (int i = 0; i < bu.length; i++)
            {
                if (p.val > bu.points[i].val && dist(p, bu.points[i]) < 2.5) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int findClosestPoint(Bucket *buckets, Point p, int buIdx, int buWidth, int *closestBucket) {
    int closestPoint = -1;
    double closestDist = 1000000000;
    //printf("%f %f %f %d \n", points[curIdx].x, points[curIdx].y, points[curIdx].val, curIdx);


    for (int y = -4; y < 5; y++) {
        for (int x = -4; x < 5; x++) {

            int idx = buIdx + y * buWidth + x;
            Bucket bu = buckets[idx];

            for (int i = 0; i < bu.length; i++)
            {
                double d = dist(p, bu.points[i]);

                if (d < closestDist) {
                    closestDist = d;
                    closestPoint = i;
                    *closestBucket = idx;
                }
            }            
        }
    }

    if (closestPoint == -1) {
        //printf("Closest point not found.\n");
        return -1;
    }
    return closestPoint; 
}

static void removePoint(Bucket *buckets, int buIdx, int pos) {
    Point* startPos = buckets[buIdx].points + pos;
    int size = buckets[buIdx].length - pos - 1;
    memmove(startPos, startPos + 1, size * sizeof(Point));
    buckets[buIdx].length--;
}

static PyObject* getContour(PyObject* self, PyObject* args) {
    PyArrayObject *arr1, *result;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) {
        return NULL;
    }

    if (PyArray_NDIM(arr1) != 2 || PyArray_TYPE(arr1) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type int.");
        return NULL;
    }

    npy_intp *shape = PyArray_SHAPE(arr1);
    int counter = 0; 
    uint8_t *data = (uint8_t*) PyArray_DATA(arr1);
    int size = shape[0] * shape[1];

    Point *points = malloc(size * sizeof(Point));

    int buWidth = (shape[0]+4) / 5;
    int buHeight = (shape[1]+4) / 5;

    Bucket *buckets = malloc(buWidth * buHeight * sizeof(Bucket)); 

    // caluculate optimal postion of pixels
    for (npy_intp y = 1; y < shape[0]-1; y++) {
        for (npy_intp x = 1; x < shape[1]-1; x++) {
            npy_intp index = y * shape[1] + x;
            if (data[index] == 255) {
                continue;
            }


            double sum = 0;
            double px = 0;
            double py = 0;


            for (npy_intp i = -1; i < 2; i++)
            {
                for (npy_intp j = -1; j < 2; j++)
                {
                    npy_intp idx = (y+i) * shape[1] + (x+j);
                    px += j * (255-data[idx]);
                    py += i * (255-data[idx]);
                    sum += (255-data[idx]);     
                }
            }


            px /= sum;
            py /= sum;

            sum /= 9.0;

            px += x;
            py += y;
            
            int buIdx = (int)py / 5 * buWidth + (int)px / 5;
            int pos = buckets[buIdx].length++;
            if (pos >= 50) {
                printf("Bucket too small: %d\n", pos);
            }
            buckets[buIdx].points[pos].x = px;
            buckets[buIdx].points[pos].y = py;
            buckets[buIdx].points[pos].val = 255-sum;
            counter++;
        }
    }
    printf("Point Count: %d \n", counter);

    //get darkest pixel in area
    for (int y = 1; y < buHeight-1; y++) {
        for (int x = 1; x < buWidth-1; x++) {
            int buIdx = y * buWidth + x;
            int leng = buckets[buIdx].length;
            for (int i = 0; i < leng; i++)
            {
                Point p = buckets[buIdx].points[i];
                if (!checkIfBiggestValue(buckets, p, buIdx, buWidth)) {
                    
                    removePoint(buckets, buIdx, i);
                    leng--;
                    i--;
                    counter--;
                    continue;
                }
            }
        }
    }

    npy_intp dims[2] = {counter+10, 3};
    printf("Point Count: %d \n", counter);


    result = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) return NULL;

    double *data_result = (double*) PyArray_DATA(result);

    // find contours by finding closest points to each other 
    int i = 0;
    int conCounter = 0;
    for (int y = 1; y < buHeight-1; y++) {
        for (int x = 1; x < buWidth-1; x++) {
            int buIdx = y * buWidth + x;
            Bucket bu = buckets[buIdx];
            if (bu.length == 0)
                continue;

            int closestBucket = -1;
            int pos = 0;
            int end = 0;

            while (!end) {
                Point p = buckets[buIdx].points[pos];
                data_result[i*3] = p.x;
                data_result[i*3+1] = p.y;
                data_result[i*3+2] = p.val;
                removePoint(buckets, buIdx, pos);
                i++;

                if (i < counter) {
                    pos = findClosestPoint(buckets, p, buIdx, buWidth, &closestBucket);
                    if (pos == -1) {
                        end = 1;
                    }
                    buIdx = closestBucket;
                } else {
                    end = 1;
                }
            }

            conCounter++;
            counter++;
            data_result[i*3+2] = -1;
            i++;
            printf("G: %d %d \n", i, counter);
            if (i == counter) {
                free(points);
                free(buckets);
                return (PyObject*) result;
            }
        }
    }
    free(points);
    free(buckets);
    return (PyObject*) result;
}

static PyObject* testContour(PyObject* self, PyObject* args) {
    PyArrayObject *arr1, *result;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) {
        return NULL;
    }

    // Ensure both arrays are of the same shape and type
    if (PyArray_NDIM(arr1) != 2 || PyArray_TYPE(arr1) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type int.");
        return NULL;
    }

    // Create a new array for the result
    npy_intp *shape = PyArray_SHAPE(arr1);
    npy_intp counter = 0; 
    uint8_t *data = (uint8_t*) PyArray_DATA(arr1);

    for (npy_intp y = 1; y < shape[0]-1; y++) {
        for (npy_intp x = 1; x < shape[1]-1; x++) {
            npy_intp index = y * shape[1] + x;
            if (data[index] != 255) {
                counter++;
            }
        }
    }

    npy_intp dims[2] = {counter, 3};

    result = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) return NULL;

    double *data_result = (double*) PyArray_DATA(result);
    counter = 0;

    // Perform element-wise addition
    for (npy_intp y = 1; y < shape[0]-1; y++) {
        for (npy_intp x = 1; x < shape[1]-1; x++) {
            npy_intp index = y * shape[1] + x;
            if (data[index] != 255) {
                data_result[counter++] = x;
                data_result[counter++] = y;
                data_result[counter++] = data[index];
            }
        }
    }
    return (PyObject*) result;
}

// Method definitions
static PyMethodDef MyMethods[] = {
    {"add", add, METH_VARARGS, "Add two arrays element-wise"},
    {"getContour", getContour, METH_VARARGS, "Add two arrays element-wise"},
    {"testContour", testContour, METH_VARARGS, "Add two arrays element-wise"},

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

// Module initialization function
PyMODINIT_FUNC PyInit_contour(void) {
    import_array();
    return PyModule_Create(&contour);
}