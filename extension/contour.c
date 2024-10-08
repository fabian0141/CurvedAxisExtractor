#include "contour.h"

#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>



//static PyObject* normalizeContour

typedef struct {
    double x;
    double y;
    double val;
} Point;

typedef struct {
    int length;
    int points[50];
} Bucket;

typedef struct {
    int bucketIdx;
    int position;
    int ref;
    double value;
} PointRef;


void swap(PointRef *a, PointRef *b) {
    PointRef temp = *a;
    *a = *b;
    *b = temp;
}

int partition(PointRef arr[], int low, int high) {
    double pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j].value < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(PointRef arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// python setup.py build && python setup.py install
// Function to add two numbers
// For Mac: export ARCHFLAGS="-arch x86_64"  
PyObject* add(PyObject* self, PyObject* args) {
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

double dist(Point p1, Point p2) {
    //printf("L: %f %f %f %f\n", p1.x, p1.y, p2.x, p2.y);
    double x = p1.x - p2.x;
    double y = p1.y - p2.y;
    return sqrt(x*x + y*y);
}

void removePoint(Bucket *buckets, int buIdx, int pos) {
    int* startPos = buckets[buIdx].points + pos;
    int size = buckets[buIdx].length - pos - 1;
    memmove(startPos, startPos + 1, size * sizeof(int));
    buckets[buIdx].length--;
}

void checkIfBiggestValue(Bucket *buckets, Point *points, Point bigPoint, int buIdx, int ref, int buWidth, int *counter) {
    for (int y = -1; y < 2; y++) {
        for (int x = -1; x < 2; x++) {
            int idx = buIdx + y * buWidth + x;
            Bucket bu = buckets[idx];
            int leng = bu.length;

            for (int i = 0; i < leng; i++)
            {
                Bucket bu = buckets[idx];
                if (ref == bu.points[i])
                    continue;

                
                Point p = points[bu.points[i]];
                if (p.val == -1) {
                    printf("Works %d %d\n", bu.length, i);
                    continue;
                }
                double d = dist(bigPoint, p);
                if ((d < 2.5 && bigPoint.val < p.val) || (d < 1 && bigPoint.val == p.val)) {
                        points[bu.points[i]].val = -1;
                        removePoint(buckets, idx, i);
                        (*counter)--;
                        i--;
                        leng--;
                }
            }
        }
    }
}

int findClosestPoint(Bucket *buckets, Point *points, Point p, int buIdx, int buWidth, int *closestBucket) {
    int closestPoint = -1;
    double closestDist = 1000000000;
    //printf("%f %f %f %d \n", points[curIdx].x, points[curIdx].y, points[curIdx].val, curIdx);


    for (int y = -4; y < 5; y++) {
        for (int x = -4; x < 5; x++) {

            int idx = buIdx + y * buWidth + x;
            Bucket bu = buckets[idx];

            for (int i = 0; i < bu.length; i++)
            {
                double d = dist(p, points[bu.points[i]]);

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

Point pixelPos(int x, int y, double val, uint8_t *data, npy_intp *shape) {
    double sum = 0;
    double px = 0;
    double py = 0;

    double Sx = x + 0.5;
    double Sy = y + 0.5;


    for (npy_intp i = 0; i < 2; i++)
    {
        for (npy_intp j = 0; j < 2; j++)
        {

            int u = x + j;
            int v = y + i;


            npy_intp idx = v * shape[1] + u;
            px += (u - Sx) * data[idx]/255;
            py += (v - Sy) * data[idx]/255;
            sum += data[idx];    

        }
    }

    sum /= 4.0;
    px += Sx;
    py += Sy;
    //if (sum < val)
    //    return (Point){x, y, val};

    return (Point){px, py, sum};
}

PyObject* getContour(PyObject* self, PyObject* args) {
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

    PointRef *pointRefs = malloc(1000000 * sizeof(PointRef));
    Point *points = malloc(1000000 * sizeof(Point));

    int buWidth = (shape[0]+4) / 5;
    int buHeight = (shape[1]+4) / 5;

    Bucket *buckets = malloc(buWidth * buHeight * sizeof(Bucket)); 
    printf("Buckets: %d\n", buWidth * buHeight);

    float max = 0;

    for (npy_intp y = 1; y < shape[0]-1; y++) {
        for (npy_intp x = 1; x < shape[1]-1; x++) {
            npy_intp index = y * shape[1] + x;

            data[index] = 255 - data[index];
            if (max < data[index]) {
                max = data[index];
            }
        }
    }

    float span = 255/max;
    printf("Span: %f, %f \n", span, max);

    for (npy_intp y = 1; y < shape[0]-1; y++) {
        for (npy_intp x = 1; x < shape[1]-1; x++) {
            npy_intp index = y * shape[1] + x;

            data[index] = data[index]*span;
        }
    }

    const int TRESHHOLD = 230;

    // caluculate optimal postion of pixels
    for (npy_intp y = 2; y < shape[0]-2; y++) {
        for (npy_intp x = 2; x < shape[1]-2; x++) {
            npy_intp index = y * shape[1] + x;
            npy_intp index2 = y * shape[1] + x+1;
            npy_intp index3 = (y+1) * shape[1] + x;
            npy_intp index4 = (y+1) * shape[1] + x+1;

            if (data[index] < TRESHHOLD && data[index2] < TRESHHOLD && data[index3] < TRESHHOLD && data[index4] < TRESHHOLD) {
                continue;
            }

            Point p = pixelPos(x, y, 0, data, shape);
            double val = 255-p.val;

            int buIdx = (int)p.y / 5 * buWidth + (int)p.x / 5;
            int pos = buckets[buIdx].length++;
            if (pos >= 50) {
                printf("Bucket too small: %d\n", pos);
            }
            points[counter] = (Point){p.x, p.y, val};
            buckets[buIdx].points[pos] = counter;
            pointRefs[counter] = (PointRef){buIdx, pos, counter, val};
            counter++;
        }
    }
    
    /*quickSort(pointRefs, 0, counter - 1);

    printf("Point Count: %d \n", counter);

    int counter2 = counter;
    for (int i = 0; i < counter2; i++)
    {
        int buIdx = pointRefs[i].bucketIdx;
        Point p = points[pointRefs[i].ref];
        if (p.val == -1) 
            continue;

        checkIfBiggestValue(buckets, points, p, buIdx, pointRefs[i].ref, buWidth, &counter);
    }
   

    npy_intp dims[2] = {counter+10, 3};
    printf("Point Count2: %d \n", counter);


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
                Point p = points[buckets[buIdx].points[pos]];
                data_result[i*3] = p.x;
                data_result[i*3+1] = p.y;
                data_result[i*3+2] = p.val;
                removePoint(buckets, buIdx, pos);
                i++;

                if (i < counter) {
                    pos = findClosestPoint(buckets, points, p, buIdx, buWidth, &closestBucket);
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
                free(pointRefs);
                free(points);
                free(buckets);
                return (PyObject*) result;
            }
        }
    }*/

    npy_intp dims[2] = {counter, 3};

    result = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *data_result = (double*) PyArray_DATA(result);

    int i = 0;
    for (int y = 1; y < buHeight-1; y++) {
        for (int x = 1; x < buWidth-1; x++) {
            int buIdx = y * buWidth + x;
            Bucket bu = buckets[buIdx];
            if (bu.length == 0)
                continue;

            for (int pos = 0; pos < bu.length; pos++) {
                Point p = points[buckets[buIdx].points[pos]];
                //printf("%f %f %d %d %d %f\n", p.x, p.y, buIdx, pos, bu.length, p.val);
                data_result[i*3] = p.x;
                data_result[i*3+1] = p.y;
                data_result[i*3+2] = p.val;
                i++;
            }
        }
    }

    free(pointRefs);
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