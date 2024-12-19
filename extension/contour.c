#include "contour.h"
#include "usenumpy.h"

#include <stdlib.h>
#include <math.h>

#include "pointlist.h"



//static PyObject* normalizeContour


typedef struct {
    int skip;
    int length;
    PointNode* points[50];
} Bucket;


// python setup.py build && python setup.py install
// Function to add two numbers
// For Mac: export ARCHFLAGS="-arch x86_64"  
int checkInput(PyObject* args, PyArrayObject **arr1) {
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, arr1)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array.");
        return 0;
    }

    if (PyArray_NDIM(*arr1) != 2 || PyArray_TYPE(*arr1) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be two-dimensional and of type int.");
        return 0;
    }
    return 1;
}

int normalize(uint8_t* data, int* nextPixel, npy_intp* shape) {
    double max = 0;
    int pixelCount = 0;
    int last = 0;

    for (npy_intp y = 10; y < shape[0]-10; y++) {
        for (npy_intp x = 10; x < shape[1]-10; x++) {
            npy_intp index = y * shape[1] + x;

            data[index] = 255 - data[index];
            if (max < data[index]) {
                max = data[index];
            }

            if (data[index] > 0) {
                nextPixel[pixelCount++] = index;
            }
        }
    }

    double span = 255/max;

    for (int i = 0; i < pixelCount; i++) {
        int idx = nextPixel[i];
        data[idx] = data[idx]*span;
    }
    return pixelCount;
}

PointNode* pixelPos(int index, uint8_t *data, npy_intp *shape) {

    int x = index % shape[1];
    int y = index / shape[1];
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

    PointNode* newNode = (PointNode*)malloc(sizeof(PointNode));
    newNode->x = px + Sx;
    newNode->y = py + Sy;
    newNode->val = 255 - sum/4;
    newNode->next = NULL;
    newNode->prev = NULL;
    return newNode;
}


int calcOptimalPoints(int pixelCount, uint8_t* data, int* nextPixel, npy_intp* shape, int buWidth, PointNode **pointList, Bucket *buckets) {
    const int TRESHHOLD = 230;
    int counter = 0;

    for (int i = 0; i < pixelCount; i++) {
        int idx = nextPixel[i];

        int index = idx;
        int index2 = idx + 1;
        int index3 = idx + shape[1];
        int index4 = idx + shape[1] + 1;

        if (data[index] < TRESHHOLD && data[index2] < TRESHHOLD && data[index3] < TRESHHOLD && data[index4] < TRESHHOLD) {
            continue;
        }
        PointNode *p = pixelPos(index, data, shape);

        int buIdx = (int)p->y / 5 * buWidth + (int)p->x / 5;
        int pos = buckets[buIdx].length++;
        if (pos >= 50) {
            printf("Bucket too small: %d\n", pos);
        }
        p->buIdx = buIdx;
        pointList[counter++] = p;
        buckets[buIdx].points[pos] = p;
    }

    free(nextPixel);
    return counter;
}

void calcBucketSkips(Bucket *buckets, int buSize) {
    int skip = 0;
    for (int i = buSize-1; i >= 0; i--) {
        buckets[i].skip = skip;
        skip = buckets[i].length > 0 ? 0 : skip + 1;
    }
}

void checkIfBiggestValue(Bucket *buckets, PointList *pList, PointNode *darkPoint, int buWidth) {
    for (int y = -1; y < 2; y++) {
        for (int x = -1; x < 2; x++) {
            int idx = darkPoint->buIdx + y * buWidth + x;
            Bucket *bu = &buckets[idx];

            for (int i = 0; i < bu->length; i++)
            {
                if (darkPoint == bu->points[i])
                    continue;

                PointNode *p = bu->points[i];

                double d = pointDist(darkPoint, p);
                if ((d < 2.5 && darkPoint->val < p->val) || (d < 1 && darkPoint->val == p->val)) {
                    bu->length--;
                    bu->points[i] = bu->points[bu->length];

                    removePoint(pList, p);
                    i--;
                }
            }
        }
    }
}

void filterPoints(PointList *pList, Bucket *buckets, int buWidth) {
    PointNode* node = pList->root;
    do {
        checkIfBiggestValue(buckets, pList, node, buWidth);
        node = node->next;
    } while (node != NULL && node->next != NULL);
}

PointNode* findClosestPoint(Bucket *buckets, PointNode *node, int buWidth) {
    PointNode* closestPoint = NULL;
    double closestDist = 1000000000;
    int buIdx = node->buIdx;

    for (int y = -1; y < 2; y++) {
        for (int x = -1; x < 2; x++) {

            int idx = buIdx + y * buWidth + x;
            Bucket *bu = &buckets[idx];

            for (int i = 0; i < bu->length; i++)
            {
                if (node == bu->points[i]) {
                    bu->length--;
                    bu->points[i] = bu->points[bu->length];
                    i--;
                    continue;
                }

                double d = pointDist(node, bu->points[i]);
                if (d < closestDist) {
                    closestDist = d;
                    closestPoint = bu->points[i];
                }
            }
        }
    }

    return closestPoint; 
}

int getSingleContour(PointList *pList, Bucket *buckets, int buWidth, double** contour) {
    int length = 0;
    PointNode* node = pList->root;

    do {
        (*contour)[length * 3] = node->x;
        (*contour)[length * 3 + 1] = node->y;
        (*contour)[length * 3 + 2] = node->val;
        length++;

        PointNode* newNode = findClosestPoint(buckets, node, buWidth);
        removePoint(pList, node);
        node = newNode;
    } while (node != NULL);

    *contour = realloc(*contour, length * 3 * sizeof(double));
    return length;
}

void free_contour(void* data, void* arr) {
    free(data);
}

PyObject* getAllContours(PointList *pList, Bucket *buckets, int buWidth) {
    PyObject* result = PyList_New(0);
    do {
        double *contour = malloc(pList->length * 3 * sizeof(double));
        int length = getSingleContour(pList, buckets, buWidth, &contour);
        
        npy_intp dims[] = {length, 3};
        PyObject* pyContour = PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT64, NULL, contour, sizeof(double), NPY_ARRAY_OWNDATA, NULL);
        PyArray_SetBaseObject((PyArrayObject*) pyContour, PyCapsule_New(contour, NULL, free_contour));
        PyList_Append(result, pyContour);
    } while (pList->root != NULL);

    free(buckets);
    deleteList(pList);
    return (PyObject*) result;
}

PyObject* returnTestPoints(int buSize, Bucket *buckets, PointList *pList) {

    npy_intp dims[2] = {pList->length, 3};
    PyArrayObject *result = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *data_result = (double*) PyArray_DATA(result);

    int idx = 0;
    for (int i = 0; i < buSize; i++) {
        Bucket bu = buckets[i];
        i += bu.skip;

        for (int pos = 0; pos < bu.length; pos++) {
            PointNode* p = bu.points[pos];
            data_result[idx*3] = p->x;
            data_result[idx*3+1] = p->y;
            data_result[idx*3+2] = p->val;
            idx++;
        }
    }
    free(buckets);
    deleteList(pList);
    return (PyObject*) result;
}


PyObject* getContour(PyObject* self, PyObject* args) {
    PyArrayObject *arr1;

    if (!checkInput(args, &arr1))
        return NULL;

    uint8_t *data = (uint8_t*) PyArray_DATA(arr1);
    npy_intp *shape = PyArray_SHAPE(arr1);

    // TODO: Use dynamic array 
    int *nextPixel = malloc(1000000 * sizeof(int));
    int pixelCount = normalize(data, nextPixel, shape);

    int buWidth = (shape[0]+4) / 5;
    int buHeight = (shape[1]+4) / 5;
    int buSize = buWidth * buHeight;

    Bucket *buckets = malloc(buSize * sizeof(Bucket));
    PointNode **pointList = malloc(1000000 * sizeof(PointNode*));
    int counter = calcOptimalPoints(pixelCount, data, nextPixel, shape, buWidth, pointList, buckets);

    calcBucketSkips(buckets, buSize);
    quickSort(pointList, 0, counter-1); //TODO: use bucketsort maybe?    

    PointList pList = convertToLinkedList(pointList, counter);
    free(pointList);
    
    filterPoints(&pList, buckets, buWidth);
    return getAllContours(&pList, buckets, buWidth);

   //return returnTestPoints(buSize, buckets, &pList);
}