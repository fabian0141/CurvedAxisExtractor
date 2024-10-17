#include "pointlist.h"

PointList convertToLinkedList(PointNode **pointList, int counter) {
    PointNode *node = pointList[0];
    PointList pList = {node, counter};
    for (int i = 1; i < counter; i++)
    {
        PointNode *nextNode = pointList[i];
        node->next = nextNode;
        nextNode->prev = node;
        node = nextNode;
    }
    return pList;
}

void swap(PointNode **a, PointNode **b) {
    PointNode *temp = *a;
    *a = *b;
    *b = temp;
}

int partition(PointNode **pointList, int low, int high) {
    double pivot = pointList[high]->val;
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (pointList[j]->val < pivot) {
            i++;
            swap(&pointList[i], &pointList[j]);
        }
    }
    swap(&pointList[i + 1], &pointList[high]);
    return (i + 1);
}

void quickSort(PointNode **pointList, int low, int high) {
    if (low < high) {
        int pi = partition(pointList, low, high);

        quickSort(pointList, low, pi - 1);
        quickSort(pointList, pi + 1, high);
    }
}

void removePoint(PointList *pList, PointNode* node) {

    if (pList->length == 1) {
        pList->root = 0;
    } else if (node->prev == 0) {
        pList->root = node->next;
        pList->root->prev = 0;
    } else if (node->next == 0) {
        node->prev->next = 0;
    } else {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    pList->length--;
    free(node);
}


void deleteList(PointList *pList) {
    PointNode* current = pList->root;
    PointNode* nextNode;

    while (current != 0) {
        nextNode = current->next;  // Save the next node
        free(current);             // Free the current node
        current = nextNode;        // Move to the next node
    }

    pList->root = 0;  // Set the head to NULL, indicating the list is now empty
}

double pointDist(PointNode *p1, PointNode *p2) {
    //printf("L: %f %f %f %f\n", p1.x, p1.y, p2.x, p2.y);
    double x = p1->x - p2->x;
    double y = p1->y - p2->y;
    return sqrt(x*x + y*y);
}