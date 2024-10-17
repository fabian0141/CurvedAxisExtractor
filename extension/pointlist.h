
typedef struct PointNode {
    double x;
    double y;
    double val;
    int buIdx;
    struct PointNode *next;
    struct PointNode *prev;
} PointNode;

typedef struct {
    PointNode *root;
    int length;
} PointList;

void quickSort(PointNode **pointList, int low, int high);
PointList convertToLinkedList(PointNode **pointList, int counter);
void removePoint(PointList *pList, PointNode* node);
void deleteList(PointList *pList);
double pointDist(PointNode *p1, PointNode *p2);
