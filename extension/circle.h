#include "point.h"

typedef struct {
    Point start;
    Point between;
    Point end;
    Point middle;
    double radius;
} Circle;

double determinant(double** a);
void circleLGS(double** a, double* b, Point* middlePoint, double* radius);
int isCircleValid(Circle* circle, double* data, int* parts, int start, int end);
