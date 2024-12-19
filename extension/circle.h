#include "point.h"

typedef struct {
    Point start;
    Point between;
    Point end;
    Point middle;
    double radius;
} Circle;

double determinant(double a[3][3]);
void circleLGS(double a[3][3], double b[3], Point* middlePoint, double* radius);
int isCircleValid(Circle* circle, double* data, int* parts, int start, int end);
