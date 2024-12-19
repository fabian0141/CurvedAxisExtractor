#include "circle.h"


double determinant(double a[3][3]) {
    return a[0][0] * a[1][1] + a[0][1] * a[2][0] + a[1][0] * a[2][1] - 
           a[2][0] * a[1][1] - a[2][1] * a[0][0] - a[1][0] * a[0][1]; 
}

void circleLGS(double a[3][3], double b[3], Point* middlePoint, double* radius) {
    
    double factor = a[1][0] / a[0][0];
    a[1][1] -= factor * a[0][1];
    a[1][2] -= factor * a[0][2];
    b[1]    -= factor * b[0];

    factor = a[2][0] / a[0][0];
    a[2][1] -= factor * a[0][1];
    a[2][2] -= factor * a[0][2];
    b[2]    -= factor * b[0];

    factor = a[2][1] / a[1][1];
    a[2][2] -= factor * a[1][2];
    b[2]    -= factor * b[1];

    double x3 =  b[2] / a[2][2];
    double x2 = (b[1] - x3 * a[1][2]) / a[1][1];
    double x1 = (b[0] - x3 * a[0][2] - x2 * a[0][1]) / a[0][0];

    (*middlePoint).x = -x1;
    (*middlePoint).y = -x2;
    *radius = sqrt(middlePoint->x*middlePoint->x + middlePoint->y*middlePoint->y - x3);
}

int isCircleValid(Circle* circle, double* data, int* parts, int start, int end) {
    start *= 3;
    end *= 3;

    for (int i = start+3; i < end; i += 3) {
        double dist = pDist2(data, parts, i, circle->middle);
        if (fabs(dist - circle->radius) > 2) {
            return 0;
        }
    }

    for (int i = start; i <= end; i += 3) {
        for (int j = parts[i]; j < parts[i+1]; j++)
        {
            double dist = pDist3(data, j*3, circle->middle);
            if (fabs(dist - circle->radius) > 3) {
                return 0;
            }
        }
    }
    return 1;
}