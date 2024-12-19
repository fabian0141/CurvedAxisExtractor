#ifndef _Point_
#define _Point_

typedef struct {
    double x;
    double y;
} Point;

Point toPoint(double* data, int idx);

Point pAdd(Point p1, Point p2);
Point pAdd2(double* data, int idx1, int idx2);

Point pSub(Point p1, Point p2);
Point pSub2(double* data, int idx1, int idx2);

Point pMul(Point p, double s);
Point pDiv(Point p, double s);

double pAbs(Point p);
double pDist(double* data, int* parts, int idx1, int idx2);
double pDist2(double* data, int* parts, int idx, Point p);
double pDist3(double* data, int idx, Point p);

double pDot(Point p1, Point p2);
double pCross(Point p1, Point p2);
Point pDir(double* data, int idx1, int idx2);
Point pPerp(Point p);
Point pMiddle(double* data, int idx1, int idx2);

#endif