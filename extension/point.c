#include "point.h"


Point toPoint(double* data, int idx) {
    return (Point) {data[idx], data[idx+1]};
}

Point pAdd(Point p1, Point p2) {
    return (Point){p1.x + p2.x, p1.y + p2.y};
}

Point pAdd2(double* data, int idx1, int idx2) {
    return (Point){data[idx1] + data[idx2], data[idx1+1] + data[idx2+1]};
}

Point pSub(Point p1, Point p2) {
    return (Point){p1.x - p2.x, p1.y - p2.y};
}

Point pSub2(double* data, int idx1, int idx2) {
    return (Point){data[idx1] - data[idx2], data[idx1+1] - data[idx2+1]};
}

Point pMul(Point p, double s) {
    return (Point){p.x * s, p.y * s};
}

Point pDiv(Point p, double s) {
    return (Point){p.x / s, p.y / s};
}

double pAbs(Point p) {
    return sqrt(p.x*p.x + p.y*p.y);
}

double pDist(double* data, int* parts, int idx1, int idx2) {
    double x = data[parts[idx1]*3] - data[parts[idx2]*3];
    double y = data[parts[idx1]*3+1] - data[parts[idx2]*3+1];
    return sqrt(x*x + y*y); 
}

double pDist2(double* data, int* parts, int idx, Point p) {
    double x = data[parts[idx]*3] - p.x;
    double y = data[parts[idx]*3+1] - p.y;
    return sqrt(x*x + y*y); 
}

double pDist3(double* data, int idx, Point p) {
    double x = data[idx] - p.x;
    double y = data[idx+1] - p.y;
    return sqrt(x*x + y*y); 
}

double pDot(Point p1, Point p2) {
    return p1.x*p2.x + p1.y*p2.y;
}

double pCross(Point p1, Point p2) {
    return  p1.x * p2.y - p1.y * p2.x;
}

Point pDir(double* data, int idx1, int idx2) {
    double x = data[idx2] - data[idx1];
    double y = data[idx2+1] - data[idx1+1];
    double dist = sqrt(x*x + y*y);
    return (Point){x/dist, y/dist}; 
}

Point pPerp(Point p) {
    return (Point){-p.y, p.x};
}

Point pMiddle(double* data, int idx1, int idx2) {
    return (Point){(data[idx1] + data[idx2]) / 2, (data[idx1+1] + data[idx2+1]) / 2};
}