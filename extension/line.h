#ifndef _LINE_
#define _LINE_

#include <Python.h>

#include "point.h"

typedef struct Line {
    int first;
    int last;
    struct Line *next;
    struct Line *prev;
    struct Line *end;
} Line;

typedef struct {
    Point p1;
    Point p2;
    int distincitveWall;
} LineSegment;

Line* partsToLines(int* parts, int partLength);
Point closestPointOnLine(double* data, int idx1, int idx2, Point p);
Point linesIntersection(double* data, Line* l1, Line* l2);

double lineAngle(double* data, int idx1, int idx2, int idx3);
double lineLength(double* data, int* parts, int idx);

void addLine(LineSegment* lines, int lineIdx, double* data, int* parts, int partIdx);
void addLines(LineSegment* lines, int* linesLength, double* data, int* parts, int start, int end);

void updateData(double* data, int idx, Point p);
void insertLine(Line* line);
Line* removeLine(Line* root, Line* line);

#endif