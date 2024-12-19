#include "line.h"

Line* partsToLines(int* parts, int partLength) {

    Line* root = malloc(sizeof(Line));
    root->first = parts[0] * 3;
    root->last = parts[1] * 3;
    root->prev = NULL;

    Line* line = root;
    for (int i = 3; i < partLength * 3; i += 3) {
        Line* next = malloc(sizeof(Line));
        next->first = parts[i] * 3;
        next->last = parts[i+1] * 3;

        line->next = next;
        next->prev = line;
        line = next;
    }
    line->next = NULL;
    return root;
}

Point closestPointOnLine(double* data, int idx1, int idx2, Point p) {
    Point p1 = toPoint(data, idx1);
    Point p2 = toPoint(data, idx2);
    Point dir1 = pSub(p2, p1);
    Point dir2 = pPerp(dir1);

    double t = pCross(dir2, pSub(p1, p));
    t /= pCross(dir1, dir2);

    return pAdd(p1, pMul(dir1, t));
}

Point linesIntersection(double* data, Line* l1, Line* l2) {
    Point v1 = pSub2(data, l1->first, l1->last);
    Point v2 = pSub2(data, l2->first, l2->last);
    Point v3 = pSub2(data, l1->first, l2->first);

    double denom = pCross(v1, v2);
    if (denom == 0)
        return (Point){-1, -1};

    double t = pCross(v3, v2) / denom;
    Point start = toPoint(data, l1->first);
    return pSub(start, pMul(v1, t));
}

double lineAngle(double* data, int idx1, int idx2, int idx3) {
    idx1 *= 3;
    idx2 *= 3;
    idx3 *= 3;

    Point line1 = pSub2(data, idx1, idx2);
    Point line2 = pSub2(data, idx3, idx2);
    double d = pDot(line1, line2);
    double n = pAbs(line1) * pAbs(line2);
    return acos(d / n);
}

double lineLength(double* data, int* parts, int idx) {
    double x = data[parts[idx]] - data[parts[idx+1]];
    double y = data[parts[idx]+1] - data[parts[idx+1]+1];
    return sqrt(x*x + y*y);
}

void addLine(LineSegment* lines, int lineIdx, double* data, int* parts, int partIdx) {
    partIdx *= 3;
    lines[lineIdx].p1 = toPoint(data, parts[partIdx]*3);
    lines[lineIdx].p2 = toPoint(data, parts[partIdx + 1]*3);
    lines[lineIdx].distincitveWall = 0;
}

void addLines(LineSegment* lines, int* linesLength, double* data, int* parts, int start, int end) {
    for (int  i = start; i <= end; i++) {
        addLine(lines, (*linesLength)++, data, parts, i);
    }
}

void updateData(double* data, int idx, Point p) {
    data[idx] = p.x;
    data[idx+1] = p.y;
}

void insertLine(Line* line) {
    Line* ins = malloc(sizeof(Line));
    Line* next = line->next;
    ins->first = line->last;
    ins->last = next->first;

    line->next = ins;
    ins->next = next;
    ins->prev = line;

    if (next != 0) {
        next->prev = ins;
    } 
}

Line* removeLine(Line* root, Line* line) {
    if (root->next == 0) {
        root = 0;
    } else if (line->prev == 0) {
        root = line->next;
        root->prev = 0;
    } else if (line->next == 0) {
        line->prev->next = 0;
    } else {
        line->prev->next = line->next;
        line->next->prev = line->prev;
    }
    free(line);
    return root;
}