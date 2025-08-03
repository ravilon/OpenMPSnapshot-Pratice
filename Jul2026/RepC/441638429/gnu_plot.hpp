#pragma once

#include <stdio.h>

class GNUPlotter {
private:
FILE* gnu = NULL;
int m;  // x-dimension size
int n;  // y-dimension size
int o;  // offset for both x and y (for ghost boundaries)

public:
// Constructor opens GNU pipe in 'w' mode
GNUPlotter() { gnu = popen("gnuplot", "w"); }

// Destructor closes GNU pipe
~GNUPlotter() { pclose(gnu); }

// Set the dimensions of plot, instead of providing it on each plot
void setDimensions(int nx, int ny, int offset) {
m = nx;
n = ny;
o = offset;
};

void plotMesh(const char* title, bool** mesh) {
fprintf(gnu, "set title %s\n", title);
fprintf(gnu, "set size square\n");
fprintf(gnu, "set key off\n");
fprintf(gnu, "plot [0:%d] [0:%d] \"-\"\n", m - 1, n - 1);
for (int i = o; i < m - o; i++) {
for (int j = o; j < n - o; j++)
if (mesh[i][j]) {
fprintf(gnu, "%d %d\n", j, n - i - o);
}
}

fprintf(gnu, "e\n");
fflush(gnu);
}
};