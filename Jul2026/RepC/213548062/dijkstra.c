/**
* CITS3402: High Performance Computing Project 2
* Parallel algorithms to solve all pair shortest path
*
* Clarifications
* The program will automatically generate an ouput file.
* The ouput file will be in binary.
* The program requires the -f flag to specify the input file.
*
* @author Bruce How (22242664)
* @date 25/10/2019
*/

#include "spath.h"

/**
* Runs the Dijkstra algorithm for a set of nodes. The spaths resulting
* array is modified directly hence the double pointer parameter.
* To further speed up the process, OMP parralellism has been implemented.
* 
* @param spaths The shortest path variable to store the results to
* @param weights The list of weights from all nodes to another
* @param numV The nunber of nodes or vertices
* @param nodes The number of nodes to perform dijkstra's Algorithm for
* @param pos The offset positional value
*/
void dijkstra(int **spaths, int *weights, int numV, int nodes, int pos) {
int i, j, k, current;

#pragma omp parallel shared(spaths)
{
#pragma omp for private(i,j,k,current)
for (i = 0; i < nodes; i++) {
int node = pos/numV + i;
int offset = i * numV;

bool *spt = allocate(numV * sizeof(bool)); // Keeps track if visitied

for (j = 0; j < numV; j++) {
(*spaths)[j + offset] = -1;
spt[j] = false;
}
(*spaths)[offset + node] = 0;

current = node;

for (j = 0; j < numV; j++) {
for (k = 0; k < numV; k++) {
// Exclude self distance and unconnected nodes
int direct = weights[(numV * current) + k]; // Direct distance to node
if (k == current || direct == 0) {
continue;
}
int dist = (*spaths)[offset + current] + direct;
if ((*spaths)[offset + k] == -1 || dist < (*spaths)[offset + k]) {
(*spaths)[offset + k] = dist;
}
}
// State that we have visited the node
spt[current] = true;

// Identify next node
int lowest = -1;
for (k = 0; k < numV; k++) {
if (!spt[k] && (*spaths)[offset + k] != -1 && (lowest == -1 || (*spaths)[offset + k] < lowest)) {
lowest = (*spaths)[offset + k];
current = k;
}
}
}
free(spt);
}
}
}
