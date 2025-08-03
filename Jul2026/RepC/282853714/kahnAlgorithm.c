#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "functions.h"
#include "linked_list.h"

int main(int argc,  char *argv[])
{
struct timeval start, end;

int val;

int threads = atoi(argv[1]);
omp_set_num_threads(threads);

// Initialize linked lists
node headS;
headS = NULL;
node headL;
headL = NULL;

int i = 0;
int j = 0;
int k = 0;
int count = 0;
double time_taken;

// Read the Matrix Market
int size = getAdjacencyMatrixSize("dagt.txt");
int **matrix = getAdjacencyMatrixArray("dagt.txt");

int *indegree = initialIndegree(matrix, size);

// Initialize S linked list with ids that have zero indegree
for (k = 0; k < size; k++)
{
if (indegree[k] == 0)
{
headS = insertNode(headS, k+1);
}
}

int tid;

// Start timer
gettimeofday(&start, NULL);

#pragma omp parallel private(i, j, tid)
{
#pragma omp single
{
// Kahn algorithm loop
while (count < size)
{
#pragma omp task
{
// Remove node from S and add it to L
i = headS->data - 1;
#pragma omp critical (c1)
headL = insertNode(headL, headS->data);
count++;

for (j = 0; j < size; j++)
{
if (matrix[i][j] == 1)
{
// Remove incoming edges
#pragma omp critical (c2)
indegree[j] -= 1;

// If node has no incoming edges then add to S
if (indegree[j] == 0)
{
#pragma omp critical (c1)
insertNode(headS, j+1);
}
}
}
}

if(headS->next != NULL)
{
headS = headS->next;
}
else
{
#pragma omp taskwait
headS = headS->next;
}
}
}
}

// End timer
gettimeofday(&end, NULL);

// Check if graph is cyclic
if (calculateSize(headL) < size)
{
printf("Graph is cyclic. No topological sort.\n");
}
else
{
printf("The topological sort is: ");
printLinkedList(headL);
}

// Calculate time taken
time_taken = (end.tv_sec - start.tv_sec) * 1e6; 
time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6; 

printf("Time elapsed is %f seconds\n",  time_taken);

return 0;
}
