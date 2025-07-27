#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "ctimer.h"

#define MAX_DISTANCE 16
#define DIMENSION 256
#define INFINITE INT_MAX

void initializeAdjacencyMatrix(int **adjacencyMatrix, int dimension);
void seqDijkstra(int *adjacencyMatrix, int **minCostMatrix, int **nextStepMatrix, int dimension);
void ompDijkstra(int *adjacencyMatrix, int **minCostMatrix, int **nextStepMatrix, int dimension);
void singleSource(int srcVtx, int *adjacencyMatrix, int *minCostMatrix, int *nextStepMatrix, int dimension);

int main() {
    int *adjacencyMatrix, *seqMinCostMatrix, *seqNextStepMatrix, *ompMinCostMatrix, *ompNextStepMatrix;
    int xPos, yPos;

    initializeAdjacencyMatrix(&adjacencyMatrix, DIMENSION);
    seqDijkstra(adjacencyMatrix, &seqMinCostMatrix, &seqNextStepMatrix, DIMENSION);
    ompDijkstra(adjacencyMatrix, &ompMinCostMatrix, &ompNextStepMatrix, DIMENSION);

    xPos = rand() % DIMENSION;
    yPos = rand() % DIMENSION;

    printf("Sequential min cost %d - %d:\t%d\n", xPos, yPos, seqMinCostMatrix[yPos * DIMENSION + xPos]);
    printf("Parallel min cost %d - %d:\t%d\n", xPos, yPos, ompMinCostMatrix[yPos * DIMENSION + xPos]);

    return 0;
}

void initializeAdjacencyMatrix(int **adjacencyMatrix, int dimension) {
    int i, j, d;

    *adjacencyMatrix = malloc(dimension * dimension * sizeof(int));

    srand(time(NULL));

    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            if (i == j) {
                *(*adjacencyMatrix + i * DIMENSION + j) = 0;
            } else if (rand() % 2) {
                d = rand() % MAX_DISTANCE + 1;
                *(*adjacencyMatrix + i * DIMENSION + j) = d;
                *(*adjacencyMatrix + j * DIMENSION + i) = d;
            } else {
                *(*adjacencyMatrix + i * DIMENSION + j) = *(*adjacencyMatrix + j * DIMENSION + i) = -1;
            }
        }
    }
}

void seqDijkstra(int *adjacencyMatrix, int **minCostMatrix, int **nextStepMatrix, int dimension) {
    double t1, t2, tucpu, tscpu;

    // Allocate memory and set the initial distance from every node to all other vertices
    int i, j, adj;
    *minCostMatrix = malloc(dimension * dimension * sizeof(int));
    *nextStepMatrix = malloc(dimension * dimension * sizeof(int));
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            adj = adjacencyMatrix[i * dimension + j];
            *(*minCostMatrix + i * dimension + j) = adj >= 0 ? adj : INFINITE;
            *(*nextStepMatrix + i * dimension + j) = adj >= 0 ? i + 1 : -1;
        }
    }

    // Calculate the minimum distance between each pair of vertices
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < dimension; i++) {
        singleSource(i, adjacencyMatrix, *minCostMatrix, *nextStepMatrix, dimension);
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
}

void ompDijkstra(int *adjacencyMatrix, int **minCostMatrix, int **nextStepMatrix, int dimension) {
    double t1, t2, tucpu, tscpu;

    // Allocate memory and set the initial distances from every node to all other vertices
    int i, j, adj;
    *minCostMatrix = malloc(dimension * dimension * sizeof(int));
    *nextStepMatrix = malloc(dimension * dimension * sizeof(int));

    #pragma omp parallel for schedule(dynamic) \
        private(i, j) shared(adjacencyMatrix, minCostMatrix, nextStepMatrix, dimension)
    for(i = 0; i < dimension; i++)  {
        for (j = 0; j < dimension; j++) {
            adj = *(adjacencyMatrix + i * dimension + j);
            *(*minCostMatrix + i * dimension + j) = adj >= 0 ? adj : INFINITE;
            *(*nextStepMatrix + i * dimension + j) = adj >= 0 ? i + 1 : -1;
        }
    }

    // Calculate the minimium distance between each pair of vertices
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < dimension; i++) {
        singleSource(i, adjacencyMatrix, *minCostMatrix, *nextStepMatrix, dimension);
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
}

void singleSource(int srcVtx, int *adjacencyMatrix, int *minCostMatrix, int *nextStepMatrix,
        int dimension) {
    int i, j;
    int *marker;    // Used to mark the vertices belonging to V_0
    int *minCostRow, *nextStepRow;
    int u, udist, altdist;
    int minpair[2];

    /* This array is used to indicate whether the shortest path to a vertex has been found.
       If marker[v] is one, then the shortest path to v
       has been found, otherwise it has been not. */
    marker = (int*)calloc(dimension, sizeof(int));

    minCostRow = minCostMatrix + srcVtx * dimension;
    nextStepRow = nextStepMatrix + srcVtx * dimension;

    // Mark the source vertex as being seen
    for (i = 0; i < dimension; i++) {
        marker[i] = i == srcVtx;
    }

    // The main loop of Dijkstra's algorithm
    for (i = 1; i < dimension; i++) {
        // Step 1: Find the local vertex that is at the smallest distance from source
        minpair[0] = INFINITE;
        minpair[1] = -1;
        for (j = 0; j < dimension; j++) {
            if (!marker[j] && minCostRow[j] < minpair[0]) {
                minpair[0] = minCostRow[j];
                minpair[1] = j;
            }
        }

        // Step 2: Compute the global minimum vertex and insert it into V_c
        udist = minpair[0];
        u = minpair[1];

        // Mark the minimum vertex as being seen
        marker[u] = 1;

        // Step 3: Update the distances given that u got inserted
        for (j = 0; j < dimension; j++) {
            if (!marker[j]) {
                altdist = adjacencyMatrix[u * dimension + j];
                altdist = altdist < 0 ? INFINITE : udist + altdist;
                if (altdist < minCostRow[j]) {
                    minCostRow[j] = altdist;
                    nextStepRow[j] = u + 1;
                }
            }
        }
    }

    free(marker);
}
