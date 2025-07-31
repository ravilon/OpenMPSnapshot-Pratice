#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 256
#define MIN(a, b) ( a < b ? a : b )

void matrixInitialization(int *adjacency, int *minCost, unsigned int dim);
void floyd(int *minCost, int *nextStep, int srcVtx, unsigned int dim);

/**
 * OpenMP program that computes the matrix of minimum distances in a directed graph
 * using Floyd's algorithm.
 */
int main() {
    int *adjacencyMatrix, *minCostMatrix, *nextStepMatrix;
    int i, j, xPos, yPos;
    double t1, t2, tucpu, tscpu;

    // Data generation
    adjacencyMatrix = malloc(N * N * sizeof(int));
    minCostMatrix = malloc(N * N * sizeof(int));
    nextStepMatrix = malloc(N * N * sizeof(int));
    srand(time(NULL));
    xPos = rand() % N;
    yPos = rand() % N;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            adjacencyMatrix[i * N + j] = rand() % 2 ? rand() % (N * N) : -1;
        }
    }

    matrixInitialization(adjacencyMatrix, minCostMatrix, N);

    // Sequential algorithm
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        floyd(minCostMatrix, nextStepMatrix, i, N);
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential Floyd algorithm:\tF[%d][%d] = %d\n", xPos, yPos, minCostMatrix[yPos * N + xPos]);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    matrixInitialization(adjacencyMatrix, minCostMatrix, N);

    // Parallel algorithm
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(minCostMatrix) private(i) schedule(dynamic)
    for (i = 0; i < N; i++) {
        floyd(minCostMatrix, nextStepMatrix, i, N);
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel Floyd algorithm:\tF[%d][%d] = %d\n", xPos, yPos, minCostMatrix[yPos * N + xPos]);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}

void matrixInitialization(int *adjacency, int *minCost, unsigned int dim) {
    unsigned int i, j, pos;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            pos = i * dim + j;
            if (i == j) {
                minCost[pos] = 0;
            } else if (adjacency[pos] > 0) {
                minCost[pos] = adjacency[pos];
            } else {
                minCost[pos] = USHRT_MAX;
            }
        }
    }
}

void floyd(int *minCost, int *nextStep, int srcVtx, unsigned int dim) {
    unsigned int j, k;
    int altDist, pos;
    for (k = 0; k < dim; k++) {
        for (j = 0; j < dim; j++) {
            pos = srcVtx * dim + j;
            //#pragma omp critical
            {
                altDist = minCost[srcVtx * dim + k] + minCost[j * dim + k];
                if (minCost[pos] > altDist) {
                    minCost[pos] = altDist;
                    nextStep[pos] = k + 1;
                }
            }
        }
    }
}
