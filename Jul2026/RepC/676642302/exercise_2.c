#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "ctimer.h"

/**
 * OpenMP program that computes the sum of the `p` elements of a vector of doubles.
 * The maximum number of threads to be expanded is `p / 2`.
 * Suggestion: Take `p` as power of two.
 */

int main() {
    const int p = 2048;
    double *A;
    double seqSum, parSum;
    const int blockSize = (int)sqrt(p);
    double t1, t2, tucpu, tscpu;
    int i;

    A = malloc(p * sizeof(double));

    srand(time(NULL));

    // Data generation
    for (i = 0; i < p; i++) {
        A[i] = rand() % p + p;
    }

    // Sequential algorithm
    seqSum = parSum = 0.;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < p; i++) {
        seqSum += A[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential sum =\t\t%.0f\n", seqSum);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(A) private(i) reduction(+ : parSum) schedule(dynamic, blockSize)
    for (i = 0; i < p; i++) {
        parSum += A[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel sum of %d blocks =\t%.0f\n", p / blockSize, parSum);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
