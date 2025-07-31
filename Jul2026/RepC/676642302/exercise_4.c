#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 2048000
#define P 8

/**
 * OpenMP program that computes the element holding the greatest absolute value
 * in a vector of doubles, using a maximum of `p` threads.
 */
int main() {
    double *A;
    double elem, maxi;
    unsigned int i;
    double t1, t2, tucpu, tscpu;

    A = malloc(N * sizeof(double));

    srand(time(NULL));

    // Data generation
    for (i = 0; i < N; i++) {
        A[i] = rand() % N + P;
    }

    // Sequential algorithm
    maxi = -1.;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        elem = A[i];
        if (elem > maxi) {
            maxi = elem;
        }
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential maximum =\t%0.f\n", maxi);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    maxi = -1.;
    omp_set_num_threads(P);
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(A) private(i, elem) reduction(max : maxi) schedule(static)
    for (i = 0; i < N; i++) {
        elem = A[i];
        if (elem > maxi) {
            maxi = elem;
        }
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel maximum =\t%.0f\n", maxi);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
