#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 131072000
#define P 8

/**
 * OpenMP program computing the scalar product of two double vectors
 * of `n` elements each, spanning a maximum of `p` threads.
 */
int main() {
    double *a, *b, *x;
    unsigned int i;
    double t1, t2, tucpu, tscpu;

    a = malloc(N * sizeof(double));
    b = malloc(N * sizeof(double));
    x = malloc(N * sizeof(double));

    srand(time(NULL));

    // Data generation
    for (i = 0; i < N; i++) {
        a[i] = rand() % N + P;
        b[i] = rand() % P + N;
    }

    // Sequential algorithm
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        x[i] = a[i] * b[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential scalar product x[5] =\t%.0f\n", x[5]);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    omp_set_num_threads(P);
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(a, b, x) private(i) schedule(static)
    for (i = 0; i < N; i++) {
        x[i] = a[i] * b[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel scalar product x[5] =\t\t%.0f\n", x[5]);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
