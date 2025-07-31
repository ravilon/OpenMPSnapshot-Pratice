#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 327680000
#define P 8

/**
 * OpenMP program that computes the sum of two float vectors,
 * one of them multiplied by an `alpha` constant (saxpy operation),
 * spanning a maximum of `p` threads.
 */
int main() {
    float *a, *b;
    const int alpha = 32;
    double seqSaxpy, parSaxpy;
    double t1, t2, tucpu, tscpu;
    unsigned int i;

    a = malloc(N * sizeof(float));
    b = malloc(N * sizeof(float));

    srand(time(NULL));

    // Data generation
    for (i = 0; i < N; i++) {
        a[i] = rand() % alpha + P;
        b[i] = rand() % alpha + P;
    }

    // Sequential algorithm
    seqSaxpy = 0.;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        seqSaxpy += alpha * a[i] + b[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential saxpy =\t%.0f\n", seqSaxpy);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    parSaxpy = 0.;
    omp_set_num_threads(P);
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(a, b) firstprivate(alpha) private(i) reduction(+ : parSaxpy) schedule(static)
    for (i = 0; i < N; i++) {
        parSaxpy += alpha * a[i] + b[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel saxpy =\t%.0f\n", parSaxpy);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
