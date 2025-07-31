#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "ctimer.h"

#define N 65536000
#define P 8

/**
 * OpenMP program that computes the euclidean norm of a float vector,
 * spanning a maximum of `p` threads.
 * Suggestion: It must not fail if the euclidean norm is smaller than the greatest
 * positive number that can be represented using [floating] simple precision.
 */
int main() {
    float *a;
    double seqNorm2, parNorm2;
    unsigned int i;
    double t1, t2, tucpu, tscpu;

    a = malloc(N * sizeof(float));

    // Data generation
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        a[i] = rand() % N + P;
    }

    // Sequential algorithm
    seqNorm2 = 0.;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        seqNorm2 += a[i] * a[i];
    }
    seqNorm2 = (float)sqrt(seqNorm2);
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential norm-2 =\t%.1f\n", seqNorm2);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    parNorm2 = 0.;
    omp_set_num_threads(P);
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel shared(a, parNorm2) private(i)
    {
        #pragma omp for reduction(+ : parNorm2) schedule(static)
        for (i = 0; i < N; i++) {
            parNorm2 += a[i] * a[i];
        }
        if (!omp_get_thread_num()) {
            parNorm2 = (float)sqrt(parNorm2);
        }
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel norm-2 =\t%.1f\n", parNorm2);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
