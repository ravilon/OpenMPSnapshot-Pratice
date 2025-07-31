#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 16384
#define P 8

/**
 * OpenMP program that computes the sum of the `n` elements of a vector of doubles,
 * spanning a maximum of `p` threads, both with and without using reduction instructions.
 * Suggestion: Take `p` as power of 2 and `n` as a multiple of `p`.
 */
int main() {
    double *A, *b;
    double sum1 = 0., sum2 = 0.;
    double t1, t2, tucpu, tscpu;
    int threadId, i;

    A = malloc(N * sizeof(double));
    b = malloc(P * sizeof(double));
    memset(b, 0, P * sizeof(double));

    omp_set_num_threads(P);

    srand(time(NULL));

    // Data generation
    for (i = 0; i < N; i++) {
        A[i] = rand() % N + P;
    }

    // Parallel algorithm with reduction instructions
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(A) private(i) reduction(+ : sum1) schedule(static)
    for (i = 0; i < N; i++) {
        sum1 += A[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel sum using reduction instructions =\t%.0f\n", sum1);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm without reduction instructions
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel shared(A, b) private(threadId, i)
    {
        threadId = omp_get_thread_num();
        for (i = threadId; i < N; i += P) {
            b[threadId] += A[i];
        }
    }
    for (i = 0; i < P; i++) {
        sum2 += b[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel sum WITHOUT using reduction instructions =\t%.0f\n", sum2);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}
