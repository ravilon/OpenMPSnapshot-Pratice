#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "ctimer.h"

#define N 9
#define P 8

/**
 * OpenMP program that multiplies two long integers stored as a couple of vectors,
 * spanning a maximum of `p` threads.
 */
int main() {
    char *av, *bv;
    unsigned long a, b, seqMult, parMult;
    double t1, t2, tucpu, tscpu;
    unsigned int i, pow10;

    av = malloc(N * sizeof(char));
    bv = malloc(N * sizeof(char));

    // Data generation
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        av[i] = rand() % 10;
        bv[i] = rand() % 10;
    }

    // Sequential algorithm
    ctimer(&t1, &tucpu, &tscpu);
    a = b = 0;
    for (i = 0; i < N; i++) {
        pow10 = pow(10, N - (i + 1));
        a += av[i] * pow10;
        b += bv[i] * pow10;
    }
    seqMult = a * b;
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential product =\t%ld\n", seqMult);
    printf("Time:\t%.9f\n", (float)(t2 - t1));

    // Parallel algorithm
    ctimer(&t1, &tucpu, &tscpu);
    a = b = 0;
    omp_set_num_threads(P);
    #pragma omp parallel for shared(av, bv) private(i, pow10) reduction(+ : a, b) schedule(dynamic)
    for (i = 0; i < N; i++) {
        pow10 = pow(10, N - (i + 1));
        a += av[i] * pow10;
        b += bv[i] * pow10;
    }
    parMult = a * b;
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel product =\t%ld\n", parMult);
    printf("Time:\t%.9f\n", (float)(t2 - t1));

    return 0;
}
