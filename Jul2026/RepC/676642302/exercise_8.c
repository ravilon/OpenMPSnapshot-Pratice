#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "ctimer.h"

#define M 3
#define N 60
#define P 8
#define C 3200

double sequentialPolynom(double* f, int grade, double x);
double parallelPolynom(double* f, int grade, double x);

/**
 * OpenMP program that computes the integral, between `a` and `b`,
 * of a polynomical function of grade `m`, splitting the interval
 * `[a, b]` in `n` subintervals and spanning a maximum of `p` threads.
 */
int main() {
    double f[M + 1];
    const double a = 0., b = 30.;
    double seqIntegral, parIntegral;
    double t1, t2, tucpu, tscpu;
    int i;

    const double xInterval = fabs(b - a) / N;
    const double xStart = a < b ? a : b;

    // Data generation
    srand(time(NULL));
    for (i = 0; i <= M; i++) {
        f[i] = rand() % (M * N * P);
    }

    // Sequential algorithm
    seqIntegral = 0.;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        seqIntegral += sequentialPolynom(f, M, xStart + xInterval * ((double)i + 0.5)) * xInterval;
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential integral =\t%.3f\n", seqIntegral + C);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    // Parallel algorithm
    omp_set_num_threads(P);
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for private(i) reduction(+ : parIntegral) schedule(dynamic)
    for (i = 0; i < N; i++) {
        parIntegral += parallelPolynom(f, M, xStart + xInterval * ((double)i + 0.5)) * xInterval;
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel integral =\t%.3f\n", parIntegral + 3200);
    printf("Time:\t%.9f seconds\n", (float)(t2 - t1));

    return 0;
}

double sequentialPolynom(double *f, int grade, double x) {
    double fx = 0.;
    int n;
    for (n = 0; n <= grade; n++) {
        fx += f[n] * pow(x, n);
    }
    return fx;
}

double parallelPolynom(double *f, int grade, double x) {
    double fx = 0.;
    int n;
    #pragma omp parallel for private(n) reduction(+ : fx)
    for (n = 0; n <= grade; n++) {
        fx += f[n] * pow(x, n);
    }
    return fx;
}
