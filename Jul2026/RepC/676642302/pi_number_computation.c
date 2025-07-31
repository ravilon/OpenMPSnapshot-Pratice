#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "ctimer.h"

double f(double x);

int main(int argc, char** argv) {
    long long int i;
    const long long int N = 1e10;

    double x, h, piSum, totalPiSec, totalPiOmp;
    double t1, t2, tucpu, tscpu;

    h = 1.0 / (double)N;

    // Sequential computation
    ctimer(&t1, &tucpu, &tscpu);
    piSum = 0.;
    for (i = 0; i < N; i++) {
        x = (i + 0.5) * h;
        piSum += f(x);
    }
    totalPiSec = piSum * h;
    ctimer(&t2, &tucpu, &tscpu);
    printf("\nSequential algorithm:\n");
    printf("pi is approximately %.16f (%lld intervals)\n", totalPiSec, N);
    printf("Time:\t%f seconds\n", (float)(t2 - t1));

    // Parallel computation with OpenMP
    ctimer(&t1, &tucpu, &tscpu);
    piSum = 0.;
    #pragma omp parallel for private(i, x) default(shared) reduction(+:piSum)
    for(i = 0; i < N; i++) {
        x = (i + 0.5) * h;
        piSum += f(x);
    }
    totalPiOmp = piSum * h;
    ctimer(&t2, &tucpu, &tscpu);
    printf("\nParallel algorithm:\n");
    printf("pi is approximately %.16f (%lld intervals)\n", totalPiOmp, N);
    printf("Time\t%f seconds\n", (float)(t2 - t1));

    printf("Result difference Sequential-Parallel: %.16f\n", fabs(totalPiSec - totalPiOmp));
    return 0;
}

double f(double x) {
    return 4.0 / (1. + x * x);
}
