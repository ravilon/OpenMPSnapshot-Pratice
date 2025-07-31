#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "ctimer.h"

#define N 16384000
#define PAD 50
#define NUM_BLOCKS 8

int main(int argc, char **argv) {
    char *a;
    unsigned int i, sum;
    double t1, t2, tucpu, tscpu;

    a = malloc(N * sizeof(char));

    srand(time(NULL));

    for (i = 0; i < N; i++) {
        a[i] = (double)(rand() % PAD) + PAD;
    }

    // Sequential implementation
    sum = 0;
    ctimer(&t1, &tucpu, &tscpu);
    for (i = 0; i < N; i++) {
        sum += a[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Sequential sum\t%d\n", sum);
    printf("Time:\t%f seconds\n", (float)(t2 - t1));

    // Parallel implementation, concurrent computation and reduction of partial additions
    const int blockSize = N / NUM_BLOCKS;
    sum = 0;
    ctimer(&t1, &tucpu, &tscpu);
    #pragma omp parallel for shared(a) private(i) reduction(+ : sum) schedule(static, blockSize)
    for (i = 0; i < N; i++) {
        sum += a[i];
    }
    ctimer(&t2, &tucpu, &tscpu);
    printf("Parallel sum\t%d\n", sum);
    printf("Time:\t%f seconds\n", (float)(t2 - t1));

    printf("That's all, folks!\n");
    return 0;
}
