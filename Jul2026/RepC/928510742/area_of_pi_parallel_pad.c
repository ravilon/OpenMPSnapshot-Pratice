/*
In this parallel version, we are using a padding of 8 doubles (64 bytes) per line of the threads_results array.

This is done to get around false sharing. By doing this padding, we avoid the cache's lines to update in all cores.
*/

#include <stdio.h>
#include <omp.h>

static long num_steps = 1'000'000'000;
double step;

#define PAD 8 // assume 64 byte L1 cache line size
#define NUM_THREADS 5

void main()
{
    int i, nthreads;
    double pi, sum[NUM_THREADS][PAD];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

#pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0)
            nthreads = nthrds;
        for (i = id, sum[id][0] = 0.0; i < num_steps; i = i + nthrds)
        {
            x = (i + 0.5) * step;
            sum[id][0] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < nthreads; i++)
        pi += sum[i][0] * step;

    double elapsed, end = omp_get_wtime();
    elapsed = end - start;

    printf("PI = %lf\n", pi);
    printf("Time = %lf s\n", elapsed);
    printf("Threads = %d\n", nthreads);
}