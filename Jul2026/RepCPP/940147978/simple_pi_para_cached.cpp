#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define PAD 16  // Assume 64-byte L1 cache line size
#define NUM_THREADS 4

static long num_steps = 100000;
double step;

int main() {
    int i, nthreads;
    double pi;
    
    step = 1.0 / (double) num_steps;
    double *sum = (double *)calloc(NUM_THREADS * PAD, sizeof(double)); // Allocate sum in heap
    
    if (sum == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    double start_time = omp_get_wtime(); // Start timer

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        for (i = id; i < num_steps; i += nthrds) {
            x = (i + 0.5) * step;
            sum[id * PAD] += 4.0 / (1.0 + x * x);
        }
    }

    for (i = 0, pi = 0.0; i < NUM_THREADS; i++) {
        pi += sum[i * PAD] * step;
    }

    double end_time = omp_get_wtime(); // End timer

    printf("Computed pi = %lf\n", pi);
    printf("Execution time: %lf seconds\n", end_time - start_time);

    free(sum); // Free allocated memory
    return 0;
}
