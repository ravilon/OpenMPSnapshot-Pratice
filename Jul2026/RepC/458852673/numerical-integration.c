#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#define NUM_THREADS 4

int main(void){
    int i, nthreads;

    long int num_steps = 1e8;

    double pi = 0.0;

    double step = 1.0/(double)num_steps;

    omp_set_num_threads(NUM_THREADS);

    double *parcialSum = (double *)malloc(omp_get_num_threads() * sizeof(double));

    #pragma omp parallel
    {
        int i, id = omp_get_thread_num();
        double x = 0.0;

        for(i = id, parcialSum[id] = 0; i < num_steps; i += NUM_THREADS){
            x = (i + 0.5) * step;
            parcialSum[id] += 4.0 / (1.0 + x * x);
        }
    }

    for(i = 0, pi = 0.0; i < nthreads; i++) pi += step * parcialSum[i];

    printf("%f\n", pi);
    
    return 0;
}