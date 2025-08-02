#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#define NUM_THREADS 4

int main(void){
    long int num_steps = 1e8;

    double pi;

    double step = 1.0/(double)num_steps;

    omp_set_num_threads(NUM_THREADS);

    // real    0m0.170s
    // user    0m0.650s
    // sys     0m0.000s
    #pragma omp parallel
    {
        double x, parcialSum = 0.0;

        #pragma omp for
        for(int i = 0; i < num_steps; i++){
            x = (i + 0.5) * step;
            parcialSum += 4.0 / (1.0 + x * x);
        }
        
        #pragma omp atomic
            pi += parcialSum * step;
    }

    printf("%f\n", pi);
    
    return 0;
}