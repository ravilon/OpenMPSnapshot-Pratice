#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#define NUM_THREADS 4

int main(void){
    long int num_steps = 1e8;

    double pi = 0.0, x = 0.0;

    double step = 1.0/(double)num_steps;

    omp_set_num_threads(NUM_THREADS);

    // real    0m0.167s
    // user    0m0.632s
    // sys     0m0.000s
    #pragma omp parallel
    {
        double x = 0.0;
        #pragma omp for reduction (+: pi)
        for(int i = 0; i < num_steps; i++){
            x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }
    }

    pi *= step;

    printf("%f\n", pi);
    
    return 0;
}