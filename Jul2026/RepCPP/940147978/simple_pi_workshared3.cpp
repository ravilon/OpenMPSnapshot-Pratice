#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 4

static long num_steps = 100000;
double step;

int main() {
    int i;
    double pi = 0.0;
    step = 1.0 / (double) num_steps;
    
    double start_time = omp_get_wtime(); // Start timer
    
    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(runtime)
        for (i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }
        
        #pragma omp barrier
        
        #pragma omp for schedule(runtime)
        for (i = 0; i < num_steps; i++) {
            pi *= step;
        }
    }
    
    double end_time = omp_get_wtime(); // End timer
    
    printf("Computed pi = %lf\n", pi);
    printf("Execution time: %lf seconds\n", end_time - start_time);
    
    return 0;
}
