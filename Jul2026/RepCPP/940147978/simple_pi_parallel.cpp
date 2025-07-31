#include <omp.h>
#include <iostream>

static long num_steps = 100000;
double step;
#define NUM_THREADS 4

int main() {
    int i, nthreads;
    double pi, sum[NUM_THREADS];
    
    step = 1.0 / (double) num_steps;
    omp_set_num_threads(NUM_THREADS);

    double start_time = omp_get_wtime(); // Start clock

    #pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        
        if (id == 0) nthreads = nthrds;
        
        sum[id] = 0.0;
        for (i = id; i < num_steps; i += nthrds) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }

    for (i = 0, pi = 0.0; i < nthreads; i++)
        pi += sum[i] * step;

    double end_time = omp_get_wtime(); // Stop clock
    double execution_time = end_time - start_time;

    std::cout << "Estimated value of Pi: " << pi << std::endl;
    std::cout << "Execution time: " << execution_time << " seconds" << std::endl;

    return 0;
}
