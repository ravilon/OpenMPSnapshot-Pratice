#include <iostream>
#include <omp.h>

static long num_steps = 100000;  // Number of iterations
#define NUM_THREADS 2
double step;

int main() {
    double pi = 0.0;
    step = 1.0 / (double) num_steps;

    omp_set_num_threads(NUM_THREADS);

    double start_time = omp_get_wtime();  // Start timer

    #pragma omp parallel
    {
        int i, id, nthrds;
        double x, sum = 0.0;

        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        for (i = id; i < num_steps; i += nthrds) {
            x = (i + 0.5) * step;
            #pragma omp critical
                pi += 4.0 / (1.0 + x * x);
        }

        pi *= step;
    }

    double end_time = omp_get_wtime();  // End timer

    std::cout << "Computed pi = " << pi << std::endl;
    std::cout << "Execution time: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;
}
