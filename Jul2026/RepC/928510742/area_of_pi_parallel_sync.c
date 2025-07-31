/*
In this parallel version, we are using syncronizations techniques to avoid using padding.

Even though padding fixes our problem, this isn't a portable solution.
*/

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        printf("Provide a number of threads!\n");
        return -1;
    }

    int nthreads = atoi(argv[1]);
    int actual_nthreads;

    if (nthreads < 1)
    {
        printf("Invalid threads number!\n");
        return -1;
    }

    const long num_steps = 1'000'000'000;
    const double step = 1.0 / (double)num_steps;

    // Time measure variables
    double start, end, elapsed;

    // Threads variables
    omp_set_num_threads(nthreads);

    double pi = 0.0;

    // Get starting time
    start = omp_get_wtime();

#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int omp_nthreads = omp_get_num_threads();

        if (id == 0)
        {
            actual_nthreads = omp_nthreads;
        }

        double x, result = 0.0;
        int i;

        for (i = id; i < num_steps; i += omp_nthreads)
        {
            x = (i + 0.5) * step;
            result += 4.0 / (1.0 + x * x);
        }

        result *= step;

#pragma omp atomic
        pi += result;
    }

    // Get end time
    end = omp_get_wtime();

    // Calculate elapsed (it's in seconds)
    elapsed = end - start;

    printf("Integral value: %lf\n", pi);
    printf("Time taken: %lf s\n", elapsed);
    printf("Threads in team: %d\n", nthreads);

    return 0;
}
