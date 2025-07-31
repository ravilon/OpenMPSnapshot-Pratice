#include <stdio.h>
#include <omp.h>

int main(int argc, char const *argv[])
{
    const long num_steps = 1'000'000'000;
    const double step = 1.0 / (double)num_steps;

    // Time measure
    double start, end, elapsed;

    long i;
    double pi, sum = 0.0;

    start = omp_get_wtime();

    #pragma omp parallel
    {
        double x;
        
        #pragma omp for reduction(+ : sum)
        for (i = 0; i < num_steps; i++)
        {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }
    pi = step * sum;

    // Get end time
    end = omp_get_wtime();

    // Calculate elapsed (it's in seconds)
    elapsed = end - start;

    printf("Integral value: %lf\n", pi);
    printf("Time taken: %lf s\n", elapsed);

    return 0;
}
