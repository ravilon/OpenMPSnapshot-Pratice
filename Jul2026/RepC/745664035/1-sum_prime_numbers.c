#include <stdio.h>
#include <omp.h>
#define NUMBER 100000

int main()
{
    int i, j, val, total = 0, N = NUMBER;
    double start, end, time;

    // Sequential
    start = omp_get_wtime();
    for (i = 2; i <= N; i++)
    {
        val = 1;
        for (j = 2; j < i; j++)
        {
            if (i % j == 0)
            {
                val = 0;
                break;
            }
        }
        total = total + val;
    }
    end = omp_get_wtime();
    time = end - start;
    printf("Done. total = %d\n", total);
    printf("Sequential time %f seconds\n", time);

    // Parallel Reduction
    total = 0;
    start = omp_get_wtime();

#pragma omp parallel private(i, j, val) num_threads(4)
    {
#pragma omp for schedule(dynamic) reduction(+ : total)
        for (i = 2; i <= N; i++)
        {
            val = 1;
            for (j = 2; j < i; j++)
            {
                if (i % j == 0)
                {
                    val = 0;
                    break;
                }
            }
            total = total + val;
        }
    }
    end = omp_get_wtime();
    time = end - start;
    printf("Done. total = %d\n", total);
    printf("Parallel time reduction %f seconds\n", time);

    // Parallel with Reduction
    total = 0;
    start = omp_get_wtime();

#pragma omp parallel private(i, j, val) num_threads(4)
    {
        int sous_total = 0;
#pragma omp for schedule(dynamic) nowait
        for (i = 2; i <= N; i++)
        {
            val = 1;
            for (j = 2; j < i; j++)
            {
                if (i % j == 0)
                {
                    val = 0;
                    break;
                }
            }
            sous_total = sous_total + val;
        }

#pragma omp critical
        {
            total = total + sous_total;
        }
    }
    end = omp_get_wtime();
    time = end - start;
    printf("Done. total = %d\n", total);
    printf("Parallel time critical %f seconds\n", time);
    return (0);
}