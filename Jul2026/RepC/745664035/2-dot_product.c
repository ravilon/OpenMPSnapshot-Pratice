#include <stdio.h>
#include <omp.h>
#define SIZE 256

int main()
{
    double sum, a[SIZE], b[SIZE];
    sum = 0.;

    omp_set_num_threads(4);

    int NB = SIZE / 4;
    int rest = SIZE % 4;

#pragma omp parallel
    {
#pragma omp single
        {
            double sum_local = 0.;
            for (int j = 0; j < 3; j++)
            {
#pragma omp task
                {
                    for (size_t i = j * NB; i < (j + 1) * NB; i++)
                    {
                        a[i] = i * 0.5;
                        b[i] = i * 2.0;
                        sum_local = sum_local + a[i] * b[i];
                    }
#pragma omp critical
                    {
                        sum = sum + sum_local;
                    }
                }
            }

#pragma omp task
            {
                double sum_local = 0.;
                for (size_t i = 3 * NB; i < (4 * NB) + rest; i++)
                {
                    a[i] = i * 0.5;
                    b[i] = i * 2.0;
                    sum_local = sum_local + a[i] * b[i];
                }
#pragma omp critical
                {
                    sum = sum + sum_local;
                }
            }
        }
    }
    printf("sum = %g\n", sum);
    return 0;
}