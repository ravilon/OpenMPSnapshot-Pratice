#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int thread_count, n;
    double comeco, fim;

    thread_count = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);

    comeco = omp_get_wtime();
#pragma omp parallel num_threads(thread_count)
    {
        int i;
        double my_sum = 0.0;

        for (i = 0; i < n; i++) {
#pragma omp atomic
            my_sum += sin(i);
        }
    }
    fim = omp_get_wtime();

    printf("Thread_count = %d, n = %d, Time = %e seconds\n", thread_count, n,
           fim - comeco);
    return 0;
}