/******************************************************************************
* ЗАДАНИЕ: bugged4.c
* ОПИСАНИЕ:
*   Очень простая программа с segmentation fault.
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1048

int main(int argc, char *argv[]) {
    int nthreads, tid, i, j;
    // double a[N][N]; память ограничена размером стека
    // есть ограничение на статический массив
    double **a = (double **) malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) {
        a[i] = (double *) malloc(sizeof(double) * N);
    }
#pragma omp parallel shared(nthreads) private(i, j, tid) //private(i, j, tid, a)
    // а может не должен быть приватным, очень много памяти выделяется
    {
        tid = omp_get_thread_num();
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d starting...\n", tid);
        #pragma omp for
        // параллелим цикл, чтобы правильно все работало
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                a[i][j] = tid + i + j;
        printf("Thread %d done. Last element= %f\n", tid, a[N - 1][N - 1]);
    }
    for (int i = 0; i < N; i++) {
        free(a[i]);
    }
    free(a);
}