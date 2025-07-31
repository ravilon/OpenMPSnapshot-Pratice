#include <omp.h>
#include <cstdlib>
#include <time.h>
#include "stdio.h"

#define CHUNK 100 
#define NMAX 3200000
#define Q 28
#define NUM_OF_THREADS 4 // 4, 8, 16
#define ITERATIONS 12

int main() {
    double a[NMAX], b[NMAX], sum[NMAX];
    int  i, step, t, j;
    omp_set_num_threads(NUM_OF_THREADS);
    for (i = 0; i < NMAX; ++i) {
        a[i] = i;
        b[i] = i;
    }
    double st_time, end_time, time_static = 0.0, time_dynamic = 0.0, time_quided = 0.0;

    for (t = 0; t < ITERATIONS; t++) {
        for (step = 0; step < 3; step++) {
            srand(time(0));
            st_time = omp_get_wtime();
#pragma omp parallel for shared (a,b,sum) private(i, j) schedule(static, CHUNK) if (step == 0)
            for (i = 0; i < NMAX; ++i) {
                for (j = 0; j < Q; ++j) {
                    sum[i] = a[i] + b[i];
                }
            }
#pragma omp parallel for shared (a,b,sum) private(i, j) schedule(dynamic, CHUNK) if (step == 1)
            for (i = 0; i < NMAX; ++i) {
                for (j = 0; j < Q; ++j) {
                    sum[i] = a[i] + b[i];
                }
            }
#pragma omp parallel for shared (a,b,sum) private(i, j) schedule(guided, CHUNK) if (step == 2)
            for (i = 0; i < NMAX; ++i) {
                for (j = 0; j < Q; ++j) {
                    sum[i] = a[i] + b[i];
                }
            }

            end_time = omp_get_wtime();
            if (step == 0)
                time_static += end_time - st_time;
            else if (step == 1)
                time_dynamic += end_time - st_time;
            else if (step == 2)
                time_quided += end_time - st_time;
        }

    }
    printf("Total %d process", NUM_OF_THREADS);
    printf(" with Q = %d", Q);
    printf("\nSTATIC time of work is %f seconds", time_static / ITERATIONS);
    printf("\nDYNAMIC time of work is %f seconds", time_dynamic / ITERATIONS);
    printf("\nQUIDED time of work is %f seconds", time_quided / ITERATIONS);
    return 0;
}
