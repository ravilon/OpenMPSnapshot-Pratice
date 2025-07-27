/*
 * 27-ParallelDo-OpenMP.c
 *
 *  Created on: 27 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

int WinMain() {
    int n = 10;
    int a[n], b[n], c[n];
    int i;

    // Initialize arrays
    for (i = 0; i < n; i++) {
        a[i] = i;
        b[i] = n - i;
    }

    // Perform computation in parallel
    #pragma omp parallel shared(a, b, c) private(i)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
            printf("Thread %d computes c[%d] = %d\n", omp_get_thread_num(), i, c[i]);
        }
    }

    return 0;
}
