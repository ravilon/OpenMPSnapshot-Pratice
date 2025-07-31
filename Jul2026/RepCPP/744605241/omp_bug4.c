/******************************************************************************
 * FILE: omp_bug4.c
 * DESCRIPTION:
 *   This very simple program causes a segmentation fault.
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main(int argc, char *argv[]) {
  int nthreads, tid, i, j;
  double a[N][N];

/*
 * NOTE    
 * Defining a double a[N][N] array in the local scope of the main() function. This array is allocated on the stack.
 * #pragma omp parallel line specifies private(i, j, tid, a), which means each thread gets its own private copy of a. 
 * This multiplies the stack size requirement.
 * The combined stack size requirement could far exceed the per-thread stack size limit, leading to a segmentation fault.
 */

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i, j, tid, a)
  {

    /* Obtain/print thread info */
    tid = omp_get_thread_num();
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d starting...\n", tid);

    /* Each thread works on its own private copy of the array */
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
        a[i][j] = tid + i + j;

    /* For confirmation */
    printf("Thread %d done. Last element= %f\n", tid, a[N - 1][N - 1]);

  } /* All threads join master thread and disband */
}
