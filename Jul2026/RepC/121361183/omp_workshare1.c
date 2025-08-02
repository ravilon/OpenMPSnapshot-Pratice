/******************************************************************************
* OpenMP Example - Loop Work-sharing - C/C++ Version
* FILE: omp_workshare1.c
* DESCRIPTION:
*   In this example, the iterations of a loop are scheduled dynamically
*   across the team of threads.  A thread will perform CHUNK iterations
*   at a time before being scheduled for the next CHUNK of work.
* SOURCE: Blaise Barney  5/99
* LAST REVISED: 
******************************************************************************/

#include <omp.h>
#define CHUNK   10
#define N       100

main ()  {

int nthreads, tid, i, n, chunk;
float a[N], b[N], c[N];

/* Some initializations */
for (i=0; i < N; i++)
  a[i] = b[i] = i * 1.0;
n = N;
chunk = CHUNK;

#pragma omp parallel shared(a,b,c,n,chunk) private(i,nthreads,tid)
  {
  tid = omp_get_thread_num();

  #pragma omp for schedule(dynamic,chunk)
  for (i=0; i < n; i++)
    {
    c[i] = a[i] + b[i];
    printf("tid= %d i= %d  c[i]= %f\n", tid,i,c[i]);
    }

  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }

  }  /* end of parallel section */

}

