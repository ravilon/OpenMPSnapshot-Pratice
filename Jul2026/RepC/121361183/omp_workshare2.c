/******************************************************************************
* OpenMP Example - Sections Work-sharing - C/C++ Version
* FILE: omp_workshare2.c
* DESCRIPTION:
*   In this example, the iterations of a loop are split into two different 
*   sections.  Each section will be executed by one thread.  Extra threads
*   will not participate in the sections code.
* SOURCE: Blaise Barney  5/99
* LAST REVISED: 
******************************************************************************/

#include <omp.h>
#define N     50

main ()
{

int i, n, nthreads, tid;
float a[N], b[N], c[N];

/* Some initializations */
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
n = N;

#pragma omp parallel shared(a,b,c,n) private(i,tid,nthreads)
{
tid = omp_get_thread_num();
printf("Thread %d starting...\n",tid);

#pragma omp sections nowait
{

#pragma omp section
for (i=0; i < n/2; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n",tid,i,c[i]);
}

#pragma omp section
for (i=n/2; i < n; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n",tid,i,c[i]);
}

}  /* end of sections */

if (tid == 0)
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}

}  /* end of parallel section */

}

