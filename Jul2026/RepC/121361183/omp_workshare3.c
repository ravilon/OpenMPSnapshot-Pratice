/******************************************************************************
* OpenMP Example - Combined Parallel Loop Work-sharing - C/C++ Version
* FILE: omp_workshare3.c
* DESCRIPTION:
*   This example attempts to show use of the parallel for construct.  However
*   it will generate errors at compile time.  Try to determine what is causing
*   the error.  See omp_workshare4.c for a corrected version.
* SOURCE: Blaise Barney  5/99
* LAST REVISED: 
******************************************************************************/

#include <omp.h>
#define N       50
#define CHUNK   5

main ()  {

int i, n, chunk, tid;
float a[N], b[N], c[N];

/* Some initializations */
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
n = N;
chunk = CHUNK;

#pragma omp parallel for      shared(a,b,c,n)             private(i,tid)              schedule(static,chunk)
{
tid = omp_get_thread_num();
for (i=0; i < n; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n", tid, i, c[i]);
}
}  /* end of parallel for construct */

}
