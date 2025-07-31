/*
* Filename: hello_openmp.c
* Author: Pradeep Singh
* Email: psingh2@sdsu.edu
* Date: 3/8/2018
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{

int nthreads, tid;

/* Fork a team of threads */
#pragma omp parallel private(nthreads, tid)
{

/* Obtain thread number */
tid = omp_get_thread_num();
printf("Hello World from thread = %d\n", tid);

/* Only master thread does this */
if (tid == 0)  {
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
}
}
