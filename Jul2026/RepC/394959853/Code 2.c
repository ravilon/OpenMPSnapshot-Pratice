#include <omp.h>
#include<stdio.h>
int main (){
int nthreads, tid;
//creating threads and printing their respective thread ID's
#pragma omp parallel private(tid)
{

tid = omp_get_thread_num();
printf("Hello World from thread = %d\n",tid);
if (tid == 0) { 
nthreads =omp_get_num_threads(); 
printf("Number of threads = %d\n", nthreads); 
} 
}
}
