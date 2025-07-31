#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ITERATIONS 200000000

/*
kompiliert mit:
gcc -std=c99 -fopenmp -O3 -o task1 task1.c
(module load gcc/8.2.0)

ausgeführt als Job mit:
export OMP_NUM_THREADS=4
export OMP_PLACES=cores 
export OMP_PROC_BIN=close
./task1
export OMP_PROC_BIN=spread
./task1
export OMP_PROC_BIN=master
./task1
*/

int main(void){
const int threads_num = 4;

//printf("Total threads: %d\n",threads_num);

int inc = 0;
double wtime = omp_get_wtime();
#pragma omp parallel num_threads(threads_num)
{
#pragma omp for
for(long i = 0;i < ITERATIONS; i++){	
#pragma omp atomic
inc++;
}	
int threadnum = omp_get_thread_num();
printf("Thread: %d\n", threadnum);

}
wtime = omp_get_wtime() - wtime;

printf("Increment is at %d after %d iterations.\n",inc, ITERATIONS);
printf("Time taken %f\n", wtime );

}
