#include <stdio.h>
#include <omp.h>
#include <math.h>

#define n 30
#define chunk_size 8

int main(){
    int i;
    int a[n], b[n], c[n];


    for (i = 0; i < n; i++){
        a[i]=i;
        b[i]=i;
    }

	double start_time = omp_get_wtime();
	#pragma omp parallel num_threads(4)
	{
	      #pragma omp for schedule(static, chunk_size)
	      //#pragma omp for schedule(dynamic, chunk_size)
	      //#pragma omp for schedule(guided, chunk_size)
	      //#pragma omp for schedule(auto)

	      for (i = 0; i < n; i++){
	          c[i] = a[i]+b[i];
	          printf("Thread %d \t iteracao %d\n",omp_get_thread_num(),i);
	      }
	}

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

	printf("\nTempo total de execucao: %f segundos\n", elapsed_time);

    return 0;
}