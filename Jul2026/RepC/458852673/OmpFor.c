#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
	omp_set_num_threads(atoi(argv[2]));

	long long int n = 10e6 * atoi(argv[1]), i = 0, sum = 0, parcial_sum = 0;
	long long int *list = (long long int*) malloc(n * sizeof(long long int));
	long long int *threads = (long long int*) calloc(omp_get_max_threads(), sizeof(long long int));
	double start = 0, end = 0;

	start = omp_get_wtime();

	for(int j = 0; j < n; j++) list[j] = 1;

	#pragma omp parallel private(i, parcial_sum)
	{
		parcial_sum = 0;

		#pragma omp for
		for(i = 0; i < n; i++) threads[omp_get_thread_num()]++;

		#pragma omp atomic
		sum += parcial_sum;

	}

	end = omp_get_wtime();

	printf("OmpFor\ntime: %f\nn: %lld\nt: %d\n", end - start, n, atoi(argv[2]));

	// for(int j = 0; j < omp_get_max_threads(); j++) printf("Thread: %d, i: %lld\n", j, threads[j]);

	return 0;
}
