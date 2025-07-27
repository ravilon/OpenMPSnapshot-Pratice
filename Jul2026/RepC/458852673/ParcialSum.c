#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
	long long int n = 10e6 * atoi(argv[1]), i = 0, sum = 0, parcial_sum = 0;
	long long int *list = (long long int*) calloc(n, sizeof(long long int));
	double start = 0, end = 0;

	start = omp_get_wtime();

	for(int j = 0; j < n; j++) list[j] = 1;

	#pragma omp parallel private(i, parcial_sum)
	{
		parcial_sum = 0;

		#pragma omp for
		for(i = 0; i < n; i++)parcial_sum += list[i];

		#pragma omp atomic
		sum += parcial_sum;

	}

	end = omp_get_wtime();

	printf("ParcialSum\ntime: %f\nsum: %lld\nn: %lld \n", end - start, sum, n);

	return 0;
}
