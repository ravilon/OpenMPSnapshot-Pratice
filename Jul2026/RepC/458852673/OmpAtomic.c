#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
	long long int n = 10e6 * atoi(argv[1]), i = 0, sum = 0;
	double start = 0, end = 0;
	long long int *list = (long long int*) calloc(n, sizeof(long long int));

	start = omp_get_wtime();

	for(int j = 0; j < n; j++) list[j] = 1;

	#pragma omp parallel for private(i)
	for(i = 0; i < n; i++)
		#pragma omp atomic
		sum += list[i];

	end = omp_get_wtime();

	printf("OmpAtomic\ntime: %f\nsum: %lld\nn: %lld \n", end - start, sum, n);

	return 0;
}

long double f(){

}
