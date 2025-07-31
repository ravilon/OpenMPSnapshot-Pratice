#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
	omp_set_num_threads(12);
	int nThreads, threadNum;
	double start_time = omp_get_wtime();

	#pragma omp parallel private(nThreads, threadNum)
	{
		nThreads = omp_get_num_threads();
		threadNum = omp_get_thread_num();

		printf("OpenMP thread %d from %d threads \n", threadNum, nThreads);
	}

	double finish_time = omp_get_wtime();
	double result_time = finish_time - start_time;

	printf("Time: %.32f\n", result_time);

	return 0;
}
