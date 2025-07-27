/*
 * 21-NestedParallelism-OpenMP.c
 *
 *  Created on: 21 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

void nested_parallel() {
	#pragma omp parallel num_threads(2)
	{
		printf("Outer thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());

		#pragma omp parallel num_threads(2)
		{
			printf("Inner thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
		}
	}
}

int main() {
	nested_parallel();
	return 0;
}
