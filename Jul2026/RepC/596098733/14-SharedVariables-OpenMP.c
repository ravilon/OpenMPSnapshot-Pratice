/*
 * 14-SharedVariables-OpenMP.c
 *
 *  Created on: 14 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 4
#define N 100

int shared_counter; // shared variable

int main(int argc, char *argv[]) {

	omp_set_num_threads(NUM_THREADS); // set number of threads
	shared_counter = 0; // initialize shared variable

	#pragma omp parallel shared(shared_counter)
	{
		int id = omp_get_thread_num(); // get thread id
		int i;
		for (i = 0; i < N; i++) {
			#pragma omp critical
			{
				shared_counter++; // increment shared variable
			}
		}
		printf("Thread %d: shared_counter = %d\n", id, shared_counter);
	}

	return 0;
}
