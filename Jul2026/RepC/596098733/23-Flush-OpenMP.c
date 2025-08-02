/*
 * 23-Flush-OpenMP.c
 *
 *  Created on: 23 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

int main() {
	int shared_var = 0;
	int tid, local_var;

	#pragma omp parallel private(local_var, tid)
	{
		tid = omp_get_thread_num();

		// Each thread updates its own copy of local_var
		local_var = tid + 1;

		// Flush the write buffer to ensure visibility of updates to shared_var
		#pragma omp flush(shared_var)

		// Update the shared variable and print its value
		#pragma omp critical
		{
			shared_var += local_var;
			printf("Thread %d updated shared_var to %d\n", tid, shared_var);
		}

		// Flush the write buffer again to ensure visibility of updates to shared_var
		#pragma omp flush(shared_var)
	}

	printf("Final value of shared_var is %d\n", shared_var);

	return 0;
}
