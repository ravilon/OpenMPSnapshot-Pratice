/*
 * 13-PrivateVariables-OpenMP.c
 *
 *  Created on: 13 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

#define N 10

int main(int argc, char *argv[]) {
	int i;
	int sum = 0;

	#pragma omp parallel private(i) reduction(+:sum)
	{
		#pragma omp for
		//Each thread has its own private variable i and local_sum.
		for (i = 0; i < N; i++) {
			int tid = omp_get_thread_num();
			int local_sum = i * tid;
			//sum is a shared variable. It is updated in a safe way using the reduction clause.
			sum += local_sum;
		}
	}

	printf("The sum is %d\n", sum);
	return 0;
}
