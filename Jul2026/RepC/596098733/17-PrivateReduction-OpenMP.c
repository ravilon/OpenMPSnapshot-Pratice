/*
 * 17-PrivateReduction-OpenMP.c
 *
 *  Created on: 17 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

#define N 1000

int main() {
	int i = 0;
	int a[N], sum = 0;

	// Initialize the array with some values
	for (int i = 0; i < N; i++) {
		a[i] = i + 1;
	}

	#pragma omp parallel for private(i) reduction(+:sum)
	//#pragma omp parallel for reduction(+:sum)	/*Also valid. Reduction makes var i private, assigning each thread a different range.*/
	for (i = 0; i < N; i++) {
		sum += a[i];
	}

	printf("The sum of the array is: %d\n", sum);

	return 0;
}
