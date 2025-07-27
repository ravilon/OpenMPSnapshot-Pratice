/*
 * 12-SingleThread-OpenMP.c
 *
 *  Created on: 12 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

int main() {
	#pragma omp parallel
	{
		#pragma omp single
		{
			//Only one thread (thread N) executes this line.
			printf("I am thread %d. There are %d threads.\n", omp_get_thread_num(), omp_get_num_threads());
		}
		//All threads, (including thread N ) executes this line.
		printf("I am thread %d. Hello World!\n", omp_get_thread_num());
	}
	return 0;
}
