/*
 * 07-ParallelSection-OpenMP.c
 *
 *  Created on: 7 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>
#include <x86intrin.h>	//Needed for tick counting.

#define N 50000

static unsigned long long start, end;


int main(int argc, char *argv[]) {
  int i, j;
  int a[N], b[N], c[N];

  // Initialize arrays with random values
  srand(time(NULL));
  for (i = 0; i < N; i++) {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  // Get number of cores in the system
  //int num_threads = omp_get_num_procs();

  start = __rdtsc();
  #pragma omp parallel sections num_threads(4)
  {
    #pragma omp section
    {
      // Calculate sum of elements in a
      for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
      }
      printf("He terminado suma\n");
    }

    #pragma omp section
    {
      // Calculate product of elements in b
      for (j = 0; j < N; j++) {
        c[j] = a[j] * b[j];
      }
      printf("He terminado multiplicacin\n");
    }
  }
  end = __rdtsc();

  // Print time taken to perform the calculation
  printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));

  /*Serial*/
  start = __rdtsc();
  // Calculate sum of elements in a
  for (i = 0; i < N; i++) {
	c[i] = a[i] + b[i];
  }

  // Calculate product of elements in b
  for (j = 0; j < N; j++) {
	 c[j] = a[j] * b[j];
  }
  end = __rdtsc();

  printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));

  return 0;
}
