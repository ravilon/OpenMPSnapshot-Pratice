/*
* 06-ParallelForLoop-OpenMP.c
*
*  Created on: 6 feb. 2023
*      Author: Jose ngel Gumiel
*/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>	//Needed for tick counting.

#define N 200000000

static unsigned long long start, end;


int main() {
int i;
double *a = (double*)malloc(N * sizeof(double));

int num_threads = omp_get_num_threads();

// Initialize array with some values
for (i = 0; i < N; i++) {
a[i] = i * 0.5;
}

// Parallelize the for loop
start = __rdtsc();
#pragma omp parallel for num_threads(6)
for (i = 0; i < N; i++) {
a[i] = a[i] * a[i];
}
end = __rdtsc();

// Print the result
printf("Result: %lf\n", a[N-1]);
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));


// Initialize array again for serial test.
for (i = 0; i < N; i++) {
a[i] = i * 0.5;
}
// In serial the for loop
start = __rdtsc();
for (i = 0; i < N; i++) {
a[i] = a[i] * a[i];
}
end = __rdtsc();

// Print the result
printf("Result: %lf\n", a[N-1]);
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));


// Free memory
free(a);

return 0;
}
