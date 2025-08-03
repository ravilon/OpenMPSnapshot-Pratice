/*
* 05-MatrixTraspose-OpenMp.c
*
*  Created on: 5 feb. 2023
*      Author: Jose ngel Gumiel
*/


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>		//Contains definition for "NULL".
#include <time.h>
#include <omp.h>
#include <x86intrin.h>	//Needed for tick counting.

#define N 10000

static unsigned long long start, end;

int main(int argc, char *argv[]) {
int i, j, num_threads, tid;
int** A = (int**)malloc(N * sizeof(int*));
int** B = (int**)malloc(N * sizeof(int*));

for (i = 0; i < N; i++) {
A[i] = (int*)malloc(N * sizeof(int));
B[i] = (int*)malloc(N * sizeof(int));
}

// Initialize matrix A with random values
srand(time(NULL));
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
A[i][j] = rand() % 10;
}
}

// Get number of cores in the system
#pragma omp parallel
{
#pragma omp single
num_threads = omp_get_num_threads();
}
printf("Number of cores: %d\n", num_threads);

// PARALLEL: Transpose matrix A and store result in matrix B
start = __rdtsc();
#pragma omp parallel for private(i, j, tid) num_threads(num_threads)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
B[i][j] = A[j][i];
}
}
end = __rdtsc();
/*
// Print matrix A and matrix B
printf("Matrix A:\n");
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
printf("%d ", A[i][j]);
}
printf("\n");
}
printf("Matrix B:\n");
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
printf("%d ", B[i][j]);
}
printf("\n");
}
*/
// Print time taken to transpose the matrix
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));

// SERIAL: Transpose matrix A and store result in matrix B
start = __rdtsc();
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
B[i][j] = A[j][i];
}
}
end = __rdtsc();
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));

// Free memory
for (i = 0; i < N; i++) {
free(A[i]);
free(B[i]);
}
free(A);
free(B);

return 0;
}
