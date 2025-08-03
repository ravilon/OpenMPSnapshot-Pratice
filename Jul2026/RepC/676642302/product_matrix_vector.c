#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "ctimer.h"

#define M 6400
#define N 3200
#define NUM_BLOCKS 4

// MATRIX-VECTOR PRODUCT    x = A * b
int main(int argc, char **argv) {

// Variables declaration
int i, j, nThreads, threadId;
double *A, *b;
double *xSeq, *xOmp, *xOmp2;
double t1, t2, tucpu, tscpu;
double sum;

A = malloc(M * N * sizeof(double));
b = malloc(N * sizeof(double));
xSeq = malloc(M * sizeof(double));
xOmp = malloc(M * sizeof(double));
xOmp2 = malloc(M * sizeof(double));

srand(time(NULL));

printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
printf("\nProgram that computes the Matrix-Vector product.\n");
printf("--------\n");

// Data generation
for (i = 0; i < N; i++) {
for (j = 0; j < M; j++) {
*(A + i * M + j) = rand() % NUM_BLOCKS;
}
*(b + i) = rand() % NUM_BLOCKS;
}

// Matrix-Vector sequential product
printf("Sequential matrix-vector product\n");
printf("--------\n");
ctimer(&t1, &tucpu, &tscpu);

for (i = 0; i < M; i++) {
sum = 0.;
for (j = 0; j < N; j++) {
sum += *(A + i * N + j) * *(b + j);
}
*(xSeq + i) = sum;
}

// Results output
ctimer(&t2, &tucpu, &tscpu);
printf("MatxVec sequential product x[5] = %f\n", xSeq[5]);
printf("--------\n");
printf("Time:\t%f seconds\n", (float)(t2 - t1));

// Matrix-vector parallel product, row interleaving
printf("Parallel matrix-vector product: row interleaving\n");
printf("--------\n");
ctimer(&t1, &tucpu, &tscpu);
#pragma omp parallel default(shared) private(i, j, sum, threadId)
{
nThreads = omp_get_num_threads();
threadId = omp_get_thread_num();
for (i = threadId; i < M; i += nThreads) {
sum = 0.;
for (j = 0; j < N; j++) {
sum += *(A + i * N + j) * *(b + j);
}
*(xOmp + i) = sum;
}
}

// Results output
ctimer(&t2, &tucpu, &tscpu);
printf("Matrix-Vector parallel product ends\n");
printf("--------\n");
printf("MatxVec parallel product x[5] = %f\n", xOmp[5]);
printf("--------\n");
printf("Time:\t%f seconds\n", (float)(t2 - t1));

// Matrix-vector parallel product, row blocks
printf("Parallel matrix-vector product: row blocks\n");
printf("--------\n");
const int blockSize = M / NUM_BLOCKS;
omp_set_num_threads(NUM_BLOCKS);
ctimer(&t1, &tucpu, &tscpu);
#pragma omp parallel  private(threadId, i, j, sum) shared(A, b, xOmp2)
{
threadId = omp_get_thread_num();
for (i = 0; i < blockSize; i++) {
sum = 0.;
for (j = 0; j < N; j++) {
sum += *(A + N * (blockSize * threadId + i) + j) * *(b + j);
}
*(xOmp2 + blockSize * threadId + i) = sum;
}
}

// Results output
ctimer(&t2, &tucpu, &tscpu);
printf("Matrix-Vector parallel product 2 ends\n");
printf("--------\n");
printf("MatxVec parallel product 2 x[5] = %f\n", xOmp2[5]);
printf("Time:\t%f seconds\n", (float)(t2 - t1));

printf("Checking\n");
sum = 0.;
for (i = 0; i < M; i++) {
sum += (xSeq[i] - xOmp[i]) * (xSeq[i] - xOmp[i]);
}
sum = sqrt(sum);
printf("Differential norm SEQUENTIAL - PARALLEL = %f\n", sum);

printf("--------\n");
printf("It's over!\n");

printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");

return 0;
}
