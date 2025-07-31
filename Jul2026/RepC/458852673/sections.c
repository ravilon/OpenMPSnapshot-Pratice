#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"

void printarVetor(int rank, int *A, int n);
void OddEvenSortParallel(int A[], int n);
void generateVector(int *vector, int m);

int main(int argc, char **argv){
int size;
int rank;
int nThreads = atoi(argv[2]);

omp_set_num_threads(nThreads);

int n = atoi(argv[1]);
int *A = (int *)malloc(sizeof(int) * n);

srand(time(NULL));
generateVector(A, n);

// printarVetor(0, A, n);

OddEvenSortParallel(A, n);

// printarVetor(0, A, n);

free(A);

return 0;
}

void OddEvenSortParallel(int A[], int n){
for(int i = 1; i <= n; i++){
#pragma omp parallel sections
{
#pragma omp section
{
if(i % 2 == 0){ // Even
for(int j = 0; j < (n/2); j++){
if(A[2 * j] > A[2 * j + 1]){
int temp = A[2 * j];
A[2 * j] = A[2 * j + 1];
A[2 * j + 1] = temp;
}
}
}
}

#pragma omp section
{
if(i % 2 == 1){ // Odd
for(int j = 0; j < (n/2) - 1; j++){
if(A[2 * j + 1] > A[2 * j + 2]){
int temp = A[2 * j + 1];
A[2 * j + 1] = A[2 * j + 2];
A[2 * j + 2] = temp;
}
}
}
}
}
}
}

void generateVector(int *vector, int m){
for(int i = 0; i < m; i++) vector[i] = rand() % m;
}

void printarVetor(int rank, int *A, int n){
printf("Processo %d\n", rank);
for(int i = 0; i < n; i++) printf("%d; ", A[i]);
printf("\n");
}