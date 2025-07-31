#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "vector.h"
#include "matrix.h"


int allocate_Matrix(Matrix* A, int M, int N){

A->M = M;
A->N = N;
A->data = malloc(sizeof(float)*M*N);

// if malloc failed
if(A->data == NULL)
return 1;

// if malloc succeeded, initialize entries to 0
memset(A->data, 0, sizeof(float)*M*N);

return 0;
}

int deallocate_Matrix(Matrix* A){

A->M = 0;
A->N = 0;

free(A->data);
A->data = NULL;

return 0;
}

int rand_fill_Matrix(Matrix* A){

int M = A->M;
int N = A->N;

srand(time(NULL));

for(int i=0; i<M; ++i){
for(int j=0; j<N; ++j){
A->data[i*N + j] = (float)rand() / (float)RAND_MAX;
}
}

return 0;
}

int matvec(Matrix* A, Vector* x, Vector* b){

int i, j;
int error_flag;

int M = A->M;
int N = A->N;

// Create local copies of the pointers
float* A_data = A->data;
float* x_data = x->data;
float* b_data = b->data;


// check input lengths
if(x->N != N)
return 1;
if(b->N != M)
return 1;

// actually loop through all rows and columns to do the matrix vector multiplication
#pragma omp parallel for shared(N,M,b_data,x_data,A_data) private(i,j)
for(i=0; i<M; i++){
b_data[i] = 0.0;
for(j=0; j<N; j++){
b_data[i] += A_data[i*N + j] * x_data[j];
}
}

return 0;
}

int matvec_triangular(Matrix* A, Vector* x, Vector* b){

int i, j;
int error_flag;

int M = A->M;
int N = A->N;

// Create local copies of the pointers
float* A_data = A->data;
float* x_data = x->data;
float* b_data = b->data;

// check input lengths
if(M != N)
return 1;
if(x->N != N)
return 1;
if(b->N != M)
return 1;

// actually loop through all rows and columns to do the matrix vector multiplication
#pragma omp parallel for shared(N,M,b_data,x_data,A_data) private(i,j)
for(i=0; i<M; i++){
b_data[i] = 0.0;
for(j=0; j<i; j++){
b_data[i] += A_data[i*N + j] * x_data[j];
}
}

return 0;
}



int matvec_triangular_guided(Matrix* A, Vector* x, Vector* b){

int i, j;
int error_flag;

int M = A->M;
int N = A->N;

// Create local copies of the pointers
float* A_data = A->data;
float* x_data = x->data;
float* b_data = b->data;

// check input lengths
if(M != N)
return 1;
if(x->N != N)
return 1;
if(b->N != M)
return 1;

// actually loop through all rows and columns to do the matrix vector multiplication
#pragma omp parallel for shared(N,M,b_data,x_data,A_data) private(i,j) schedule(guided)
for(i=0; i<M; i++){
b_data[i] = 0.0;
for(j=0; j<i; j++){
b_data[i] += A_data[i*N + j] * x_data[j];
}
}

return 0;
}

