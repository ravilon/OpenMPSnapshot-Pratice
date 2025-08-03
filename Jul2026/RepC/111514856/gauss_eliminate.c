/* Gaussian elimination code.
* By Greg Matthews 1/30/17
* Skeleton Code By: Prof. Naga Kandasamy
*
* Compile as follows: 
* gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -O3 -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
int gauss_eliminate_using_openmp(const Matrix, Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);


int 
main(int argc, char** argv) {
if(argc > 1){
printf("Error. This program accepts no arguments. \n");
exit(0);
}	

/* Allocate and initialize the matrices. */
Matrix  A;                                              /* The N x N input matrix. */
Matrix  U;                                              /* The upper triangular matrix to be computed. */

srand(time(NULL));

A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */

/* Gaussian elimination using the reference code. */
Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
struct timeval start, stop;	
gettimeofday(&start, NULL);

printf("Performing gaussian elimination using the reference code. \n");
int status = compute_gold(reference.elements, A.elements, A.num_rows);

gettimeofday(&stop, NULL);
printf("CPU run time = %0.3f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

if(status == 0){
printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
exit(0);
}
status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
if(status == 0){
printf("The upper triangular matrix is incorrect. Exiting. \n");
exit(0); 
}
printf("Gaussian elimination using the reference code was successful. \n");

/* WRITE THIS CODE: Perform the Gaussian elimination using the multi-threaded OpenMP version. 
* The resulting upper triangular matrix should be returned in U
* */
/* check if the OpenMP result is equivalent to the expected solution. */

//Timing Step
struct timeval start_omp, stop_omp;
gettimeofday(&start_omp, NULL);
gauss_eliminate_using_openmp(A, U);
gettimeofday(&stop_omp, NULL);	

int size = MATRIX_SIZE*MATRIX_SIZE;
int res = check_results(reference.elements, U.elements, size, 0.001f);
printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");


if (res == 1)
printf("CPU run time using openmp =  %0.3f s. \n", (float)(stop_omp.tv_sec - start_omp.tv_sec + (stop_omp.tv_usec - start_omp.tv_usec)/(float)1000000));

free(A.elements); A.elements = NULL;
free(U.elements); U.elements = NULL;
free(reference.elements); reference.elements = NULL;

return 0;
}


int gauss_eliminate_using_openmp(const Matrix A, Matrix U)                  
/* Write code to perform gaussian elimination using OpenMP. */
{
unsigned int i, j, k;
int thread_count = 16;

// copying contents of A to U
for (i=0; i < A.num_rows; i++)
for(j=0; j < A.num_rows; j++)
U.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
// Gaussian elimination on U, reduce current row
for (k=0; k < A.num_rows; k++)
{
#		pragma omp parallel for num_threads(thread_count) default(none) private(j) shared(k,U,thread_count) schedule(dynamic, A.num_rows/thread_count)
for (j=(k+1); j < A.num_rows; j++)
{
// Division
U.elements[A.num_rows*k+j]=(float)(U.elements[A.num_rows*k+j]/U.elements[A.num_rows*k+k]);
}
U.elements[A.num_rows * k + k] = 1;
#		pragma omp parallel for num_threads(thread_count) default(none) private(i,j) shared(k,U,thread_count) schedule(dynamic, A.num_rows/thread_count)
for (i=(k+1); i < A.num_rows; i++)
{
for (j = (k+1); j< A.num_rows; j++)	
// Elimination
U.elements[A.num_rows*i+j]=U.elements[A.num_rows*i+j]-(U.elements[A.num_rows*i+k]*U.elements[A.num_rows*k+j]);			
U.elements[A.num_rows*i+k] = 0;
}
}
return 1;
}


int 
check_results(float *A, float *B, unsigned int size, float tolerance)   /* Check if refernce results match multi threaded results. */
{
for(int i = 0; i < size; i++)
if(fabsf(A[i] - B[i]) > tolerance)
return 0;

return 1;
}


/* Allocate a matrix of dimensions height*width. 
* If init == 0, initialize to all zeroes.  
* If init == 1, perform random initialization.
* */
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
Matrix M;
M.num_columns = M.pitch = num_columns;
M.num_rows = num_rows;
int size = M.num_rows * M.num_columns;
M.elements = (float*) malloc(size*sizeof(float));

for(unsigned int i = 0; i < size; i++){
if(init == 0) M.elements[i] = 0; 
else
M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
}

return M;
}	


float 
get_random_number(int min, int max){                                    /* Returns a random FP number between min and max values. */
return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
perform_simple_check(const Matrix M){                                   /* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
for(unsigned int i = 0; i < M.num_rows; i++)
if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;

return 1;
} 


