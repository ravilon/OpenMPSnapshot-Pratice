#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>


int main(){

#define max_val 10

#define min_val 1

//Initializing Variables
int dimA =0,dimB =0,dimC =0;
int	thread_ID, numThreads, i, j, k,N,s,sum=0;
int num_threads =0;
double timetaken = 0, timetaken2 = 0,timetaken2=0;


//Input matrix size and blocks
	printf("Enter the dimension N*N: ");
	scanf("%d" , &N);
	printf("Enter the block_size of Matrix: ");
	scanf("%d" , &s);

//Initialize all the matrix used for Matrix Multiplication and comparison
	int (*matrix1)[N] = malloc(N * sizeof(*matrix1));
    int (*matrix2)[N] = malloc(N * sizeof(*matrix2));
    int (*matrixC)[N] = malloc(N * sizeof(*matrixC));
    int (*matrixOp1)[N] = malloc(N * sizeof(*matrixOp1));
    int (*matrixOp2)[N] = malloc(N * sizeof(*matrixOp2));
	
//Initializing matrix1 and matrix 2 --> Input Matrix


	for(i=0;i<N;i++){
		for(j=0; j<N ;j++){
			matrix1[i][j] = rand() % max_val + min_val;
		}
	}
		

	for(i=0;i<N;i++){
		for(j=0; j<N;j++){
			matrix2[i][j] = rand() % max_val + min_val;
		}
	}
	

// OMP Code		
struct timeval t0,t1;
gettimeofday(&t0, 0);

//Block multiplication
int i1=0,k1=0,j1=0;
#pragma omp parallel for shared(matrix1,matrix2,matrixOp1,N,s) private(i,j,k,i1,j1,k1,sum) schedule(auto) num_threads(4) collapse(3) // **OPENMP CODE**
for (i=0; i<N; i+=s)
    for (j=0; j<N; j+=s)
      for (k=0; k<N; k+=s)
        for (i1=i;i1<i+s;i1++)
          for (j1=j;j1<j+s;j1++)
            {
              int sum=0;
              for (k1=k;k1<k+s;k1++)
                {
                  sum+=matrix1[i1][k1]*matrix2[k1][j1];
                }
              matrixOp1[i1][j1]+=sum;
            }

gettimeofday(&t1, 0);
timetaken = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

//Serial multiplication
 double matrix_mult_serial(int N);
{
	int i,j,k;
	//double st=omp_get_wtime();
	//DWORD start,end;
	//start = GetTickCount();
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			
			matrixOp2[i][j] = 0.0;
			

			for(k=0;k<N;k++)
			{
				matrixOp2[i][j] +=  matrix1[i][k]*matrix2[k][j];
			}
		}
	}
	//end = GetTickCount();
	//return matrixOp2[i][j];
	//double en=omp_get_wtime();
	//printf("Total time taken by Serial multiplication: %lf\n",omp_get_wtime()-st);
	gettimeofday(&t1, 0);
	timetaken2 = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	//printf("Total time taken by Serial multiplication: %lf\n",en-st);
	
}


/*		
	printf("Result A is : \n");
	for(int i=0;i<N;i++){
		for(int j =0;j<N;j++){
			printf("%d	",matrix1[i][j]);
		}
		printf("\n");
	}
	
	printf("Result B is : \n");
	for(int i=0;i<N;i++){
		for(int j =0;j<N;j++){
			printf("%d	",matrix2[i][j]);
		}
		printf("\n");
	}
	

	printf("Result C for Parallel : \n");
	for(int i=0;i<N;i++){
		for(int j =0;j<N;j++){
			printf("%d	",matrixOp1[i][j]);
		}
		printf("\n");
	}
*/	
	
	printf("Elapsed time for OpenMP Parallel: %f \n ", timetaken);
	
	
		
/*	

	printf("Result C for Sequential : \n");
	for(int i=0;i<N;i++){
		for(int j =0;j<N;j++){
			printf("%d	",matrixOp2[i][j]);
		}
		printf("\n");
	} 
*/	
	printf("Elapsed time for Sequential : %f \n ", timetaken2);
		
	
	
	
	
	
	
	return (0);
}
