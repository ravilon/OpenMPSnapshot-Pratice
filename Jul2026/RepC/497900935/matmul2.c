#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


int  main(){
	
	printf("Enter no of rows of matrix A ");
	int r1;
	scanf("%d", &r1);

	printf("Enter no of columns of matrix A ");
	int c1;
	scanf("%d", &c1);

	printf("Enter no of rows of matrix B ");
	int r2;
	scanf("%d", &r2);

	printf("Enter no of columns of matrix B ");
	int c2;
	scanf("%d", &c2);

	int A[r1][c1], B[r2][c2];

	//printf("Enter elements of matrix A \n");
	for(int i = 0 ; i<r1 ; i++){ 
		for(int j = 0 ; j<c1 ; j++){
			//scanf("%d", &A[i][j]);
			A[i][j] = i*c1 + j +1;
		}
	}

	//printf("Enter elements of matrix B \n");
	for(int i = 0 ; i<r2 ; i++){ 
		for(int j = 0 ; j<c2 ; j++){
			//scanf("%d", &B[i][j]);
			B[i][j] = i*c2 + j + 1;
		}
	}
	int C[r1][c2];
	printf("Enter No of threads ");
	int nthreads; scanf("%d", &nthreads);      

	omp_set_num_threads(nthreads);
	LIKWID_MARKER_INIT;

	#pragma omp parallel
	{
	    LIKWID_MARKER_REGISTER("Multiplication");
	}
	double start = omp_get_wtime();
	
	#pragma omp parallel
	{	
		LIKWID_MARKER_START("Multiplication");
		int id = omp_get_thread_num();
		#pragma omp for
			for(int i = id ; i<r1*c2 ; i+=nthreads){

				int row = i/c2;
				int col = i%c2;

				C[row][col] = 0;
		 
				for (int k = 0; k < r2; k++) {
				   C[row][col] += A[row][k] * B[k][col];
				}
			}
		LIKWID_MARKER_START("Multiplication");    
	}	

	double end = omp_get_wtime();
	LIKWID_MARKER_CLOSE;
   
	//printf("Resultant matrix\n");

	//for(int i = 0 ; i<r1 ; i++){ 
	//	for(int j = 0 ; j<c2 ; j++){
	//		printf("%d ", C[i][j]);
	//	}
	//	printf("\n");
	//}	

	printf("Time elapsed in matrix muliplication %f seconds\n", end - start);

	return 0;
}
