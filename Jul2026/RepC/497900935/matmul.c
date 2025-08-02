#include <stdio.h>
#include <omp.h>

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
 
	double start1 = omp_get_wtime();                    // without threads
	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			    C[i][j] = 0;
		 
			    for (int k = 0; k < r2; k++) {
				C[i][j] += A[i][k] * B[k][j];
			    }
		 
		}
	 
    	}
	double end1 = omp_get_wtime();
	printf("Enter No of threads ");
	int nthreads; scanf("%d", &nthreads);       // with threads

	omp_set_num_threads(nthreads);
	double start = omp_get_wtime();

	#pragma omp parallel
	{	
		 
		int id = omp_get_thread_num();

		for(int i = id ; i<r1*c2 ; i+=nthreads){

			int row = i/c2;
			int col = i%c2;

			C[row][col] = 0;
	 
			for (int k = 0; k < r2; k++) {
			   C[row][col] += A[row][k] * B[k][col];
			}
		}    
	}	

	double end = omp_get_wtime();
	
	//printf("Resultant matrix\n");

	//for(int i = 0 ; i<r1 ; i++){ 
	//	for(int j = 0 ; j<c2 ; j++){
	//		printf("%d ", C[i][j]);
	//	}
	//	printf("\n");
	//}	

	printf("\nTime elapsed in matrix muliplication without threads %f seconds\n", end1 - start1);	
	printf("Time elapsed in matrix muliplication with threads %f seconds\n", end - start);
}
