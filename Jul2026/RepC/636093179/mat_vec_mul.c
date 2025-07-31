#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 10000

int main() {
	// int mat[N][N];
	// int vector[N];
	// int res[N]; 
	long **mat = (long **)malloc(N * sizeof(long *));
	long *vector = (long *)malloc(N * sizeof(long));
	long *res = (long *)malloc(N * sizeof(long));

	#pragma omp parallel num_threads(2)
	{
		#pragma omp for schedule(dynamic, 8000)
		for (int i = 0; i < N; i++) {
			mat[i] = (long *)malloc(N * sizeof(long));
		}

		#pragma omp for collapse(2) schedule(dynamic, 8000)
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				mat[i][j] = 1;
			}
		}
		#pragma omp for schedule(dynamic, 8000)
		for(int i = 0; i < N; i++) {
			vector[i] = 1;
		}
	}

	#pragma omp parallel num_threads(2)
	{
		#pragma omp for collapse(2) schedule(dynamic, 8000)
		for(int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				res[i] += mat[j][i] * vector[i];
			}
		}
	}
	// for(int i = 0; i < N; i++) {
	// 	printf("i = %d, vec[%d] = %ld\n", i, i, res[i]);
	// }
	// for(int i = 0; i < N; i++) {
	// 	for(int j = 0; j < n; j++) {
	// 		printf("i= %d, j= %d, %ld\n", i, j, mat[i][j]);
	// 	}
	// }

	// for(int i = 0; i < N; i++) {
	// 	printf("i = %d %ld\n", i, vector[i]);
	// }	
}