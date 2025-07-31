#include <stdio.h>
#include <omp.h>
#define N 100



void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k) shared(A, B, C) default(none)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }}}}
        
    

int main() {
    int A[N][N], B[N][N], C[N][N];
    int i, j;
    double start_time, end_time;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0;
        }
    }
    


    // Parallel matrix multiplication with default scheduling
    start_time = omp_get_wtime();
    matrix_multiply(A, B, C);
    end_time = omp_get_wtime();
    printf("Parallel matrix multiplication with default scheduling: %f seconds\n", end_time - start_time);
    



    // Parallel matrix multiplication with static scheduling
    start_time = omp_get_wtime();
    #pragma omp parallel for private(i, j) shared(A, B, C) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end_time = omp_get_wtime();
    printf("Parallel matrix multiplication with static scheduling: %f seconds\n", end_time - start_time);
    



    // Parallel matrix multiplication with dynamic scheduling
    start_time = omp_get_wtime();
    #pragma omp parallel for private(i, j) shared(A, B, C) schedule(dynamic)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end_time = omp_get_wtime();
    printf("Parallel matrix multiplication with dynamic scheduling: %f seconds\n", end_time - start_time);
    
    return 0;
}
