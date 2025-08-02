#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int** gerarMatriz(int n, int value);
void printarMatriz(int **A, int n);

int main(void){
    int n = 1e3;
    int **A = gerarMatriz(n, 2);
    int **B = gerarMatriz(n, 3);
    int **C = gerarMatriz(n, 0);

    int i, j, k;

    double start = omp_get_wtime();

    #pragma omp parallel for private(i, j, k) shared(A, B, C, n) schedule(dynamic, 5)
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            for(k = 0; k < n; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("end: %f\n", omp_get_wtime() - start);

    // #pragma omp parallel private(i, j, k) shared(A, B, C, n)
    // {
    //     #pragma omp for
    //     for(i = 0; i < n; i++){
    //         #pragma omp parallel for
    //         for(j = 0; j < n; j++){
    //             for(k = 0; k < n; k++){
    //                 C[i][j] = C[i][j] + A[i][k] * B[k][j];
    //             }
    //         }
    //     }
    // }

}

int** gerarMatriz(int n, int value){
    int **A = (int **)malloc(sizeof(int *) * n);

    for(int i = 0; i < n; i++){
        A[i] = (int *)malloc(sizeof(int) * n);
        for(int j = 0; j < n; j++){
            A[i][j] = value;
        }
    }

    return A; 
}

void printarMatriz(int **A, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
}