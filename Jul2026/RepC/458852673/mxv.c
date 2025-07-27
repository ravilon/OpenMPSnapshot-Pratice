#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void mxv(int m, int n, int *matrix, int *vector, int *result);
void generateMatrix(int m, int n, int *matrix);
void generateVector(int m, int *vector);
void printMatrix(int m, int n, int *matrix);
void printVector(int m, int *vector);

int main(int argc, char **argv){
    srand(time(NULL));

    if(!argv[1] || !argv[2]){
        printf("Enter the matrix row and column sizes when calling the command (e.g ./file.out m n)\n");
        exit(1);
    }

    int m = atoi(argv[1]), n = atoi(argv[2]);

    int *B = (int *)malloc(m * n * sizeof(int));
    int *a = (int *)malloc(m * 1 * sizeof(int));
    int *r = (int *)calloc(m * 1, sizeof(int));

    double start = 0, end = 0;

    generateMatrix(m, n, B);
    generateVector(m, a);

    // printf("B:\n");
    // printMatrix(m, n, B);
    // printf("a:\n");
    // printVector(m, a);
	start = omp_get_wtime();
    mxv(m, n, B, a, r);
    end = omp_get_wtime();
    // printf("r:\n");
    // printVector(m, r);

    printf("%f\n", end - start);

    free(B);
    free(a);
    free(r);

    return 0;
}

void mxv(int m, int n, int *matrix, int *vector, int *result){
    int i = 0, j = 0;

    #pragma omp parallel for default(none) \
        shared(m, n, matrix, vector, result) private(i, j)
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            result[i] += matrix[i * n + j] * vector[i];
        }
    }
}

void generateMatrix(int m, int n, int *matrix){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            matrix[i * n + j] = rand() % 10;
        }
    }
}

void generateVector(int m, int *vector){
    for(int i = 0; i < m; i++){
        vector[i] = rand() % 10;
    }
}

void printMatrix(int m, int n, int *matrix){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void printVector(int m, int *vector){
    for(int i = 0; i < m; i++){
        printf("%d ", vector[i]);
        printf("\n");
    }
}