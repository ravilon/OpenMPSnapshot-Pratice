#include <stdio.h>
#include <omp.h>
#include <iostream>

using namespace std;

const int SIZE = 10;
const int RAND_MAX_VALUE = 10;

int main() {
    int** matrix = new int*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        matrix[i] = new int[SIZE];
    }

    int* vector = new int[SIZE];
    int* result = new int[SIZE];

    srand(time(NULL));

    double start = omp_get_wtime();
    #pragma omp parallel shared(vector, matrix)
    #pragma omp for 
    for (int i = 0; i < SIZE; i++) {
        vector[i] = rand() % RAND_MAX_VALUE;

        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % RAND_MAX_VALUE;
        }
    }

    printf("Initialization of arrays done in %.6f seconds.\n", (omp_get_wtime() - start));

    // for (int i = 0; i < SIZE; i++) {
    //     for(int j = 0; j < SIZE; j++) {
    //         printf("%d ", matrix[i][j]);
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < SIZE; i++) {
    //     printf("%d\n", vector[i]);
    // }

    printf("1. Sequential algorithm.\n");
    start = omp_get_wtime();

    for (int i = 0; i < SIZE; i++) {
        result[i] = 0;

        for(int j = 0; j < SIZE; j++) {
            result[i] += (matrix[i][j] * vector[j]);
        }
    }
    printf("     Time: %.6f\n", (omp_get_wtime() - start));

    
    // for (int i = 0; i < SIZE; i++) {
    //     printf("%d\n", result[i]);
    // }


    printf("\n2. Parallel algorithm.\n");
    start = omp_get_wtime();

    #pragma omp parallel shared(vector, matrix, result)
    #pragma omp for
    for (int i = 0; i < SIZE; i++) {
        result[i] = 0;

        for(int j = 0; j < SIZE; j++) {
            result[i] += (matrix[i][j] * vector[j]);
        }
    }
    
    printf("     Time: %.6f\n", (omp_get_wtime() - start));

    
    // for (int i = 0; i < SIZE; i++) {
    //     printf("%d\n", result[i]);
    // }

    for (int i = 0; i < SIZE; i++) 
        delete[] matrix[i];
    delete[] matrix;
    delete[] vector;

    return 0;
}