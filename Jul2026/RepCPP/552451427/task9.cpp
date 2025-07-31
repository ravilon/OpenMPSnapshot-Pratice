#include <stdio.h>
#include <omp.h>
#include <iostream>

const int ROWS = 6;
const int COLUMNS = 8;
const int RAND_MAX_VALUE = 1000;

int main() {
    int** matrix = new int*[ROWS];
    int min = INT32_MAX;
    int max = INT32_MIN;

    srand(time(NULL));

    for (int i = 0; i < ROWS; i++) {
        matrix[i] = new int[COLUMNS];
        for(int j = 0; j < COLUMNS; j++) {
            matrix[i][j] = rand() % RAND_MAX_VALUE;
        }
    }

    printf("Matrix initialization done. Check how it looks like: \n");
    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLUMNS; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    printf("\nCalculating max and min value.\n");

    double start = omp_get_wtime();

    #pragma omp parallel shared(matrix, max, min)
    #pragma omp for
    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLUMNS; j++) {
            #pragma omp critical
            {
                if (matrix[i][j] > max) {
                    max = matrix[i][j];
                }
            }
            #pragma omp critical
            {
                if (matrix[i][j] < min) {
                    min = matrix[i][j];
                }
            }
        }
    }

    double time = omp_get_wtime() - start;

    printf("Max value = %d, min = %d. \nElapsed time: %.6f\n", max, min, time);

    return 0;
}
