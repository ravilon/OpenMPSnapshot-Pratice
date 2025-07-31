#include <stdio.h>
#include <omp.h>
#include <iostream>

const int ROWS = 5;
const int COLUMNS = 5;
const int RAND_MAX_VALUE = 100;

int main() {
    int** matrix = new int*[ROWS];
    int max = INT32_MIN;
    int mod = 7;

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

    printf("\nCalculating max value that is multiple of %d.\n", mod);

    double start = omp_get_wtime();

    #pragma omp parallel shared(matrix, max, mod) num_threads(9)
    #pragma omp for
    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLUMNS; j++) {
            #pragma omp critical
            {
                if (matrix[i][j] % mod == 0 && matrix[i][j] > max) {
                    printf("    Hello from thread №%d, I found new value: %d. Total threads: %d\n", omp_get_thread_num(), matrix[i][j], omp_get_num_threads());
                    max = matrix[i][j];
                }
            }
        }
    }

    double time = omp_get_wtime() - start;

    if (max == INT32_MIN) {
        printf("There is no multiplies of %d in the array \n\n", mod);
    } else {
        printf("Result max value = %d. \nElapsed time: %.6f\n\n", max, time);
    }

    return 0;
}
