	#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MATRIX_SIZE 1000
#define BLOCK_SIZE 32 

int MatrixA[MATRIX_SIZE][MATRIX_SIZE];
int MatrixB[MATRIX_SIZE][MATRIX_SIZE];
int ResultMatrix[MATRIX_SIZE][MATRIX_SIZE];

void multiply_matrices(int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                ResultMatrix[i][j] += MatrixA[i][k] * MatrixB[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    //limiting test to 20 threads as no speed up apparent after few increases
    if (num_threads < 1 || num_threads > 20) {
        fprintf(stderr, "Number of threads must be between 1 and 20.\n");
        return 1;
    }

    int i, j, k;
    double start_time, end_time;
    double elapsed;
    srand(time(NULL));

    // Initialize the matrices with random values
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            MatrixA[i][j] = 2;
            MatrixB[i][j] = 2;
        }
    }

    start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);

    // Perform matrix multiplication with blocking
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int rows_per_thread = MATRIX_SIZE / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = (thread_id == num_threads - 1) ? MATRIX_SIZE : start_row + rows_per_thread;

        multiply_matrices(start_row, end_row);
    }

    end_time = omp_get_wtime();

    printf("Execution time: %f seconds using %d threads\n", end_time - start_time, num_threads);

    return 0;
}
