//
// Created by Cavide Balki Gemirter on 20.11.2020.
//

#include <stdio.h>
#include <omp.h>
#include "common.c"

#define N               20
#define ITERATION_COUNT 10
#define NUM_OF_THREADS  4
#define VERBOSE         0
#define PRINT_RESULTS   0
#define PRINT_THREADS   0

void assignValues(int matrix[N][N]);

int sumAdjacents(int matrix[N][N], int row, int column);

void playGame(int srcMatrix[N][N], int destMatrix[N][N]);

int getValue(int matrix[N][N], int i, int j);

int main(int argc, char *argv[]) {
    // OpenMP settings
    omp_set_num_threads(NUM_OF_THREADS);
    printf("Program started with %d threads and %d iterations.\n", NUM_OF_THREADS, ITERATION_COUNT);

    // Declarations
    int matrixA[N][N];
    int matrixB[N][N];

    // Start timer
    double start = omp_get_wtime();

    // Initial assignment to matrix - random 0 and 1s
    assignValues(&matrixA);
    if (PRINT_RESULTS > 0) {
        printMatrix("Initial Matrix", N, N, matrixA);
    }

    // matrix A --> matrix B and than matrix B to matrix A
    // so loop count = ITERATION_COUNT / 2
    for (int iteration = 0; iteration < (ITERATION_COUNT / 2); iteration++) {
        if(VERBOSE > 0) {
            printf("Iteration started: %d.\n", iteration * 2 + 1);
        }
        playGame(matrixA, matrixB);
        if(VERBOSE > 0) {
            printf("Iteration started: %d.\n", (iteration + 1) * 2);
        }
        playGame(matrixB, matrixA);
    }

    // Stop timer and log
    logTime("Program finished. \t\t\t", start, omp_get_wtime());
}

// Playing the game
void playGame(int srcMatrix[N][N], int destMatrix[N][N]) {
    int threads_matrix[N][N];

    int i, j;
#pragma omp parallel for private(j) shared(destMatrix) num_threads(NUM_OF_THREADS) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (VERBOSE > 0) {
                printf("Play Game\ti = %d, j = %d, threadId = %d \n", i, j, omp_get_thread_num());
            }
            threads_matrix[i][j] = omp_get_thread_num();
            destMatrix[i][j] = getValue(srcMatrix, i, j);
        }
    }
#pragma omp barrier

    // Print the results
    if (PRINT_RESULTS > 0) {
        printMatrix("Final Matrix :", N, N, destMatrix);
    }

    // Print the threads
    if (PRINT_THREADS > 0) {
        printMatrix("Thread Matrix :", N, N, threads_matrix);
    }
}

int getValue(int matrix[N][N], int i, int j) {
    // 1 paddings from left and right
    // 1 paddings from top and below
    if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
        // Put 0 for padding
        return 0;
    }

    if (sumAdjacents(matrix, i, j) > 5) {
        return 1;
    } else {
        return 0;
    }
}

int sumAdjacents(int matrix[N][N], int row, int column) {
    int sum = 0;

    // Filter has 3 rows : row - 1, row and row + 1
    for (int i = row - 1; i <= row + 1; i++) {
        // Filter has 3 columns : col - 1, col and col + 1
        for (int j = column - 1; j <= column + 1; j++) {
            sum += matrix[i][j];
        }
    }

    return sum;
}

void assignValues(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Assign 0 or 1 - random decision
            matrix[i][j] = rand() % 2;
        }
    }
}