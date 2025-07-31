#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

int nSolution(int n, int rank);
int solve(int **board, int row, int n, int rank);
int isSafe(int **board, int row, int col, int n);

int main(int argc, char *argv[]) {

    //Inicializando as variáveis e configurações necessárias
    int n, rank, size, rcv, t;
    int solutions = 0, result = 0;


    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = atoi(argv[1]);

    if (argc < 3)
        t = 0;
    else
        t = atoi(argv[2]);

    omp_set_num_threads(t);

    //Contando o tempo e fazendo a chamada da função que vai resolver todas soluções
    double start_time = omp_get_wtime();

    //i assume papel de rank
    for (int i = 0; i < n; i++) {
        if((i % size) == rank) {
            solutions += nSolution(n, i);
        }
    }

    //os ranks que não são zero enviam as solutions parciais
    if (rank != 0) {
        MPI_Send(&solutions, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*---------------------------*/

    if(rank == 0) {
        for (int i = 1; i < size; i++) {
            //recebe as solutions parciais de todos ranks != 0
            MPI_Recv(&rcv, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            result += rcv;
        }

        double end_time = omp_get_wtime();

        //Imprimindo as informações
        printf("Soluções: %d\n", result + solutions);
        printf("Tempo de execução: %fs\n", end_time - start_time);

    }

    MPI_Finalize();
    return 0; 
}

int nSolution(int n, int rank) {
    int solutions = 0;

    #pragma omp parallel reduction(+:solutions) 
    {

    //Alocando memória para o tabuleiro
    int **board = (int **) malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        board[i] = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            board[i][j] = 0;
        }
    }

    board[0][rank] = 1;

    #pragma omp for
    for (int col = 0; col < n; col++)  {
        if (isSafe(board, 1, col, n)) {
            board[1][col] = 1;
            solutions += solve(board, 2, n, rank);   
            board[1][col] = 0;
            }
        }

        for (int i = 0; i < n; i++) {
            free(board[i]);
        }

        free(board);

    }
    return solutions;
}

int solve(int **board, int row, int n, int rank) {

    int solutions = 0;

    //Verifica se já atingimos todas as soluções
    if (row == n) {
        return 1;
    }

    for (int col = 0; col < n; col++) {

        //Cada linha vai ter uma rainha em sua primeira coluna safe encontrada
        if (isSafe(board, row, col, n)) {

            board[row][col] = 1;
            solutions += solve(board, row + 1, n, rank);
            board[row][col] = 0;

        }

    }

    return solutions;
}

int isSafe(int **board, int row, int col, int n) {
    for (int i = 0; i < row; i++) {
        if (board[i][col])
            return 0;
    }

    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j])
            return 0;
    }

    for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
        if (board[i][j])
            return 0;
    }

    return 1;
}