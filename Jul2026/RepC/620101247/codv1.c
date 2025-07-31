#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    A impressão das soluções está comentada para não interferir na contagem de tempo de resolução
*/

int solve(int **board, int row, int n);
int isSafe(int **board, int row, int col, int n);
//void printBoard(int **board, int n);

int main(int argc, char *argv[]) {

    //Inicializando as variáveis e configurações necessárias
    int n, solutions;
    n = atoi(argv[1]);

    int t;
    if (argc < 3)
        t = 0;
    else
        t = atoi(argv[2]);

    omp_set_num_threads(t);

    //Alocando memória para o tabuleiro
    int **board = (int **) malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        board[i] = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            board[i][j] = 0;
        }
    }

    //Contando o tempo e fazendo a chamada da função que vai resolver
    double start_time = omp_get_wtime();
    solutions = solve(board, 0, n);
    double end_time = omp_get_wtime();

    //Imprimindo as informações
    printf("Soluções: %d\n", solutions);
    printf("Tempo de execução: %fs\n", end_time - start_time);

    //Liberando a memória alocada
    for (int i = 0; i < n; i++) {
        free(board[i]);
    }
    free(board);

    return 0;
}

int solve(int **board, int row, int n) {

    int solutions = 0;

    //Verifica se já atingimos todas as soluções
    if (row == n) {
        //printBoard(board, n);
        return 1;
    }

    //O for que percorre cada coluna é paralelo
    //A variável solutions está com reduction na soma pois é uma variável compartilhada entre as threads
    // que faz uma acumulação dos resultados
    #pragma omp parallel 
    #pragma omp for reduction(+:solutions)
    for (int col = 0; col < n; col++) {

        //Cada linha vai ter uma rainha em sua primeira coluna safe encontrada
        if (isSafe(board, row, col, n)) {

            //Copia toda matriz para a próxima chamada recursiva para 2 threads não operarem na mesma matriz
            int **newBoard = (int **) malloc(n * sizeof(int *));
            for (int i = 0; i < n; i++) {
                newBoard[i] = (int *) malloc(n * sizeof(int));
                for (int j = 0; j < n; j++) {
                    newBoard[i][j] = board[i][j];
                }

            }

            //Coloca a rainha na coluna correspondente e passa para a próxima linha
            newBoard[row][col] = 1;
            solutions += solve(newBoard, row + 1, n);

            //Libera a matriz copiada
            for (int i = 0; i < n; i++) {
                free(newBoard[i]);
            }
            free(newBoard);
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

/*void printBoard(int **board, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
    printf("-----------------------\n");
}*/
