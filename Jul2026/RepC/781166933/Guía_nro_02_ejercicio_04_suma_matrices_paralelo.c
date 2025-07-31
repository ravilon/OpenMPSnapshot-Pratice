#include <stdio.h>
#include <omp.h>

#define N 10

void asignacionMatrices(int matrizA[N][N], int matrizB[N][N], int resultado[N][N])
{
    // Inicializamos las matrices A y B con valores 1.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            resultado[i][j] = 0; // Inicializamos matriz resultado.
            matrizA[i][j] = i+j;  // Puedes cambiar los valores según necesites
            matrizB[i][j] = i-j;  // Puedes cambiar los valores según necesites
        }
    }
}

// Función para sumar dos matrices en paralelo
void sumarMatricesParalelo(int matrizA[N][N], int matrizB[N][N], int resultado[N][N]) 
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            resultado[i][j] = matrizA[i][j] + matrizB[i][j];
        }
    }
}

// Función para imprimir una matriz
void imprimirMatriz(int matriz[N][N]) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            printf("%d\t", matriz[i][j]);
        }
        printf("\n");
    }
}

int main() 
{
    int matrizA[N][N], matrizB[N][N], resultado[N][N];

    // Inicializamos las matrices A, B y resultados.
    asignacionMatrices(matrizA, matrizB, resultado);

    // Sumar las matrices en paralelo
    sumarMatricesParalelo(matrizA, matrizB, resultado);

    // Imprimir las matrices y el resultado
    printf("Matriz A:\n");
    imprimirMatriz(matrizA);

    printf("\nMatriz B:\n");
    imprimirMatriz(matrizB);

    printf("\nResultado de la suma:\n");
    imprimirMatriz(resultado);

    return 0;
}
