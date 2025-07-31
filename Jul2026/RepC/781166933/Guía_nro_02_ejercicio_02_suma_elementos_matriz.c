#include <stdio.h>
#include <omp.h>

#define N 10

void asignacionMatrices(int matriz[N][N])
{
    // Inicializar la matriz
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            matriz[i][j] = i + j;  // Valores de ejemplo
        }
    }
}

// Función para sumar elementos de una matriz en paralelo
int sumarElementosMatrizParalelo(int matriz[N][N]) 
{
    int resultado = 0;
    #pragma omp parallel for reduction(+:resultado)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            resultado += matriz[i][j];
        }
    }

    return resultado;
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
    int matriz[N][N];
    int resultado = 0;

    // Asignación matriz
    asignacionMatrices(matriz);

    // Sumar elementos de la matriz en paralelo
    resultado = sumarElementosMatrizParalelo(matriz);

    // Imprimir las matrices
    printf("Valores matriz:\n");
    imprimirMatriz(matriz);

    // Imprimir resultado
    printf("La suma de los elementos de la matriz es: %d\n", resultado);

    return 0;
}
