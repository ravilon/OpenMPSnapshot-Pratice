#include <stdio.h>
#include <omp.h>

#define N 10

void asignacionMatrizVector(int matriz[N][N], int vector[N], int resultado[N])
{
    // Inicializar la matriz y el vector
    for (int i = 0; i < N; i++) 
    {
        resultado[i] = 0;
        vector[i] = i;  // Valores de ejemplo
        for (int j = 0; j < N; j++) 
        {
            matriz[i][j] = i+j;  // Valores de ejemplo
        }
    }
}

// Función para multiplicar matriz por vector en paralelo
int multiplicarMatrizVectorParalelo(int matriz[N][N], int vector[N], int resultado[N]) 
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            resultado[i] += matriz[i][j] * vector[j]; // Multiplicación de la fila i de la matriz por el vector
        }
    }

    return resultado[N];
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

// Función para imprimir arreglo
void imprimirVector(int vector[N])
{
    for (int i = 0; i < N; i++)
    {   
        printf("%d\t", vector[i]);
    }
    printf("\n");
}

int main() 
{
    int matriz[N][N];
    int vector[N];
    int resultado[N];

    // Inicializamos matriz, vector y resultado.
    asignacionMatrizVector(matriz, vector, resultado);

    // Multiplicar matriz por vector en paralelo
    multiplicarMatrizVectorParalelo(matriz, vector, resultado);

    // Imprimir Matriz
    printf("Valores Matriz:\n");
    imprimirMatriz(matriz);

    // Imprimir Vector
    printf("Valores Vector:\n");
    imprimirVector(vector);

    // Imprimir el resultado
    printf("Resultado de la multiplicacion matriz y vector es:\n");
    imprimirVector(resultado);

    return 0;
}
