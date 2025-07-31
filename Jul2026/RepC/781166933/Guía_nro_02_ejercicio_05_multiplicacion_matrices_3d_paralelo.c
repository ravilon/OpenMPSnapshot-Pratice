#include <stdio.h>
#include <omp.h>

#define N 10

void asignacionMatrices3DParalelo(int matrizA[N][N][N], int matrizB[N][N][N], int resultado[N][N][N])
{
    // Inicializamos las matrices A, B y resultado.
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            for (int k = 0; k < N; k++) 
            {
                resultado[i][j][k] = 0;  // Inicializamos matriz resultado.
                matrizA[i][j][k] = i+j;  // Puedes cambiar los valores según necesites
                matrizB[i][j][k] = i-j;  // Puedes cambiar los valores según necesites
            }
        }
    }
}

// Función para multiplicar dos matrices tridimensionales en paralelo
void multiplicarMatrices3DParalelo(int matrizA[N][N][N], int matrizB[N][N][N], int resultado[N][N][N]) 
{
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            for (int k = 0; k < N; k++) 
            {
                resultado[i][j][k] = 0;
                for (int l = 0; l < N; l++) 
                {
                    resultado[i][j][k] += matrizA[i][j][l] * matrizB[l][j][k];
                }
            }
        }
    }
}

// Función para imprimir una matriz tridimensional
void imprimirMatriz3D(int matriz[N][N][N]) 
{
    for (int i = 0; i < N; i++) 
    {
        // printf("Matriz coordenadas [%d]-[%d]-[%d]:\n", i + 1, j + 1, k + 1);
        for (int j = 0; j < N; j++) 
        {
            for (int k = 0; k < N; k++) 
            {
                printf("%d\t", matriz[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() 
{
    int matrizA[N][N][N], matrizB[N][N][N], resultado[N][N][N];

    // Asignación matrices.
    asignacionMatrices3DParalelo(matrizA, matrizB, resultado);

    // Multiplicar las matrices tridimensionales en paralelo
    multiplicarMatrices3DParalelo(matrizA, matrizB, resultado);

    // Imprimir las matrices y el resultado
    printf("Matriz A:\n");
    imprimirMatriz3D(matrizA);

    printf("\nMatriz B:\n");
    imprimirMatriz3D(matrizB);

    printf("\nResultado de la multiplicacion:\n");
    imprimirMatriz3D(resultado);

    return 0;
}
