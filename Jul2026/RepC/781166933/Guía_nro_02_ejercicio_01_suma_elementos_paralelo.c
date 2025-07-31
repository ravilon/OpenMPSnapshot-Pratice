#include <stdio.h>
#include <omp.h>

#define N 1000

// Función para asignar valores a cada elemento del arreglo.
void asignacionValoresParalelo(int arreglo[N])
{
    // Inicializar el arreglo
    #pragma omp parallel for
    for (int i = 0; i < N; i++) 
    {
        arreglo[i] = i + 1;  // Asignamos i-ésimo valor + 1.
    }   
}

// Función para imprimir arreglo
void imprimirArreglo(int arreglo[N])
{
    for (int i = 0; i < N; i++)
    {   
        printf("%d\t", arreglo[i]);
    }
    printf("\n");
}

// Función para sumar elementos en paralelo
int sumarElementosParalelo(int arreglo[N], int resultado) 
{
    #pragma omp parallel for reduction(+:resultado)
    for (int i = 0; i < N; i++) 
    {
        resultado += arreglo[i];
    }

    return resultado;
}


int main() 
{
    // Declaramos variables.
    int resultado = 0;
    int arreglo[N];

    // Asignar valores al arreglo en paralelo
    asignacionValoresParalelo(arreglo);

    // Imprimir arreglo
    printf("Valores arreglo:\n");
    imprimirArreglo(arreglo);

    // Sumar los elementos en paralelo
    resultado = sumarElementosParalelo(arreglo, resultado);

    // Imprimir resultado
    printf("La suma de los elementos del arreglo es: %d\n", resultado);

    return 0;
}
