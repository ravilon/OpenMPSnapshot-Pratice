#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10

int main() 
{
    int valores[N];
    int i; 
    int maximo = -1, minimo = N, suma = 0;
    unsigned long long int producto = 1;    // Para la multiplicación, cuidado, ya que da valores muy grandes que no caben en una palabra.

    // Inicializar valores aleatorios
    srand(omp_get_wtime());
    #pragma omp parallel for
    for (i = 0; i < N; i++) 
    {
        valores[i] = 1 + (rand() % N); // Generamos valores aleatorios entre 1 y 99.
    }

    for (i = 0; i < N; i++) 
    {
        printf("%d\t", valores[i]);
    }
    printf("\n");

    // Realizar la reducción paralela
    #pragma omp parallel for reduction(max:maximo) reduction(min:minimo) reduction(+:suma) reduction(*:producto)
    for (i = 0; i < N; i++) 
    {
        // Calcular el máximo
        if (valores[i] > maximo) 
        {
            maximo = valores[i];
        }
        // Calcular el mínimo
        if (valores[i] < minimo) 
        {
            minimo = valores[i];
        }

        // Calcular la suma
        suma += valores[i];
        
        // Calcular el producto
        producto *= valores[i];
    }

    // Mostramos resultados por consola
    printf("Máximo: %d\n", maximo);
    printf("Mínimo: %d\n", minimo);
    printf("Suma: %d\n", suma);
    printf("Producto: %llu\n", producto);

    return 0;
}
