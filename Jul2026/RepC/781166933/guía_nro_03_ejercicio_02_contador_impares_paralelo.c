#include <stdio.h>
#include <omp.h>

int N = 1;

int contarNumerosPares(int N) 
{
    int count = 0;

    #pragma omp parallel for reduction(+:count)
    for (int i = 1; i <= N; i++) 
    {
        if (i % 2 != 0) 
        {
            count++;
        }
    }

    return count;
}

int main() 
{
    do 
    {
        printf("Ingrese un número entero positivo N, mayor o igual que 1: ");
        scanf("%d", &N);

        if (N <= 0) 
        {
            printf("N debe ser un número entero positivo mayor o igual que 1.\n");
        }
    } while (N <= 0);

    int cantidadPares = contarNumerosPares(N);
    printf("La cantidad de números impares entre 1 y %d es: %d\n", N, cantidadPares);

    return 0;
}
