#include <stdio.h>
#include <omp.h>

int N = 1;

int esPrimo(int num) 
{
    if (num <= 1) 
    {
        return 0; // 0 No es primo
    }

    for (int i = 2; i * i <= num; i++) 
    {
        if (num % i == 0) 
        {
            return 0; // No es primo
        }
    }
    return 1; // Es primo
}

int contarNumerosPrimos(int N) 
{
    int count = 0;

    #pragma omp parallel for reduction(+:count)
    for (int i = 1; i <= N; i++) 
    {
        if (esPrimo(i)) 
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
            printf("N debe ser un número entero positivo.\n");
        }
    } while (N <= 0);

    int cantidadPrimos = contarNumerosPrimos(N);
    printf("La cantidad de números primos entre 1 y %d es: %d\n", N, cantidadPrimos);

    return 0;
}
