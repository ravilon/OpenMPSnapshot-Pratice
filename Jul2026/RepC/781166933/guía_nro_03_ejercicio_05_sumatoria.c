#include <stdio.h>
#include <limits.h>
#include <omp.h>

int main() 
{
    double suma = 0.0;

    #pragma omp parallel for reduction(+:suma)
    for (unsigned long long int n = 1; n <= INT_MAX; n++) // Usamos maximo entero positivo INT_MAX = 2.147.483.647
    {
        suma += 1.0 / (double) n;
    }

    printf("La sumatoria es: %lf\n", suma);

    return 0;
}
