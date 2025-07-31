#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_PUNTOS 100000000

int main() 
{
    int puntos_dentro_circulo = 0;

    // Inicializar la semilla aleatoria
    srand(omp_get_wtime());

    #pragma omp parallel for reduction(+:puntos_dentro_circulo)
    for (int i = 0; i < NUM_PUNTOS; i++) 
    {
        // Generar coordenadas aleatorias dentro del cuadrado
        double x = (double)rand() / RAND_MAX; // Coordenada x entre 0 y 1
        double y = (double)rand() / RAND_MAX; // Coordenada y entre 0 y 1

        // Verificar si el punto está dentro del círculo
        if (x * x + y * y <= 1) 
        {
            puntos_dentro_circulo++;
        }
    }

    // Calcular el valor estimado de Pi
    double pi_estimado = 4.0 * puntos_dentro_circulo / NUM_PUNTOS;

    printf("Valor estimado de Pi: %lf\n", pi_estimado);

    return 0;
}
