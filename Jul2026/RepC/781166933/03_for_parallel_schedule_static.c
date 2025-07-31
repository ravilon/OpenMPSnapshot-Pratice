#include <stdio.h>
#include <omp.h>
#define N 10

int main()
{
    int num_t, i, tamanio_tope = 0; 

    num_t = omp_get_num_procs();

    printf("El numeros de nucleos es: %d\n", num_t);

    omp_set_num_threads(num_t);

    tamanio_tope = 5;

    printf("Tamanio del tope = %d\n", tamanio_tope);

    #pragma omp parallel
    {
        #pragma omp for schedule(static, (tamanio_tope))
        for (i = 0; i < N; i++)
            printf("Hilo %d - Ejecutando la iteracion: %d\n", omp_get_thread_num(), i);

    }
    
    return 0;
}
