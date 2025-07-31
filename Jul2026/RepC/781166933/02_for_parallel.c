#include <stdio.h>
#include <omp.h>
#define N 10

int main()
{
    int num_t, i; 

    num_t = omp_get_num_procs();

    omp_set_num_threads(num_t);

    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < N; i++)
            printf("Hilo %d - Ejecutando la iteracion: %d\n", omp_get_thread_num(), i);

    }
    
    return 0;
}
