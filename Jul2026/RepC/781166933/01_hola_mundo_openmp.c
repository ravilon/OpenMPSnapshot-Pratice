#include <stdio.h>
#include <omp.h>

int main()
{
    int num_t, tid, nprocs;

    num_t = omp_get_num_procs();

    printf("Nuestro sistema tiene %d nucleos\n\n", num_t); 

    omp_set_num_threads(num_t);
    
    #pragma omp parallel
    {
        tid = omp_get_thread_num();
        nprocs = omp_get_num_threads();
        printf("Hola Mundo. Soy el hilo %d, de un total de %d.\n", tid, nprocs); 
    }
    
    return 0;
}
