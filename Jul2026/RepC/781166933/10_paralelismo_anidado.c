#include <stdio.h>
#include <omp.h>

int main()
{
    int num_t;

    num_t = omp_get_num_procs();
    omp_set_num_threads(num_t);
    
    if(!omp_get_nested())
    {
        omp_set_nested(1); // Habilitando paralelismo anidado.
    }

    #pragma omp parallel
    {
        int t_id = omp_get_thread_num();

        printf("Hilo %d, region externa\n", t_id);
        
        #pragma omp parallel num_threads(2) firstprivate(t_id)
        {
            printf("\tHilo %d, region interna. Invocado por Hilo %d\n", omp_get_thread_num(), t_id);
        }
    }

    if(omp_get_nested())
    {
        omp_set_nested(0); // Deshabilitamos paralelismo anidado.
    }

    return 0;
}
