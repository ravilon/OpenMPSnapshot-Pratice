#include <stdio.h>
#include <omp.h>

int main()
{
    int var_sh = 0, num_t;

    num_t = omp_get_num_procs();
    omp_set_num_threads(num_t);

    #pragma omp parallel shared(var_sh)
    {
        int t_id = omp_get_thread_num();

        #pragma omp single
        {
            printf("Hilo %d. Actualizando var_sh a 10\n", t_id);
            var_sh = 10;
        }
        /* En este punto una barrera es agregada automáticamente */

        printf("Hilo %d: var_sh = %d\n", t_id, var_sh);
    }
    
    return 0;
}
