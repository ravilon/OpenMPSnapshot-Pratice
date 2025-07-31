#include <stdio.h>
#include <omp.h>

int main()
{
    int var_sh = 0, num_t;
    omp_lock_t lock;

    omp_init_lock(&lock);

    num_t = omp_get_num_procs();

    printf("El numeros de nucleos es: %d\n", num_t);

    omp_set_num_threads(num_t);

    #pragma omp parallel shared(var_sh)
    {
        int t_id = omp_get_thread_num();

        omp_set_lock(&lock);
            var_sh = var_sh + t_id;
            printf("Valores var_sh = %d y t_id = %d\n", var_sh, t_id);
        omp_unset_lock(&lock);

        #pragma omp barrier

        #pragma omp master
        {
            printf("Hilo %d: var_sh = %d\n", t_id, var_sh);
        }
    }
            
    omp_destroy_lock(&lock);

    return 0;
}
