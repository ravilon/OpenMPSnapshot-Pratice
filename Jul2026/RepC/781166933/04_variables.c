#include <stdio.h>
#include <omp.h>
#define N 8

int main()
{
    int var_sh = 1, var_priv = 2, var_fpriv=3;
    int num_t, i; 

    num_t = omp_get_num_procs();
    omp_set_num_threads(num_t);

    printf("Inicialmente: var_sh = %d (%p), var_priv = %d (%p), var_fpriv = %d (%p)\n", var_sh, &var_sh, var_priv, &var_priv, var_fpriv, &var_fpriv); 
    
    #pragma omp parallel shared(var_sh) private(var_priv) firstprivate(var_fpriv)
    {
        int tid = omp_get_thread_num();
        var_priv = 5;
        #pragma omp for
        for (i = 0; i< N; i++)
            printf("Hilo %d: var_sh = %d (%p), var_priv = %d (%p), var_fpriv = %d (%p)\n", tid, var_sh += i, &var_sh, var_priv += i, &var_priv, var_fpriv += 1, &var_fpriv); 
    }

    printf("Finalmente: var_sh = %d (%p), var_priv = %d (%p), var_fpriv = %d (%p)\n", var_sh, &var_sh, var_priv, &var_priv, var_fpriv, &var_fpriv); 

    return 0;
}
