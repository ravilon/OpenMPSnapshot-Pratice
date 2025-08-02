#include <stdio.h>
#include <omp.h>
#define N 3

int main()
{
    int i, num_t, t_id;

    num_t = omp_get_num_procs();
    omp_set_num_threads(num_t);

    for (i = 0; i < N; i++)
    {
        printf("\nIteracion %d:\n", i); 

        #pragma omp parallel if (i == 0)
        if (omp_in_parallel())
        {
            printf("Ejecucion en paralelo con %d hilos\n", omp_get_num_threads()); 
        }
        else
        {
            printf("Ejecucion en serie\n"); 
        }
    }

    return 0;
}
