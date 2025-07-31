#include <stdio.h>
#include <omp.h>
#define N 2

int main()
{
    int i, num_t, t_id;

    num_t = omp_get_num_procs();
    omp_set_num_threads(num_t);

    for (i = 0; i < N; i++)
    {
        printf("\nIteracion %d:\n", i); 

        #pragma omp parallel if (i == 1)
        {
            t_id = omp_get_thread_num();

            printf("Hilo %d, i = %d, dentro if\n", t_id, i); 
        }

        t_id = omp_get_thread_num();
        printf("Hilo %d, i = %d, fuera if\n", t_id, i); 
    }

    return 0;
}
