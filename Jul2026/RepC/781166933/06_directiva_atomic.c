#include <stdio.h>
#include <omp.h>

int main()
{
    int count = 0, num_t;
    
    num_t = omp_get_num_procs();

    printf("Cantidad de nucleos es: %d\n", num_t);
    printf("Valor inicial de count es: %d\n", count);

    omp_set_num_threads(num_t);

    #pragma omp parallel
    {
        #pragma omp atomic
        count++;
    }
    printf("Resultado final de count es: %d\n", count);
}