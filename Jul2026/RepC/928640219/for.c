#include <omp.h>
#include <stdio.h>

int main(){
    omp_set_num_threads(4);

    int v[100];
    for (int i=1; i<=100; i++)
        v[i-1] = i;

    int soma1 = 0;
    int soma2 = 0;
    int soma3 = 0;
    int soma4 = 0;

    #pragma omp parallel
    {
        // Default static,1
        #pragma omp for reduction(+:soma1)
            for (int i=0; i<100; i++) 
            {
                printf("Default: thread = %d, i = %d\n", omp_get_thread_num(), i);
                soma1 = soma1 + v[i];
            }

        // Cada thread executa 4 ciclicamente
        #pragma omp for schedule(static, 4) reduction(+:soma2)
            for (int i=0; i<100; i++)
            {
                printf("Static, 4: thread = %d, i = %d\n", omp_get_thread_num(), i);
                soma2 = soma2 + v[i];
            }

        // Cada thread executa 4 mas a que terminar pega mais        
        #pragma omp for schedule(dynamic, 4) reduction(+:soma3)
            for (int i=0; i<100; i++)
            {
                printf("Dynamic, 4: thread = %d, i = %d\n", omp_get_thread_num(), i);
                soma3 = soma3 + v[i];
            }

        // Tamanho do chunk pode diminuir
        #pragma omp for schedule(guided, 4) reduction(+:soma4)
            for (int i=0; i<100; i++)
            {
                printf("Guided, 4: thread = %d, i = %d\n", omp_get_thread_num(), i);
                soma4 = soma4 + v[i];
            }
    }

    printf("Soma default: %d\n", soma1);
    printf("Soma static: %d\n", soma2);
    printf("Soma dynamic: %d\n", soma3);
    printf("Soma guided: %d\n", soma4);
}