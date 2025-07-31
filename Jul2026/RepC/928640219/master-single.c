#include <omp.h>
#include <stdio.h>

int main(){
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Oi da thread %d\n", id);

        // Apenas executado pela thread master(0)
        #pragma omp master
        {
            printf("Essa mensagem vem da thread mestre!\n");
        }

        // Executa apenas 1x por uma thread qualquer
        #pragma omp single
        {
            printf("Essa mensagem vem da thread %d\n", omp_get_thread_num());
        }
    }

    return 0;
}
