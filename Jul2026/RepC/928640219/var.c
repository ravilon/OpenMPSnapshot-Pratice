#include <omp.h>
#include <stdio.h>

int main() {
    omp_set_num_threads(4);

    // Shared
    int cont = 0; 
    #pragma omp parallel shared(cont)
    {
        int thread_id = omp_get_thread_num();
        // Condição de corrida
        cont++;
    }
    cont = 0; 
    #pragma omp parallel shared(cont)
    {
        int thread_id = omp_get_thread_num();

        #pragma omp critical
        {
            cont++;
        }
    }

    // Private = não mantém valor de fora na cópia
    cont = 0; // Variável compartilhada
    int local_cont; // Variável privada, cada thread tem uma copia
    #pragma omp parallel private(local_cont) shared(cont)
    {
        local_cont = 0; // Inicializa a variável privada
        local_cont++;   // Incrementa a variável privada
        printf("Thread %d: local_cont = %d\n", omp_get_thread_num(), local_cont);
    
        #pragma omp critical
        {
            cont += local_cont;
        }
    }

    // First private = mantém valor de fora na cópia
    int x = 10;
    #pragma omp parallel firstprivate(x)
    {
        int thread_id = omp_get_thread_num();
        printf("Thread %d: x inicial = %d\n", thread_id, x);
        x += thread_id; // Modifica a cópia privada de x
        printf("Thread %d: x modificado = %d\n", thread_id, x);
    }

    // Last private = última thread a executar muda o valor da variável
    x = 0; // Variável inicializada antes da região paralela
    #pragma omp parallel for lastprivate(x)
    {
        for (int i = 0; i < 10; i++) {
            x = i; // Cada iteração modifica sua própria cópia de x
        }
        // A ultima thread sobreescreve x global
    }
    

    return 0;
}