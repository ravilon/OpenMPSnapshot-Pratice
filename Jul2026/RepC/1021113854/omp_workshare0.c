/******************************************************************************
* ARQUIVO: omp_workshare1.c
* DESCRIO:
*   Iteracoes de um loop atribuidas a varios threads.
* AUTHOR: Rodrigo Malara - derivado de Blaise Barney  5/99
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N       100

int main (int argc, char *argv[]) {
    int nthreads, tid, i;
    float a[N], b[N], c[N];

    /* Inicializacoes */
    for (i=0; i < N; i++)
        a[i] = b[i] = i * 1.0;

    #pragma omp parallel shared(a,b,c,nthreads) private(i,tid)
    {
        tid = omp_get_thread_num();
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Numero de threads = %d\n", nthreads);
        }
        printf("Thread %d iniciando...\n",tid);

        #pragma omp for
        for (i=0; i<N; i++)
        {
            c[i] = a[i] + b[i];
            printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
        }

    }  /* Fim da secao paralela */
}


