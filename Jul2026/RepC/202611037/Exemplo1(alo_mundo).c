/*
Compilação: gcc -o cod cod.c -fopenmp
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
int nthreads, tid;
   #pragma omp parallel private(nthreads, tid)
   {
   /* Obtém o número da thread  */
   tid = omp_get_thread_num();
   printf("Alô mundo da thread = %d\n", tid);
   /* Apenas as thread master faz isto */
   if (tid == 0)  {
       nthreads = omp_get_num_threads();
       printf("Número de  threads = %d\n", nthreads);
       }
    } /* Todas as threads se juntam à thread master e terminam */
}

