#include  <stdio.h>
#include  <omp.h>

int  main(){

     printf("--Fora da regiao paralela--\n");

     omp_set_num_threads(3); //configura no código o numero de threads (sobrepoe a var. ambiente)

     #pragma omp parallel
     {
        int id = omp_get_thread_num();
        int nt = omp_get_num_threads();
        printf("Regiao Paralela - Hello, world da thread %d - %d threads disponiveis\n",id, nt);
     }

     printf("--Fora da regiao paralela--\n");
     return  0;
}
